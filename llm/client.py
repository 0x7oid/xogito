'''
client.py - the ONE place llm calls happen .

three responsibilities , nothing else :
- ask_llm : one structured call , native http timeout (FIX for the old
  cosmetic ThreadPoolExecutor timeout : the context-manager exit used to
  block on the hung thread , so the timeout never actually freed the
  caller . the google client's own http timeout cancels the request for
  real , so the wrapper is gone entirely)
- the call cache : hash of (model , prompt , schema) -> stored response ,
  json-lines , append-only , same file discipline as calibration .
  deterministic calls (temperature == 0) read it ; voted calls at
  temperature > 0 BYPASS the read (each vote must be a fresh sample)
  but their responses are still appended so an eval/replay path can
  read them later
- ask_llm_voted : self-consistency voting (Wang et al. 2023) . majority
  vote on LABELS only - never average anything . callers get the full
  response of one majority voter back , so rationale text survives

voting logic lives HERE and only here . call sites swap the function
they call , nothing more .
'''

from dotenv import load_dotenv
import hashlib
import json
import os
import shutil
import subprocess
import time

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI as OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from parametres import (
    CALL_TIMEOUT_SECONDS,   # moved to parametres.py - constants live in code , in one place
    CLAUDE_CALL_TIMEOUT_SECONDS,
    CLAUDE_MODEL,
    DEFAULT_MODEL,
    OPENAI_MODEL,
    LLM_CACHE_PATH,
    VOTING_TEMPERATURE,
    TRANSPORT_RETRIES,
    TRANSPORT_BACKOFF_BASE_SECONDS,
    TRANSPORT_MAX_BACKOFF_SECONDS,
)

# transient transport failures worth retrying at the client : overload ,
# rate limit , gateway hiccups . anything else raises immediately - a bad
# request will not get better by asking again
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


def _server_retry_delay_seconds(error):
    # a 429 usually carries the server's OWN retry delay (RetryInfo /
    # retryDelay , e.g. "37s") . when the api states a wait , that wait is
    # authoritative - guessing shorter re-triggers the limiter , guessing
    # longer wastes the run . returns None when the error carries no delay
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in ("retrydelay", "retry_delay") and isinstance(value, str):
                    stripped = value.strip().rstrip("s")
                    try:
                        return float(stripped)
                    except ValueError:
                        return None
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, (list, tuple)):
            for value in node:
                found = walk(value)
                if found is not None:
                    return found
        return None

    details = getattr(error, "details", None)
    if details is None:
        return None
    return walk(details)

load_dotenv()
API_KEY = os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PROVIDER SELECTION : an explicit LLM_PROVIDER in .env always wins .
# otherwise gemini is preferred when its key exists , openai is the
# fallback . claude is explicit-only : it depends on the local claude
# CLI being installed and logged in , which code cannot verify cheaply
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower()

# FIX: the client used to be created at import time , so a missing api
# key crashed EVERY import of every module that transitively touches this
# file - including pure-code paths (checkpoint , report) that never call
# an llm . creation is now lazy : the failure still raises loudly , but
# at the first actual call , where it belongs
_client = None
_openai_client = None
_provider = None
_claude_cli_path = None


def _get_provider():
    global _provider
    if _provider is None:
        if LLM_PROVIDER in ("gemini", "openai", "claude"):
            _provider = LLM_PROVIDER
        elif API_KEY and GEMINI_AVAILABLE:
            _provider = "gemini"
        elif OPENAI_API_KEY and OPENAI_AVAILABLE:
            _provider = "openai"
        else:
            raise ValueError(
                "No LLM provider configured. Set API_KEY (Gemini) or "
                "OPENAI_API_KEY (OpenAI) in .env, or LLM_PROVIDER=claude "
                "with a logged-in claude CLI")
        print(f"[client] provider: {_provider}")
    return _provider


def _default_model():
    # the real default is per provider - a model name only exists on its
    # own api , so resolving it any earlier than call time would bake the
    # wrong provider's name into calls and cache keys
    provider = _get_provider()
    if provider == "claude":
        return CLAUDE_MODEL
    if provider == "openai":
        return OPENAI_MODEL
    return DEFAULT_MODEL


def _get_client():
    global _client
    if _client is None:
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package not installed")
        if not API_KEY:
            raise ValueError("API_KEY not set in .env")
        _client = genai.Client(api_key=API_KEY)
    return _client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed (pip install openai)")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")
        _openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_claude_cli():
    # the claude CLI , wherever it lives . which() first ; the windows
    # default install path as the fallback (the cli is often not on the
    # PATH of whatever shell launched python)
    global _claude_cli_path
    if _claude_cli_path is None:
        found = shutil.which("claude")
        if found is None:
            candidate = os.path.expanduser("~/.local/bin/claude.exe")
            if os.path.exists(candidate):
                found = candidate
        if found is None:
            raise FileNotFoundError(
                "LLM_PROVIDER=claude but the claude CLI was not found - "
                "install Claude Code or fix PATH")
        _claude_cli_path = found
    return _claude_cli_path


# ===========================================================================
# the call cache . loaded once into a dict , appended on every real call .
# a half-written last line (process died mid-append) is skipped , never a
# crash - same read-side tolerance as the calibration log
# ===========================================================================

_cache_by_key = None   # None = not loaded yet


def _cache_key(model, prompt, schema):
    material = json.dumps([model, prompt, schema], sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_cache():
    global _cache_by_key
    if _cache_by_key is not None:
        return
    _cache_by_key = {}
    if not os.path.exists(LLM_CACHE_PATH):
        return
    with open(LLM_CACHE_PATH, "r") as cache_file:
        for line in cache_file:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print("[client] skipped a malformed cache line (interrupted write?)")
                continue
            _cache_by_key[entry["key"]] = entry["response"]


def _append_to_cache(key, response_text):
    _load_cache()
    _cache_by_key[key] = response_text
    entry = {"key": key, "response": response_text}
    with open(LLM_CACHE_PATH, "a") as cache_file:
        cache_file.write(json.dumps(entry) + "\n")


# ===========================================================================
# the single call
# ===========================================================================

def _strip_markdown_fences(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _schema_in_prompt(prompt, schema):
    # for providers that enforce no response schema at the api level ,
    # the schema rides in the prompt verbatim : the model sees the exact
    # field spec the downstream json.loads/validation code expects
    if not schema:
        return prompt
    return (
        prompt
        + "\n\nRespond with a single JSON object (no markdown fences, "
          "no commentary) that conforms exactly to this JSON Schema - "
          "every 'required' field must be present:\n"
        + json.dumps(schema)
    )


def _gemini_call(prompt, schema, model, temperature):
    response = _get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            # FIX: the old code passed timeout= directly to
            # GenerateContentConfig , which has no such field . the real
            # knob is http_options.timeout , in MILLISECONDS - this is the
            # native timeout that makes the executor's thread-pool
            # timeout wrapper unnecessary
            http_options=types.HttpOptions(timeout=CALL_TIMEOUT_SECONDS * 1000),
        )
    )
    return response.text


def _openai_call(prompt, schema, model, temperature):
    # json_object mode does not enforce the schema server-side , so the
    # schema is appended to the prompt - which also satisfies the api's
    # requirement that the prompt mention json when that mode is on
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": _schema_in_prompt(prompt, schema)}],
        "temperature": temperature,
        "timeout": CALL_TIMEOUT_SECONDS,
    }
    if schema:
        kwargs["response_format"] = {"type": "json_object"}
    response = _get_openai_client().chat.completions.create(**kwargs)
    return _strip_markdown_fences(response.choices[0].message.content)


def _claude_env():
    # the subprocess must NOT inherit api-routing overrides from whatever
    # environment launched python (a hosting harness may set
    # ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY , which would silently
    # redirect the cli away from the user's subscription login)
    env = dict(os.environ)
    env.pop("ANTHROPIC_BASE_URL", None)
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def _claude_call(prompt, schema, model, temperature):
    # headless claude cli : prompt on stdin , plain text out , the same
    # schema-in-prompt discipline as openai (the cli enforces no schema) .
    # temperature is not a cli knob - voting still works because cli
    # sampling is non-deterministic by default . no --allowedTools : a
    # pure generation call stays a pure generation call
    result = subprocess.run(
        [_get_claude_cli(), "-p", "--model", model, "--output-format", "text"],
        input=_schema_in_prompt(prompt, schema),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=CLAUDE_CALL_TIMEOUT_SECONDS,
        env=_claude_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude cli exited {result.returncode}: "
            f"{(result.stderr or result.stdout or '').strip()[:300]}")
    text = _strip_markdown_fences(result.stdout or "")
    if not text:
        raise RuntimeError("claude cli returned empty output")
    return text


_PROVIDER_CALLS = {
    "gemini": _gemini_call,
    "openai": _openai_call,
    "claude": _claude_call,
}


def ask_llm(prompt, schema, model=None, temperature=0):
    provider = _get_provider()
    if model is None:
        model = _default_model()

    _load_cache()
    key = _cache_key(model, prompt, schema)

    # temperature > 0 means the caller WANTS a fresh sample (voting) -
    # the cache read is bypassed on purpose . the write below still
    # happens , so the replay path can read these samples later
    if temperature == 0 and key in _cache_by_key:
        return _cache_by_key[key]

    # FIX: a transient 503 ("high demand") used to propagate straight out
    # of ask_llm and kill whichever component made the call - only the
    # executor had retries . the transport belongs to the client , so the
    # client retries transient server errors with backoff for EVERY
    # caller , and still raises loudly when the retries run out
    last_error = None
    for attempt in range(1, TRANSPORT_RETRIES + 1):
        try:
            response_text = _PROVIDER_CALLS[provider](prompt, schema, model,
                                                      temperature)
            _append_to_cache(key, response_text)
            return response_text
        except Exception as error:
            # status code lives at .code on genai errors and .status_code
            # on openai errors ; a claude cli failure carries neither and
            # raises immediately - a bad exit will not get better by
            # asking again
            status_code = getattr(error, "code", None)
            if status_code is None:
                status_code = getattr(error, "status_code", None)
            if status_code not in RETRYABLE_STATUS_CODES:
                raise
            last_error = error
            print(f"[client] transient {status_code} from the api "
                  f"(attempt {attempt}/{TRANSPORT_RETRIES}) - backing off")
            if attempt < TRANSPORT_RETRIES:
                # exponential backoff , capped : 3s, 9s, 27s, 60s, 60s -
                # UNLESS the api stated its own retry delay (429s do) , in
                # which case the server's number wins : it knows its own
                # rate window , our guess does not
                wait_seconds = TRANSPORT_BACKOFF_BASE_SECONDS ** attempt
                if wait_seconds > TRANSPORT_MAX_BACKOFF_SECONDS:
                    wait_seconds = TRANSPORT_MAX_BACKOFF_SECONDS
                server_delay = _server_retry_delay_seconds(error)
                if server_delay is not None:
                    wait_seconds = server_delay + 1  # +1s of margin past the window
                time.sleep(wait_seconds)

    raise last_error


# ===========================================================================
# self-consistency voting . n fresh samples at temperature > 0 , a label
# extracted from each , majority wins . the split (vote counts , descending)
# goes back to the caller so it can be logged to calibration as a label
# tier - no arithmetic ever leaves this file
# ===========================================================================

def ask_llm_voted(prompt, schema, extract_label_fn, n, model=None):
    responses = []
    labels = []
    for _sample_index in range(n):
        response_text = ask_llm(prompt, schema, model=model,
                                temperature=VOTING_TEMPERATURE)
        responses.append(response_text)
        labels.append(extract_label_fn(response_text))

    votes_by_label = {}
    for label in labels:
        if label not in votes_by_label:
            votes_by_label[label] = 0
        votes_by_label[label] += 1

    # majority label = the most-voted one . with an odd n over a binary
    # label a tie is impossible ; if labels are not binary and the top
    # counts tie , the label sampled FIRST among the tied wins - stated
    # rule , never per-case discretion
    majority_label = labels[0]
    majority_count = 0
    for label in labels:
        if votes_by_label[label] > majority_count:
            majority_count = votes_by_label[label]
            majority_label = label

    split = tuple(sorted(votes_by_label.values(), reverse=True))

    # hand back the full response of one majority voter , so the caller
    # still gets rationale text (gap lists , reasons) to work with
    winning_response = ""
    for position in range(len(labels)):
        if labels[position] == majority_label:
            winning_response = responses[position]
            break

    return majority_label, split, winning_response


def vote_split_tier(split):
    # the split as a LABEL tier - this is what crosses the boundary to
    # calibration . unanimous : every vote agreed . majority : a strict
    # majority agreed . split : no strict majority (plurality or tie)
    total_votes = 0
    for count in split:
        total_votes += count
    if split[0] == total_votes:
        return "unanimous"
    if split[0] * 2 > total_votes:
        return "majority"
    return "split"
