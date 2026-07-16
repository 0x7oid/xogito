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
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import hashlib
import json
import os
import time

from parametres import (
    CALL_TIMEOUT_SECONDS,   # moved to parametres.py - constants live in code , in one place
    DEFAULT_MODEL,
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

load_dotenv()
API_KEY = os.getenv("API_KEY")

# FIX: the client used to be created at import time , so a missing api
# key crashed EVERY import of every module that transitively touches this
# file - including pure-code paths (checkpoint , report) that never call
# an llm . creation is now lazy : the failure still raises loudly , but
# at the first actual call , where it belongs
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=API_KEY)
    return _client


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

def ask_llm(prompt, schema, model=DEFAULT_MODEL, temperature=0):
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
            _append_to_cache(key, response.text)
            return response.text
        except genai_errors.APIError as error:
            if error.code not in RETRYABLE_STATUS_CODES:
                raise
            last_error = error
            print(f"[client] transient {error.code} from the api "
                  f"(attempt {attempt}/{TRANSPORT_RETRIES}) - backing off")
            if attempt < TRANSPORT_RETRIES:
                # exponential backoff , capped : 3s, 9s, 27s, 60s, 60s
                wait_seconds = TRANSPORT_BACKOFF_BASE_SECONDS ** attempt
                if wait_seconds > TRANSPORT_MAX_BACKOFF_SECONDS:
                    wait_seconds = TRANSPORT_MAX_BACKOFF_SECONDS
                time.sleep(wait_seconds)

    raise last_error


# ===========================================================================
# self-consistency voting . n fresh samples at temperature > 0 , a label
# extracted from each , majority wins . the split (vote counts , descending)
# goes back to the caller so it can be logged to calibration as a label
# tier - no arithmetic ever leaves this file
# ===========================================================================

def ask_llm_voted(prompt, schema, extract_label_fn, n, model=DEFAULT_MODEL):
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
