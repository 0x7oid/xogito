'''
eval_gpqa.py - an independent reasoning benchmark for the local model .

GPQA-Diamond (Rein et al. , NYU - not Google) : 198 graduate-level ,
expert-written multiple-choice science questions designed to be
"Google-proof" , so it measures reasoning rather than recall . widely
used by third-party leaderboards , which avoids the model-card
cherry-picking concern .

WHAT THIS SCRIPT NEEDS BEFORE IT CAN RUN (not done automatically) :
1. pip install datasets            (into the project venv)
2. a Hugging Face account that has ACCEPTED THE TERMS of the gated
   dataset Idavidrein/gpqa , and `huggingface-cli login` done locally -
   the dataset is gated to keep the answers out of training crawls ,
   which is exactly why it stays a fair test
3. the model reachable through llm/client.py . pass the model name as
   the first argument , e.g. :
       python scripts/eval_gpqa.py gemma-3n-e4b

design notes :
- temperature 0 , so reruns hit the call cache in client.py and cost
  nothing - a crashed run resumes for free
- choice order is shuffled DETERMINISTICALLY per question (seeded by the
  question index) , so the correct answer is not always in the same slot
  and reruns stay comparable
- per-question correctness is logged to gpqa_eval_log.jsonl (append-only ,
  same file discipline as calibration) so wrong answers can be
  spot-checked individually , not just counted
- a 4B on-device model is EXPECTED to score far below frontier models
  here . the point is an honest floor , not a flattering number
'''

import json
import os
import random
import sys

# make project-root imports work when run as `python scripts/eval_gpqa.py`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from llm.client import ask_llm
from parametres import PROJECT_ROOT


EVAL_LOG_PATH = os.path.join(PROJECT_ROOT, "gpqa_eval_log.jsonl")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "One of: A, B, C, D",
        },
    },
    "required": ["answer"],
}

CHOICE_LETTERS = ("A", "B", "C", "D")


def load_diamond_split():
    try:
        from datasets import load_dataset
    except ImportError:
        print("the 'datasets' package is not installed. run:")
        print("    pip install datasets")
        sys.exit(1)

    try:
        # gpqa_diamond : 198 questions , small enough for one full pass
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    except Exception as error:
        print(f"could not load the dataset: {type(error).__name__}: {error}")
        print("Idavidrein/gpqa is GATED on Hugging Face - accept its terms")
        print("on the dataset page and run `huggingface-cli login` first.")
        sys.exit(1)

    return dataset


def build_question(row, question_index):
    # one correct + three incorrect answers , shuffled deterministically
    # so the correct letter is not always in the same slot
    choices = [
        (row["Correct Answer"], True),
        (row["Incorrect Answer 1"], False),
        (row["Incorrect Answer 2"], False),
        (row["Incorrect Answer 3"], False),
    ]
    shuffler = random.Random(question_index)   # seeded : reruns identical
    shuffler.shuffle(choices)

    correct_letter = ""
    choice_lines = []
    for position in range(len(choices)):
        letter = CHOICE_LETTERS[position]
        text, is_correct = choices[position]
        choice_lines.append(f"{letter}. {text}")
        if is_correct:
            correct_letter = letter

    prompt = (
        "Answer the multiple-choice question. Reason it through, then "
        "give the letter of the single best answer.\n\n"
        f"QUESTION:\n{row['Question']}\n\n"
        "CHOICES:\n" + "\n".join(choice_lines) + "\n"
    )
    return prompt, correct_letter


def extract_letter(response_text):
    parsed = json.loads(response_text)
    letter = parsed["answer"].strip().upper()
    # code validates the llm's label , always . anything that is not a
    # clean single letter counts as wrong , never as a re-roll
    if letter not in CHOICE_LETTERS:
        return ""
    return letter


def run_eval(model_name):
    dataset = load_diamond_split()
    total = 0
    correct = 0

    with open(EVAL_LOG_PATH, "a") as log_file:
        for question_index, row in enumerate(dataset):
            prompt, correct_letter = build_question(row, question_index)

            try:
                # temperature 0 : deterministic AND cache-hitting on reruns
                response = ask_llm(prompt, ANSWER_SCHEMA, model=model_name,
                                   temperature=0)
                chosen = extract_letter(response)
            except Exception as error:
                chosen = ""
                print(f"[eval] question {question_index} errored: "
                      f"{type(error).__name__}: {error}")

            total += 1
            is_correct = (chosen == correct_letter)
            if is_correct:
                correct += 1

            log_file.write(json.dumps({
                "question_index": question_index,
                "question": row["Question"],
                "chosen": chosen,
                "correct_letter": correct_letter,
                "is_correct": is_correct,
                "model": model_name,
            }) + "\n")

            print(f"[eval] {question_index + 1}/{len(dataset)} "
                  f"running accuracy: {correct}/{total}")

    print("\n=========================================")
    print(f"model: {model_name}")
    print(f"GPQA-Diamond accuracy: {correct}/{total}")
    print(f"per-question log: {EVAL_LOG_PATH}")
    print("(random guessing on 4 choices lands near 1 in 4 - compare "
          "against that floor, not against frontier-model numbers)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/eval_gpqa.py <model-name>")
        print("e.g.:  python scripts/eval_gpqa.py gemma-3n-e4b")
        sys.exit(1)
    run_eval(sys.argv[1])
