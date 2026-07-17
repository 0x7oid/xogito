# Xogito

> **A second brain for large language models.**

<p align="center">
  <img src="docs/LOGO.png" height="400">
</p>

Ask a language model a hard question and you get a fluent answer. Whether you can trust it is another story. A citation might not exist. A guess made early in the reasoning quietly becomes the foundation of the conclusion. When real money or a real decision rides on the answer, "it sounds right" is not enough.

Xogito treats the question as an investigation instead. It breaks the problem down, checks the claims, keeps score of what actually has evidence behind it, and ends with a report where you can see why each conclusion deserves your trust, and which ones don't yet.

## The problems it's built for

- Spending decisions that rest on disputed evidence. "The classic study says do A, a recent paper says the effect isn't real, industry practice says B. Where do we put the budget?"
- Choosing between options when every source you read contradicts the last one.
- Checking whether a widely repeated "best practice" or statistic is actually backed by anything before you build on it.
- Any conclusion you will have to defend in front of a boss, a client, or a committee, where "the AI said so" won't fly.

Quick lookups, brainstorming and writing tasks are not what this is for. A run takes minutes and ends in a report.

## What you get

A report, not a chat transcript. The recommendation comes first, in plain language. Behind it, every supporting claim is labeled by how well it survived checking, and disagreements in the evidence are shown side by side instead of smoothed over. Facts you declared at the start are kept word for word. Guesses the system had to make are shown to you before the run starts, not discovered after.

When the evidence isn't sufficient, the report says exactly that. An honest "this couldn't be established" is treated as a valid result, because a confident answer built on nothing is the failure the whole tool exists to prevent.

## Usage

```bash
pip install -r requirements.txt   # google-genai, python-dotenv, jinja2
# put your Gemini API key in .env as API_KEY
python main.py
```

The intake asks four questions and only the problem statement is required. Reports land in `reports/` as self-contained HTML.

On Windows, run with UTF-8 enabled (`PYTHONUTF8=1`).

## Learn more

How it works under the hood, and why each piece exists:

**→ [docs/DESIGN.html](docs/DESIGN.html)** — *Xogito: The Architecture of Auditable Reasoning*

---

> **Xogito** — from the same root as *cogito*, "I think."
> The difference is what happens **after** the thinking.
