"""
Conversational layer over the detected events. Answers are grounded ONLY in
the events passed as context — the brief explicitly asks for this
("should respond using the events detected... rather than inventing
information"), and it's also the honest thing to do.

Set OPENAI_API_KEY in a .env file (see .env.example). Works with any
OpenAI-compatible provider (Groq, etc.) by also setting OPENAI_BASE_URL and
OPENAI_MODEL — Groq's endpoint is OpenAI-schema-compatible, just a different
base_url and model catalog. To swap to Gemini's own SDK instead: replace the
client call below with google-generativeai, keep the same system prompt and
context-building logic.
"""
import os
import json
from openai import OpenAI, OpenAIError

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Groq free tier caps this model at 8000 tokens/request — dumping all ~264
# events pretty-printed blew past that (67k+ tokens). Cap detail rows and
# fall back to aggregate counts for anything older than the cap.
MAX_DETAIL_EVENTS = 25

SYSTEM_PROMPT = """You are a warehouse safety assistant. You answer questions from a supervisor
about handling-risk events detected by a computer vision system.

Rules:
- Answer ONLY using the event data provided below. Never invent numbers, counts, or incidents not present in the data.
- ANY counting or "most common" question — behavior, bay, risk level, status, or a combination — MUST be answered from "aggregate_counts", never by counting rows in "recent_events". recent_events is capped and will silently undercount. Use "by_behavior" for "most common behavior" questions, "by_bay" for bay volume, "by_bay_and_risk_level" for bay+risk, "by_status" for observed_behaviour/potential_risk/confirmed_damage counts. Only use "recent_events" to cite a specific event_id or incident detail, never for a count.
- If the data doesn't contain enough information to answer, say so plainly instead of guessing.
- Every risk claim about a specific incident must be traceable to a specific event_id from "Recent events".
- Keep answers short and concrete, supervisors are busy."""


def _aggregate(events: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for e in events:
        k = e.get(key) or "unknown"
        counts[k] = counts.get(k, 0) + 1
    return counts


def _cross_tab(events: list[dict], key_a: str, key_b: str) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for e in events:
        a = e.get(key_a) or "unknown"
        b = e.get(key_b) or "unknown"
        counts.setdefault(a, {})
        counts[a][b] = counts[a].get(b, 0) + 1
    return counts


def _build_context(events: list[dict]) -> dict:
    recent = sorted(events, key=lambda e: e.get("timestamp") or "", reverse=True)[:MAX_DETAIL_EVENTS]
    trimmed = [
        {
            "event_id": e["event_id"],
            "bay": e.get("bay"),
            "package_id": e.get("package_id"),
            "detected_behavior": e.get("detected_behavior"),
            "risk_level": e.get("risk_level"),
            "status": e.get("status"),
            "policy": e.get("policy"),
            "evidence": e.get("evidence"),
            "timestamp": e.get("timestamp"),
        }
        for e in recent
    ]

    return {
        "total_matching_events": len(events),
        "aggregate_counts": {
            "by_bay": _aggregate(events, "bay"),
            "by_risk_level": _aggregate(events, "risk_level"),
            "by_behavior": _aggregate(events, "detected_behavior"),
            "by_bay_and_risk_level": _cross_tab(events, "bay", "risk_level"),
            "by_status": _aggregate(events, "status"),
        },
        "recent_events": trimmed,
        "note": (
            f"recent_events shows the {len(trimmed)} most recent of {len(events)} matching events; "
            "use aggregate_counts for totals across all of them."
            if len(events) > MAX_DETAIL_EVENTS else
            "recent_events contains all matching events."
        ),
    }


def ask(question: str, events: list[dict]) -> str:
    if not events:
        return "No events recorded yet for that query — check the filters or whether the pipeline has run on this footage."

    context = _build_context(events)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Events data:\n{json.dumps(context)}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"Assistant unavailable right now (API error: {e}). Check OPENAI_API_KEY in .env."


SHIFT_SUMMARY_PROMPT = """Summarize this shift/day for a warehouse supervisor in 3-5 sentences.
Cover: total events, the riskiest bay and why, the most common behavior, and how many
events still need human review. For the review count, use aggregate_counts.by_status
(potential_risk + confirmed_damage) — it covers ALL matching events. Do NOT count
statuses within recent_events, that list is capped and will undercount. Use only the
aggregate_counts and recent_events given — do not invent a number that isn't there."""


def shift_summary(events: list[dict], date: str | None = None) -> str:
    if not events:
        label = f"on {date}" if date else "yet"
        return f"No events recorded {label} — check whether the pipeline has run on this footage."

    context = _build_context(events)
    label = f"for {date}" if date else "across all recorded data"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Events data {label}:\n{json.dumps(context)}\n\n{SHIFT_SUMMARY_PROMPT}"},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"Shift summary unavailable right now (API error: {e}). Check OPENAI_API_KEY in .env."
