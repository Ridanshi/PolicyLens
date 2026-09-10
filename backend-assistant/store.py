"""
In-memory event store, reloaded from events.json whenever the file's mtime
changes. Good enough for a hackathon demo (hundreds of events); swap for
sqlite only if you end up with way more data than expected.
"""
import json
import os
from threading import Lock

EVENTS_PATH = os.environ.get("EVENTS_PATH", os.path.join("..", "sample_data", "sample_events_for_dashboard_dev.json"))
# Evidence clips are written under vision-pipeline/output_real/<clip>/clips/*.mp4 — mounted at /media in app.py.
MEDIA_ROOT_PREFIX = "output_real" + os.sep

_lock = Lock()
_cache = {"mtime": None, "events": []}
# Human-confirmed damage, set only via POST /events/{id}/confirm — never inferred
# automatically. Brief asks for "observed behaviour -> potential risk -> confirmed
# damage" as a distinct, human-reviewed escalation, not something the model claims
# on its own (Responsible AI: avoid automated punitive/definitive decisions).
_confirmed_damage: set[str] = set()


def _add_clip_url(event: dict) -> dict:
    clip_path = event.get("evidence", {}).get("clip_path", "")
    if clip_path.startswith(MEDIA_ROOT_PREFIX):
        event["evidence"]["clip_url"] = "/media/" + clip_path[len(MEDIA_ROOT_PREFIX):].replace(os.sep, "/")
    else:
        event["evidence"]["clip_url"] = None
    return event


def _status_for(event: dict) -> str:
    if event.get("event_id") in _confirmed_damage:
        return "confirmed_damage"
    if event.get("risk_level") in ("high", "critical"):
        return "potential_risk"
    return "observed_behaviour"


def _with_status(event: dict) -> dict:
    event["status"] = _status_for(event)
    return event


def confirm_event(event_id: str) -> bool:
    """Human review action — a supervisor confirming an event actually caused
    damage. Never set automatically."""
    if get_event(event_id) is None:
        return False
    with _lock:
        _confirmed_damage.add(event_id)
    return True


def get_events() -> list[dict]:
    with _lock:
        if not os.path.exists(EVENTS_PATH):
            return []
        mtime = os.path.getmtime(EVENTS_PATH)
        if mtime != _cache["mtime"]:
            with open(EVENTS_PATH) as f:
                raw = json.load(f)
            _cache["events"] = [_add_clip_url(e) for e in raw]
            _cache["mtime"] = mtime
        return _cache["events"]


def filter_events(
    bay: str | None = None,
    risk_level: str | None = None,
    package_id: str | None = None,
    date: str | None = None,
) -> list[dict]:
    events = get_events()
    if bay:
        events = [e for e in events if e.get("bay") == bay]
    if risk_level:
        events = [e for e in events if e.get("risk_level") == risk_level]
    if package_id:
        events = [e for e in events if e.get("package_id") == package_id]
    if date:
        events = [e for e in events if (e.get("timestamp") or "")[:10] == date]
    return [_with_status(dict(e)) for e in events]


def get_event(event_id: str) -> dict | None:
    for e in get_events():
        if e.get("event_id") == event_id:
            return _with_status(dict(e))
    return None


def list_dates() -> list[dict]:
    """Distinct calendar dates present in the data, with event counts —
    powers the dashboard's date picker."""
    counts: dict[str, int] = {}
    for e in get_events():
        d = (e.get("timestamp") or "")[:10]
        if d:
            counts[d] = counts.get(d, 0) + 1
    return [{"date": d, "count": c} for d, c in sorted(counts.items())]


def summary(date: str | None = None) -> dict:
    events = filter_events(date=date) if date else get_events()
    counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for e in events:
        counts[e.get("risk_level", "low")] = counts.get(e.get("risk_level", "low"), 0) + 1
    by_bay = {}
    by_bay_risk = {}
    for e in events:
        bay = e.get("bay", "unknown")
        risk = e.get("risk_level", "low")
        by_bay[bay] = by_bay.get(bay, 0) + 1
        by_bay_risk.setdefault(bay, {"low": 0, "medium": 0, "high": 0, "critical": 0})
        by_bay_risk[bay][risk] = by_bay_risk[bay].get(risk, 0) + 1
    return {"total": len(events), "by_risk": counts, "by_bay": by_bay, "by_bay_risk": by_bay_risk}
