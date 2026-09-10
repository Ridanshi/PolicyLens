"""
Run: uvicorn app:app --reload --port 8000
Dashboard: http://localhost:8000/
"""
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # must run before importing llm_assistant — it reads OPENAI_API_KEY at import time

import store
import llm_assistant

app = FastAPI(title="Warehouse Field Intelligence Backend")


@app.get("/")
def dashboard():
    return FileResponse("static/dashboard.html")


@app.get("/events")
def list_events(bay: str | None = None, risk_level: str | None = None, package_id: str | None = None, date: str | None = None):
    return store.filter_events(bay=bay, risk_level=risk_level, package_id=package_id, date=date)


@app.get("/events/{event_id}")
def get_event(event_id: str):
    event = store.get_event(event_id)
    return event or {"error": "not found"}


@app.post("/events/{event_id}/confirm")
def confirm_event(event_id: str):
    """Supervisor marks an event as confirmed damage — a human review action,
    never inferred automatically (Responsible AI: no automated punitive/definitive
    decisions)."""
    ok = store.confirm_event(event_id)
    if not ok:
        return {"error": "not found"}
    return {"event_id": event_id, "status": "confirmed_damage"}


@app.get("/dates")
def list_dates():
    return store.list_dates()


@app.get("/summary")
def get_summary(date: str | None = None):
    return store.summary(date=date)


class AskRequest(BaseModel):
    question: str
    bay: str | None = None
    risk_level: str | None = None
    date: str | None = None


@app.post("/assistant/ask")
def ask(req: AskRequest):
    events = store.filter_events(bay=req.bay, risk_level=req.risk_level, date=req.date)
    answer = llm_assistant.ask(req.question, events)
    return {"answer": answer, "events_considered": len(events)}


class ShiftSummaryRequest(BaseModel):
    bay: str | None = None
    date: str | None = None


@app.post("/assistant/shift-summary")
def shift_summary(req: ShiftSummaryRequest):
    events = store.filter_events(bay=req.bay, date=req.date)
    summary_text = llm_assistant.shift_summary(events, date=req.date)
    return {"summary": summary_text, "events_considered": len(events)}


app.mount("/static", StaticFiles(directory="static"), name="static")
# Serves evidence clips referenced by events (see store._add_clip_url) so the dashboard can play incident replays.
MEDIA_DIR = os.environ.get("MEDIA_DIR", os.path.join("..", "vision-pipeline", "output_real"))
if os.path.isdir(MEDIA_DIR):
    app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
