# Backend + Assistant + Dashboard (Hritwik's track)

Reads `events.json` produced by the vision pipeline, serves it via API,
answers supervisor questions grounded only in detected events, and shows a
live dashboard.

## Setup

```bash
cd backend-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill in `OPENAI_API_KEY` in `.env` — this is the one thing I couldn't test live
myself (no real key available), everything else in this README has been
verified working end-to-end against real pipeline output.

`EVENTS_PATH` and `MEDIA_DIR` — for real data from all 7 Godrej pilot clips:
```
EVENTS_PATH=../vision-pipeline/output_real/merged_events.json
MEDIA_DIR=../vision-pipeline/output_real
```
(Run `python ../vision-pipeline/merge_real_events.py` first if
`merged_events.json` doesn't exist yet — combines all per-clip outputs into
one file with distinct bay labels.) Defaults to the hand-written sample at
`../sample_data/sample_events_for_dashboard_dev.json` if unset.

## Run

```bash
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000/ — dashboard auto-refreshes every 5s by polling
`/events`. You don't need the vision pipeline finished to start this: point
`EVENTS_PATH` at a hand-written fake `events.json` (matching
`../shared/events_schema.json`) and build the whole UI/API against that first.

## Endpoints

- `GET /events?bay=&risk_level=&package_id=` — filtered event list
- `GET /events/{event_id}` — single event
- `GET /summary` — counts by risk level and bay, plus `by_bay_risk` (feeds the dashboard's Risk-by-Bay chart)
- `POST /assistant/ask {"question": "..."}` — LLM answer grounded in current events only
- `GET /media/<clip_dir>/clips/<file>.mp4` — serves evidence clips (mounted from vision-pipeline's `output_real/`), backs incident replay

## Built and verified working (via browser, against real detection output)

- **Incident replay** — click "▶ Replay" on any event row, plays the actual evidence clip inline in a modal. Confirmed via network inspection: real 206 Partial Content video streaming, not a placeholder.
- **Risk-by-Bay chart** — stacked bar per bay, segmented by risk level, driven by `/summary`'s `by_bay_risk`.
- **Assistant error handling** — a bad/missing API key now returns a readable in-chat message instead of a raw 500.

## Don't build (per Responsible AI section in the brief)

No automated punitive action off any event. `requires_human_review: true`
events should visually stand out as "needs a human to look," not auto-file
against an employee. Say this explicitly during the pitch — it directly
answers a whole section of the brief almost every other team will skip.

## Don't build (per Responsible AI section in the brief)

No automated punitive action off any event. `requires_human_review: true`
events should visually stand out as "needs a human to look," not auto-file
against an employee. Say this explicitly during the pitch — it directly
answers a whole section of the brief almost every other team will skip.
