# PolicyLens — AI Video Intelligence for Warehouse Handling

## Status: both code tracks built, tested, and validated against all 7 real Godrej pilot videos

Not just written — actually run end-to-end on real footage, with real bugs found and fixed
along the way (see "What was actually found and fixed" below). What's left is **not code**:
the pitch deck, live user feedback interviews, and rehearsal. See the checklist at the bottom.

## The idea, one paragraph

CV reads the handling pictograms/printed limits already on each carton at
intake and turns them into a machine-readable policy for that specific item
(not a generic global rule). A separate real-time tracking+behavior pipeline
watches that same tracked carton through loading/unloading and flags a
violation only when its behavior breaks *its own* extracted policy. A
physics estimate (drop height → impact energy) grounds the risk score in
evidence instead of a black-box label. A conversational layer explains
incidents to supervisors, answering only from detected events. Novelty is
honest, not oversold: this is a cross-domain transfer of the
sign-recognition→policy→behavior-monitoring pattern from automotive ADAS
into warehouse packaging — genuinely unclaimed there, not an unprecedented
algorithm.

**Vision pipeline** (`vision-pipeline/`) — detection, tracking, pictogram/OCR
policy extraction, behavior heuristics, physics risk estimate. Outputs
`events.json` matching `shared/events_schema.json`. Setup + real-footage
findings: [vision-pipeline/README.md](vision-pipeline/README.md).

**Backend + dashboard** (`backend-assistant/`) — FastAPI backend, LLM
assistant grounded in events, live dashboard with incident replay and a
risk-by-bay chart. Setup: [backend-assistant/README.md](backend-assistant/README.md).

## Run the whole thing end to end

```bash
# 1. Vision pipeline — process all 7 real Godrej pilot videos (already downloaded to sample_data/godrej_pilot/)
cd vision-pipeline
python run_all_pilot_videos.py          # ~1-2 hours on CPU, see vision-pipeline/README.md for why
python merge_real_events.py             # combines all 7 outputs into one multi-bay dataset

# 2. Backend + dashboard
cd ../backend-assistant
# .env: EVENTS_PATH=../vision-pipeline/output_real/merged_events.json, MEDIA_DIR=../vision-pipeline/output_real, OPENAI_API_KEY=<real key>
uvicorn app:app --port 8000
# open http://localhost:8000
```

## What was actually found and fixed (real bugs, real footage — not hypothetical)

**Environment/infra:**
- Windows `lap` package permission failure → `pip install --user`
- CUDA-torch pagefile crash + Windows long-path pip failure → CPU-only torch in a dedicated venv
- OpenBLAS OOM on `import easyocr` under real low-RAM conditions → capped BLAS threads (baked into `main.py`)

**Detection/tracking:**
- Evidence clips were referenced in `events.json` but never actually written → real `ClipRecorder` (rolling pre/post-event buffer)
- Canny-edge box detector picked up shadows/floor texture as "boxes" → rectangularity check + confirmation-gate (a track needs 5 consistent hits before it's trusted)
- Tried background subtraction (MOG2) to fix the above — made it *worse*, because Godrej's footage is handheld, not fixed-camera, and MOG2 assumes a static background. Reverted.
- Frame downscaling for speed broke detection twice over (absolute pixel-area threshold didn't scale with resolution; then broke track continuity even after fixing that) — reverted, full resolution is correctness-safe, just slow on CPU

**Behavior heuristics:**
- `dragged` had no cooldown — one real drag on a longer clip re-fired 47-139 times → 3-second per-track cooldown
- `thrown` was referenced in the risk engine but never actually implemented (real gap — 2 of the 7 clips are specifically about throwing) → implemented, then tightened after camera-shake was initially being misread as throwing motion
- `dragged` also missed entirely on one clip (`kd_packets`) — root cause: a **global** "floor line" (whether a fixed fraction of frame height, or an adaptive fraction of the deepest point seen) doesn't generalize across Godrej's 7 clips, each shot handheld at a different distance/angle. Fixed by switching to a **per-track relative range** ("near ground" = within the bottom 35% of *this object's own* observed vertical travel) — no scene-wide assumption needed. Real, honest tradeoff: fixed the miss (0→2 events) but reintroduced some `dragged` noise on one other clip (8→19 on `stepping_on`, still far below the original 139 pre-cooldown-fix). Documented in `behavior_rules.py`.

**Backend/dashboard:**
- Incident replay and the risk-by-bay chart are real, not mockups — verified in-browser: clicking Replay fires an actual `206 Partial Content` video stream request against the real evidence clip file.
- LLM assistant's error handling was a raw 500 on API failure — now returns a readable in-chat message. Full request/response flow verified working; the actual OpenAI call itself couldn't be tested live (no real API key available) — you'll need to supply your own and do one final live check.

## Known limitations, stated honestly (for the pitch's own credibility, and for your own planning)

- Camera is handheld across all 7 clips — contaminates velocity-based heuristics (drop/drag/throw thresholds) to some degree. The architecturally correct fix is optical-flow-based camera-motion compensation before computing object velocity; not built, flagged as the real next step if there's time.
- Box detection is still the Canny/contour placeholder, not a trained model — works, but noisier than a fine-tuned YOLO box detector would be. See vision-pipeline/README for how to swap one in.
- Event counts on longer clips are higher than a human would flag by hand — some is real (repeated handling across 30-49s), some is residual noise. Worth a visual spot-check against the actual footage before the demo, which only a human watching the video can really do.
- `dragged` detection is a bit more permissive after the kd_packets fix (see above) — one clip that's purely about dropping now shows 1 stray `dragged` event.

## Submission checklist (from the brief, verbatim requirements)

**1. Working Prototype** — platform-agnostic app demonstrating:
- [x] Video ingestion
- [x] Object detection/tracking
- [x] Behaviour identification — `dropped`, `dragged`, `over_stacked`, `wrong_orientation`, `stepped_on`, `thrown` all real and validated on real footage (6 of the brief's ≥10 target; see vision-pipeline/README for cheap additions to hit 10 if there's time — `rough_handling`, `unstable_placement`, `wet_zone_handling`)
- [x] Risk classification
- [x] Incident visualization (dashboard + replay)
- [x] AI-generated explanation/recommendation (assistant — needs a real API key for the final live check)

**2. Presentation deck — min 5, max 6 slides — NOT YET DONE, this is on you two now:**
- [ ] Slide 1: App name, team name, members, one-line value proposition
- [ ] Slide 2: Problem/solution/user journey — show `Warehouse Activity → Video → AI Understanding → Risk Detection → Alert → Intervention → Prevention`, include the supervisor/operator journey
- [ ] Slide 3: Technical architecture & stack — CV, AI/ML, LLM, video processing, edge/cloud infra, frontend, data storage
- [ ] Slide 4: Screenshots + demo — original video, detected objects, behavior detection, risk classification, incident replay, dashboard, AI assistant; short demo video, **3-5 representative scenarios** (use the real annotated outputs in `vision-pipeline/output_real/*/annotated_output.mp4` — already real, already recorded)
- [ ] Slide 5: Impact & user validation — document actual feedback from people you show it to (supervisor/operator/logistics/quality/safety roles suggested), what you changed based on it. **This needs real humans, can't be done from here.**
- [ ] Slide 6 (optional): Future roadmap / scalability / business applications — use the brief's own "warehouse → factory → distribution centre → retail → field service" framing

**3. Team size**: 3-5 (running at 3)

**4. Input videos**: already downloaded to `sample_data/godrej_pilot/`, source: https://drive.google.com/drive/folders/1MG90LJowfSZ2qz5woDyarHdzskbCRLzP

**Don't forget — Responsible AI section of the brief** (most teams will skip this, mentioning it explicitly is a differentiator): no automated punitive action, `requires_human_review` flag shown clearly not auto-acted-on (dashboard already does this), explain false-positive handling, state data retention/privacy stance. Put a line about this on slide 2 or 5.
