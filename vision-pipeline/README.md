# Vision Pipeline (Your track)

Detects cartons + people, reads each carton's own printed handling policy,
watches its handling behavior, cross-references behavior against that specific
item's policy, and outputs risk-scored events as JSON for the backend to consume.

## Setup

```bash
cd vision-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Before first run

1. **Icon references** — create `assets/icons/<icon_name>/` folders:
   `fragile`, `this_side_up`, `keep_dry`, `stack_limit`. Drop 3-5 clear photos
   or clean symbol renders of each icon per folder (crop tight to just the
   icon). No training needed, this is ORB feature matching against these refs.
   Source real box photos from your own house/campus mailroom/Godrej's pilot
   videos (freeze-frame a clear label shot), or search "ISO 7000 fragile
   symbol" etc. for clean reference renders.

2. **Box detector** — COCO has no "carton" class. Two options, pick based on
   time left:
   - Fast path (recommended given the timeline): omit `--box-model`, the
     contour-based fallback in `detection_tracking.py` kicks in. Good enough
     for a controlled single-bay demo with clean backgrounds.
   - Better path: pull a pallet/box dataset from Roboflow Universe (search
     "pallet detection" or "carton detection"), export YOLO format, run
     `yolo detect train data=data.yaml model=yolov8n.pt epochs=50`, pass the
     resulting `best.pt` as `--box-model`.

3. **Calibrate meters-per-pixel** — measure one real carton's height in
   metres, freeze a frame, measure its height in pixels, divide. Pass as
   `--meters-per-pixel`. This is an approximation — say so on the pitch slide,
   it's the "don't overclaim precision" point the brief itself asks for.

## Run

```bash
python main.py --video ../sample_data/clip1.mp4 --bay bay-1 --meters-per-pixel 0.004
```

Outputs to `output/`: `events.json` (hand this file/format to Hritwik),
`annotated_output.mp4` (boxes + risk labels, use in the demo slide), and
`clips/` (short evidence clips per flagged event).

## Files

- `detection_tracking.py` — person + carton detection/tracking
- `pictogram_ocr.py` — reads icons + printed numbers off a carton crop → policy dict
- `behavior_rules.py` — trajectory/geometry heuristics → behavior label (no trained action classifier, no dataset exists for this — documented tradeoff)
- `depth_risk.py` — pixel drop → drop height → impact energy (physics estimate)
- `risk_engine.py` — behavior + THAT item's policy → risk verdict (this is the core novel-angle logic)
- `main.py` — glues it all together, writes `events.json`

## Tuning before the demo

Thresholds in `behavior_rules.py` (`DROP_VELOCITY_PX_PER_FRAME` etc.) are
starting guesses. Run against Godrej's pilot videos (link in the brief) and
adjust until false positives/negatives look reasonable — budget real time for
this, it's the part most likely to need iteration.
