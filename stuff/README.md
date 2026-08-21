# Moeller Stuff+

A Stuff+ grade for every (pitcher, pitch type) on the staff, computed from the
same Rapsodo data the dashboard already shows. Modelled on
[tjStuff+](https://github.com/tnestico/tjstuff_plus) (MIT): a gradient-boosted
model trained on MLB Statcast run values, using only a pitch's physical
characteristics.

## How it works

1. **`train_stuff.py`** (one-off) downloads a season of Statcast pitches
   (2024 April–August train, September holdout, ~577K/112K pitches), converts
   them into Rapsodo's exact units and sign conventions (pitcher-view inches,
   lefties mirrored into the right-handed frame), and trains
   `HistGradientBoostingRegressor` to predict each pitch's run value from:
   velocity, spin, IVB, HB, release height/side, and the pitch's velocity/IVB/HB
   differentials off the pitcher's own primary fastball. Velocity carries a
   monotonic constraint (harder is never worse, holding shape). Extension and
   accelerations are deliberately absent — the Pitching 2.0 unit doesn't
   measure them.
2. **`stuff.py`** scores Moeller pitches. Because a high-school staff sits
   below the model's training range (where trees extrapolate flat), each
   feature is quantile-mapped onto the MLB training distribution first — the
   model is used strictly in-support as a *shape-weighting function*, and the
   within-staff ordering survives.
3. **The scale is the program, not MLB.** Stuff+ 100 = the average Moeller
   tracked pitch; every 10 points = one staff standard deviation. Same
   philosophy as the percentile tab: rank against the program. Fewer than 10
   scored reps = provisional, same threshold as the arsenal table.

## Validation

* `test_stuff.py` — 14 invariant checks (ordering, mirroring, hygiene,
  scaling). A retrained model that fails the ordering checks must not ship.
* Holdout (Sept 2024): per-pitch RMSE barely beats the mean baseline — correct
  and expected; pitch-level run value is mostly batted-ball luck. The signal
  is in aggregation.
* **External check on our own field data:** fastball Stuff+ vs charted GAME
  results for the six pitchers with a real FB grade and 30+ charted game
  fastballs — **r = 0.59 with game whiff%, r = −0.11 with game strike%**. The
  grades predict swings-and-misses and know nothing about command, which is
  exactly what Stuff+ is supposed to be.

## Files

| File | What |
|---|---|
| `train_stuff.py` | one-off trainer (caches `statcast_2024.csv`, ~250MB, not committed) |
| `stuff.py` | scoring module — `staff_stuff(pitch_lists, throws)` |
| `stuff_model.joblib` | the fitted pipeline + metadata (features, MLB quantile grids) |
| `test_stuff.py` | invariant checks — run before shipping a retrained model |
| `score_moeller.py` | print the full staff table from the database |

## Honest limits

* The model has never seen a high-school pitch. The quantile mapping keeps the
  ordering meaningful, but treat grades as a ranking with error bars, not a
  scouting verdict — that's why every score carries its n.
* ~21% of charted game pitches are logged only as "Breaking Ball" and can't be
  matched to a Rapsodo pitch type; the validation above uses fastballs only.
