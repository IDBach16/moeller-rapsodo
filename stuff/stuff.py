"""
stuff.py -- score Moeller Rapsodo pitches with the Stuff+ model.

The model (train_stuff.py) was trained on MLB Statcast run values using only
the characteristics Rapsodo measures, already converted into Rapsodo's own
units and sign conventions, so scoring is: build the same nine features from a
pitcher's tracked pitches and predict.

TWO HONEST CAVEATS, both by design:

  * THE SCALE IS THE PROGRAM, NOT MLB. High-school pitches sit far below the
    velocity range the model was trained on, so raw MLB-anchored scores would
    compress everyone into "well below average" and stop separating our arms.
    Instead the predicted run values are standardised against ALL Moeller
    tracked pitches: Stuff+ 100 = the average Moeller pitch, every 10 points
    = one standard deviation. Same philosophy as the percentile tab -- rank
    against the program, not the big leagues. The ORDERING within the staff
    comes from what the model learned about shape quality; only the scale is
    local.
  * EXTRAPOLATION IS REAL. The model has never seen an 82mph primary
    fastball, so treat scores as a ranking with error bars, not a scouting
    verdict. The n is reported with every score for exactly that reason.

Left-handers are mirrored into the right-handed frame before scoring, exactly
as in training. A pitcher's fastball reference is his own most-used
fastball-family pitch (FB/SI/CT) in the Rapsodo log.
"""

from __future__ import annotations

import os
from collections import defaultdict

import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "stuff_model.joblib")

FEATURES = ["velo", "spin", "ivb", "hb", "rel_h", "rel_s",
            "velo_diff", "ivb_diff", "hb_diff"]
FB_FAMILY = ("FB", "SI", "CT")

# A pitcher-pitch needs this many scored reps before its Stuff+ is shown as a
# real grade; below it the number is reported but flagged provisional --
# the same threshold the arsenal table uses.
PROVISIONAL_N = 10

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fastball_ref(pitch_list):
    """(velo, ivb, hb) of this pitcher's most-used fastball-family pitch."""
    groups = defaultdict(list)
    for p in pitch_list:
        if p.get("pt") in FB_FAMILY and p.get("velo") is not None:
            groups[p["pt"]].append(p)
    if groups:
        g = max(groups.values(), key=len)
        return (_mean([x.get("velo") for x in g]),
                _mean([x.get("ivb") for x in g]),
                _mean([x.get("hb") for x in g]))
    # No fastball on record: hang the arsenal off his hardest pitch.
    velos = [p["velo"] for p in pitch_list if p.get("velo") is not None]
    return (max(velos) if velos else None, None, None)


def _features(p, ref, mirror):
    """One pitch -> the model's feature row, or None if a field is missing."""
    fb_velo, fb_ivb, fb_hb = ref
    velo, spin = p.get("velo"), p.get("spin")
    ivb, hb = p.get("ivb"), p.get("hb")
    rel_h, rel_s = p.get("rel_h"), p.get("rel_s")
    if None in (velo, spin, ivb, hb, rel_h, rel_s) or fb_velo is None:
        return None
    if mirror:                       # lefties into the RHP frame, as trained
        hb, rel_s = -hb, -rel_s
    f_ivb = fb_ivb if fb_ivb is not None else ivb
    f_hb = (-fb_hb if mirror else fb_hb) if fb_hb is not None else hb
    return [velo, spin, ivb, hb, rel_h, rel_s,
            velo - fb_velo, ivb - f_ivb, hb - f_hb]


def score_pitches(pitch_lists_by_player, throws_by_player):
    """Predicted run value for every scoreable pitch on the staff.

    pitch_lists_by_player: {player_id: [pitch dicts from data.pitches()]}
    throws_by_player:      {player_id: 'R' | 'L' | None}

    Returns {player_id: [(pitch_type, rv), ...]}. Run value is the model's
    output -- LOWER is better for the pitcher. Scaling happens in staff_stuff.
    """
    bundle = _load()
    model = bundle["model"]

    rows, keys = [], []
    for pid, plist in pitch_lists_by_player.items():
        ref = _fastball_ref(plist)
        mirror = throws_by_player.get(pid) == "L"
        for p in plist:
            feats = _features(p, ref, mirror)
            if feats is None or p.get("pt") in (None, "UNK"):
                continue
            rows.append(feats)
            keys.append((pid, p["pt"]))
    if not rows:
        return {}
    preds = model.predict(rows)
    out = defaultdict(list)
    for (pid, pt), rv in zip(keys, preds):
        out[pid].append((pt, float(rv)))
    return dict(out)


def _quantile_map(rows, mlb_quantiles):
    """Map the staff's feature distribution onto the MLB training support.

    Trees extrapolate FLAT below their training range, and a high-school
    staff lives below it -- scored raw, a 78 and an 84 fastball would grade
    identically. Mapping each feature through staff-percentile -> same MLB
    percentile keeps every pitch inside the model's support, so the model
    acts purely as a shape-weighting function and the within-staff ordering
    survives. The output scale is program-relative anyway, so nothing about
    this pretends the kid throws 95.
    """
    import bisect
    n_feat = len(FEATURES)
    cols = [sorted(r[i] for r in rows) for i in range(n_feat)]
    n = len(rows)
    mapped = []
    for r in rows:
        row = []
        for i in range(n_feat):
            # Percentile of this value within the staff (midrank), then the
            # same percentile of the MLB training distribution.
            lo = bisect.bisect_left(cols[i], r[i])
            hi = bisect.bisect_right(cols[i], r[i])
            pct = 100.0 * ((lo + hi) / 2.0) / max(n - 1, 1)
            pct = min(max(pct, 0.0), 100.0)
            grid = mlb_quantiles[FEATURES[i]]
            j = min(int(pct), 99)
            frac = pct - j
            row.append(grid[j] + frac * (grid[j + 1] - grid[j]))
        mapped.append(row)
    return mapped


def score_pitches_mapped(pitch_lists_by_player, throws_by_player):
    """Like score_pitches, but quantile-mapped into MLB support first.
    This is the scoring path staff_stuff uses; raw score_pitches exists for
    tests and for anything already inside the training range."""
    bundle = _load()
    model = bundle["model"]
    mlb_q = bundle["meta"]["mlb_quantiles"]

    rows, keys = [], []
    for pid, plist in pitch_lists_by_player.items():
        ref = _fastball_ref(plist)
        mirror = throws_by_player.get(pid) == "L"
        for p in plist:
            feats = _features(p, ref, mirror)
            if feats is None or p.get("pt") in (None, "UNK"):
                continue
            rows.append(feats)
            keys.append((pid, p["pt"]))
    if not rows:
        return {}
    preds = model.predict(_quantile_map(rows, mlb_q))
    out = defaultdict(list)
    for (pid, pt), rv in zip(keys, preds):
        out[pid].append((pt, float(rv)))
    return dict(out)


def staff_stuff(pitch_lists_by_player, throws_by_player):
    """Stuff+ per (pitcher, pitch type), scaled to the program.

    100 = the average Moeller tracked pitch; +10 = one staff SD better. The
    z-score runs on run value NEGATED, because lower run value is better.
    """
    scored = score_pitches_mapped(pitch_lists_by_player, throws_by_player)
    all_rv = [rv for pitches in scored.values() for _pt, rv in pitches]
    if len(all_rv) < 2:
        return {}
    mean_rv = sum(all_rv) / len(all_rv)
    sd = (sum((v - mean_rv) ** 2 for v in all_rv) / len(all_rv)) ** 0.5
    if sd == 0:
        return {}

    out = {}
    for pid, pitches in scored.items():
        by_pt = defaultdict(list)
        for pt, rv in pitches:
            by_pt[pt].append(rv)
        out[pid] = {
            pt: {
                "stuff_plus": round(100 + 10 * (mean_rv - _mean(rvs)) / sd, 1),
                "n": len(rvs),
                "provisional": len(rvs) < PROVISIONAL_N,
            }
            for pt, rvs in by_pt.items()
        }
    return out
