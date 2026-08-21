"""
data.py -- every query the Rapsodo app makes.

The database does the work; the app only renders. Nothing here returns raw pitch
rows to a template except the plot series, which need one point per pitch.

Percentiles are computed against other Moeller pitchers ON THE SAME LEVEL. A
freshman ranked against varsity arms is measuring age, not development.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select

import db
import metrics

SOURCE = "rapsodo"

# Long-format metric keys -> the short names used in templates and plots.
FIELDS = {
    "velocity": "velo",
    "spin_rate": "spin",
    "true_spin": "true_spin",
    "spin_efficiency": "eff",
    "induced_vertical_break": "ivb",
    "horizontal_break": "hb",
    "total_break": "total_break",
    "release_height": "rel_h",
    "release_side": "rel_s",
    "spin_axis": "axis",
    "gyro_degree": "gyro",
    "plate_side": "px",
    "plate_height": "pz",
    "is_strike": "strike",
}

# What the Percentile tab can rank. Extension is deliberately absent: the
# Pitching 2.0 unit does not measure it, and an empty column reads as a bug.
PERCENTILE_METRICS = [
    ("velo", "Velocity", "mph", 1, True),
    ("spin", "Spin rate", "rpm", 0, True),
    ("eff", "Spin efficiency", "%", 0, True),
    ("ivb", "Induced vertical break", "in", 1, None),
    ("hb", "Horizontal break", "in", 1, None),
    ("rel_h", "Release height", "ft", 2, None),
    ("rel_s", "Release side", "ft", 2, None),
]

PITCH_ORDER = ["FB", "SI", "CT", "SL", "CB", "CH", "SP"]

# A pitch type below this in a pitcher's own log is shown but marked provisional:
# Rapsodo auto-classifies and nobody has validated a single pitch, so three reps
# of a "curveball" is a guess, not an offering.
PROVISIONAL_N = 10


def _order(pt):
    return PITCH_ORDER.index(pt) if pt in PITCH_ORDER else 99


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _round(v, dp=1):
    if v is None:
        return None
    return round(v, dp) if dp else int(round(v))


# Arm slot is inferred from FASTBALL SPIN AXIS, not from release-point geometry.
#
# The geometric version (release point vs an assumed shoulder position) was tried
# first and produced garbage: release side measures where the ball came out
# relative to the FIELD's centerline, so it confounds arm slot with where the kid
# stands on the rubber. A pitcher who sets up on the 1B side releases near zero
# and read as perfectly over the top; a third of the staff pinned at an identical
# 90 degrees.
#
# A fastball's spin axis tracks the arm that threw it -- over the top backspins
# near 12:00, three-quarters tilts toward 1:30 (11:30 for a lefty), sidearm lies
# near 3:00 -- and it does not care where he stands. Cross-checked against the
# geometry for pitchers the geometry DID handle cleanly (Medinger: 30.1 by tilt
# vs 30.4 geometric), and it separates the over-the-top group the geometry
# saturated. Known deviation between tilt and true slot is ~10 degrees; the same
# for everyone, so within-staff comparison holds.
#
# Bonus: the axis side is a more reliable handedness read than release side --
# axis < 180 is a right-hander's tilt, > 180 a lefty's.
FB_FAMILY = ("FB", "SI", "CT")   # pitches whose axis reflects the slot
MIN_AXIS_N = 10
DEFAULT_HEIGHT_IN = 72


def circular_mean(degs):
    """Mean of angles in degrees. A naive average of 355 and 5 is 180."""
    import math
    if not degs:
        return None
    x = sum(math.cos(math.radians(d)) for d in degs)
    y = sum(math.sin(math.radians(d)) for d in degs)
    if x == 0 and y == 0:
        return None
    return math.degrees(math.atan2(y, x)) % 360


def arm_angle_from_axis(axis_mean):
    """(degrees above horizontal, side) from a mean fastball spin axis.

    0 on the axis dial is 12:00; deviation from it is the arm's tilt off
    vertical, so slot = 90 minus that deviation. Past 3:00 goes negative
    (submarine). Side: 'R' tilts clockwise (axis < 180), 'L' counter.
    """
    if axis_mean is None:
        return None, None
    dev = min(axis_mean, 360.0 - axis_mean)
    side = "R" if axis_mean < 180 else "L"
    return round(90.0 - dev, 1), side


def pitcher_slot(pitch_list):
    """One slot per pitcher, from his fastball family. Slot is a property of the
    delivery, so it is a single number -- but only fastball-shaped spin reflects
    it (a slider's gyro axis says nothing about the arm)."""
    for pt in FB_FAMILY:
        axes = [p["axis"] for p in pitch_list
                if p.get("pt") == pt and p.get("axis") is not None]
        if len(axes) >= MIN_AXIS_N:
            angle, side = arm_angle_from_axis(circular_mean(axes))
            return {"angle": angle, "side": side, "from_pt": pt, "n": len(axes)}
    return None


def slot_name(angle):
    """The words a coach would use for that number."""
    if angle is None:
        return None
    if angle >= 70:
        return "over the top"
    if angle >= 45:
        return "high three-quarters"
    if angle >= 20:
        return "three-quarters"
    if angle >= 0:
        return "low three-quarters"
    if angle >= -20:
        return "sidearm"
    return "submarine"


def ordinal(n):
    """1st, 2nd, 3rd, 4th... 11th-13th are the exceptions that catch people out."""
    if n is None:
        return None
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }".replace(" ", "")


def tilt(degrees):
    """Spin axis in degrees -> the clock face coaches actually read.

    Rapsodo reports 0-359. 0 degrees is 12:00 and it advances clockwise, so
    every 30 degrees is an hour.
    """
    if degrees is None:
        return None
    total_minutes = (float(degrees) / 360.0) * 720.0     # 12 hours on the face
    hour = int(total_minutes // 60) % 12 or 12
    minute = int(round(total_minutes % 60))
    if minute == 60:
        hour, minute = (hour % 12) + 1, 0
    return f"{hour}:{minute:02d}"


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

def pitchers(engine, season=None):
    """Everyone with Rapsodo data, with their level and pitch count."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(db.players.c.id, db.players.c.slug,
                   db.players.c.first_name, db.players.c.last_name,
                   db.players.c.throws,
                   db.player_seasons.c.level, db.player_seasons.c.season,
                   db.player_seasons.c.height_in,
                   db.sessions.c.id.label("session_id"))
            .select_from(
                db.players
                .join(db.sessions, db.sessions.c.player_id == db.players.c.id)
                .outerjoin(db.player_seasons,
                           db.player_seasons.c.player_id == db.players.c.id))
            .where(db.sessions.c.source == SOURCE)).all()

    out = {}
    for r in rows:
        p = out.setdefault(r.id, {
            "id": r.id, "slug": r.slug,
            "name": f"{r.first_name} {r.last_name}",
            "level": r.level or "unassigned",
            "throws": r.throws, "height_in": r.height_in,
            "season": r.season, "sessions": set()})
        p["sessions"].add(r.session_id)
    for p in out.values():
        p["sessions"] = len(p["sessions"])
    order = {"varsity": 0, "jv": 1, "freshman": 2, "unassigned": 3}
    return sorted(out.values(),
                  key=lambda p: (order.get(p["level"], 9), p["name"]))


# ---------------------------------------------------------------------------
# Pitches
# ---------------------------------------------------------------------------

def _pitch_rows(conn, player_id=None):
    q = (select(db.pitch_metrics.c.player_id, db.pitch_metrics.c.session_id,
                db.pitch_metrics.c.seq, db.pitch_metrics.c.pitch_type,
                db.pitch_metrics.c.metric_key, db.pitch_metrics.c.value,
                db.sessions.c.session_date)
         .select_from(db.pitch_metrics.join(
             db.sessions, db.sessions.c.id == db.pitch_metrics.c.session_id))
         .where(db.sessions.c.source == SOURCE))
    if player_id is not None:
        q = q.where(db.pitch_metrics.c.player_id == player_id)
    return conn.execute(q).all()


def pitches(engine, player_id):
    """One dict per pitch, pivoted out of the long-format table."""
    with engine.connect() as conn:
        rows = _pitch_rows(conn, player_id)
    by_pitch = defaultdict(dict)
    for r in rows:
        p = by_pitch[(r.session_id, r.seq)]
        p["pt"] = r.pitch_type or "UNK"
        p["date"] = str(r.session_date)
        p["session_id"] = r.session_id
        short = FIELDS.get(r.metric_key)
        if short:
            p[short] = r.value
    # A pitch with no velocity is a failed radar track that slipped through.
    return [p for p in by_pitch.values() if p.get("velo") is not None]


def arsenal(pitch_list, height_in=None):
    """Per-pitch-type summary -- the table every tab sits under."""
    by_pt = defaultdict(list)
    for p in pitch_list:
        by_pt[p["pt"]].append(p)
    total = len(pitch_list) or 1

    out = []
    for pt, g in by_pt.items():
        velos = [x["velo"] for x in g if x.get("velo") is not None]
        strikes = [x["strike"] for x in g if x.get("strike") is not None]
        # Slot only for fastball-family rows: a breaking ball's gyro-heavy axis
        # says nothing about the arm that threw it.
        angle = None
        if pt in FB_FAMILY:
            axes = [x.get("axis") for x in g if x.get("axis") is not None]
            if len(axes) >= MIN_AXIS_N:
                angle, _side = arm_angle_from_axis(circular_mean(axes))
        out.append({
            "arm_angle": angle,
            "slot": slot_name(angle),
            "pt": pt,
            "label": metrics.PITCH_TYPE_LABELS.get(pt, "Unlabeled"),
            "n": len(g),
            "usage": _round(100.0 * len(g) / total),
            "velo": _round(mean(velos)),
            "max_velo": _round(max(velos)) if velos else None,
            "spin": _round(mean([x.get("spin") for x in g]), 0),
            "true_spin": _round(mean([x.get("true_spin") for x in g]), 0),
            "eff": _round(mean([x.get("eff") for x in g]), 0),
            "ivb": _round(mean([x.get("ivb") for x in g])),
            "hb": _round(mean([x.get("hb") for x in g])),
            "total_break": _round(mean([x.get("total_break") for x in g])),
            "rel_h": _round(mean([x.get("rel_h") for x in g]), 2),
            "rel_s": _round(mean([x.get("rel_s") for x in g]), 2),
            "tilt": tilt(mean([x.get("axis") for x in g])),
            "gyro": _round(mean([x.get("gyro") for x in g]), 0),
            "strike_pct": _round(100.0 * mean(strikes)) if strikes else None,
            # Rapsodo auto-classifies and nothing has been coach-validated, so a
            # handful of reps is a guess rather than a pitch he has.
            "provisional": len(g) < PROVISIONAL_N,
        })
    out.sort(key=lambda a: (_order(a["pt"]), -a["n"]))
    return out


def sessions(pitch_list):
    """Per-session, per-pitch-type averages -- the velocity tab's series."""
    by = defaultdict(list)
    for p in pitch_list:
        by[(p["date"], p["pt"])].append(p)
    out = defaultdict(dict)
    for (dt, pt), g in by.items():
        velos = [x["velo"] for x in g if x.get("velo") is not None]
        if len(velos) < 3:          # one mis-tagged pitch shouldn't draw a line
            continue
        out[dt][pt] = {"velo": _round(mean(velos)), "max": _round(max(velos)),
                       "n": len(velos)}
    return [{"date": d, "by_pt": out[d]} for d in sorted(out)]


# ---------------------------------------------------------------------------
# Percentiles -- against the same level, never the whole program
# ---------------------------------------------------------------------------

def _staff_means(engine, level):
    """Every pitcher's mean for each (pitch type, metric), for one level."""
    with engine.connect() as conn:
        levels = {r.player_id: r.level for r in conn.execute(
            select(db.player_seasons.c.player_id, db.player_seasons.c.level))}
        rows = _pitch_rows(conn)

    by_pitch = defaultdict(dict)
    for r in rows:
        if levels.get(r.player_id) != level:
            continue
        p = by_pitch[(r.player_id, r.session_id, r.seq)]
        p["player_id"] = r.player_id
        p["pt"] = r.pitch_type
        short = FIELDS.get(r.metric_key)
        if short:
            p[short] = r.value

    buckets = defaultdict(lambda: defaultdict(list))
    for p in by_pitch.values():
        if p.get("velo") is None or not p.get("pt"):
            continue
        buckets[(p["pt"], p["player_id"])]["_n"].append(1)
        for short in {v for v in FIELDS.values()}:
            if p.get(short) is not None:
                buckets[(p["pt"], p["player_id"])][short].append(p[short])

    means = defaultdict(dict)
    for (pt, pid), vals in buckets.items():
        # Too few reps to represent a pitcher's pitch in a ranking.
        if len(vals["_n"]) < PROVISIONAL_N:
            continue
        for short, series in vals.items():
            if short == "_n":
                continue
            means[(pt, short)][pid] = mean(series)
    return means


def percentiles(engine, player_id, level, pitch_type="FB"):
    """Where this pitcher's pitch sits among same-level Moeller arms."""
    means = _staff_means(engine, level)
    out = []
    for short, label, unit, dp, higher_better in PERCENTILE_METRICS:
        peers = means.get((pitch_type, short), {})
        mine = peers.get(player_id)
        if mine is None:
            continue
        values = sorted(peers.values())
        n = len(values)
        if n < 3:
            # A percentile out of two people is theatre, not information.
            out.append({"key": short, "label": label, "unit": unit,
                        "value": _round(mine, dp), "pct": None, "n_peers": n,
                        "note": "not enough same-level arms to rank"})
            continue
        below = sum(1 for v in values if v < mine)
        pct = int(round(100.0 * below / (n - 1))) if n > 1 else 50
        out.append({
            "key": short, "label": label, "unit": unit,
            "value": _round(mine, dp), "pct": pct, "pct_label": ordinal(pct),
            "n_peers": n,
            "median": _round(values[n // 2], dp),
            # Break and release have no better direction, so the bar is neutral.
            "directional": higher_better is not None,
        })
    return out


def levels_with_data(engine):
    return sorted({p["level"] for p in pitchers(engine)})
