"""
Moeller Rapsodo — the pitching dashboard.

Modelled on Ian's Mine_Pitcher_app.R (Shiny): a pitcher selector, pitch-type
filters, and four tabs — Pitch Location, Percentile Rankings, Movement Profile,
Velocity/Pitch. Rewritten in Flask so it reads the player-development Postgres
directly instead of a CSV.

The data arrives on its own: `rapsodo-cron` pulls Rapsodo every morning at 5am
into the same database. Nothing here fetches from Rapsodo, and nothing here
writes — this app is read-only by design.

    python app.py            local, against playerdev.db unless DATABASE_URL is set
"""

import os
import sys
import time

from flask import Flask, jsonify, render_template, request

import data
import db
import metrics

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "stuff"))


def _load_dotenv():
    """Local convenience; a no-op on Railway, which supplies real variables."""
    path = os.path.join(APP_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

app = Flask(__name__)


def _engine():
    return db.get_engine()


# Stuff+ is a staff-wide computation -- the scale is the program, so scoring
# one pitcher means scoring everyone. Cached in-process because the data only
# changes once a day when the 5am pull runs.
_STUFF_TTL = 900
_stuff_cache = {"at": 0.0, "rows": None, "by_player": None, "error": None}


def _bar(stuff_plus):
    """Stuff+ 70..130 -> a 0..100% bar with the program average mid-track."""
    return int(round(min(max((stuff_plus - 70.0) / 60.0 * 100.0, 2.0), 100.0)))


def _staff_stuff(engine):
    now = time.time()
    if _stuff_cache["rows"] is not None and now - _stuff_cache["at"] < _STUFF_TTL:
        return _stuff_cache
    try:
        import stuff
        roster = data.pitchers(engine)
        plists = {p["id"]: data.pitches(engine, p["id"]) for p in roster}
        throws = {p["id"]: p.get("throws") for p in roster}
        table = stuff.staff_stuff(plists, throws)
        rows = []
        for p in roster:
            for pt, d in (table.get(p["id"]) or {}).items():
                rows.append({"player_id": p["id"], "name": p["name"],
                             "level": p["level"], "slug": p["slug"], "pt": pt,
                             "label": metrics.PITCH_TYPE_LABELS.get(pt, pt),
                             "stuff": d["stuff_plus"], "bar": _bar(d["stuff_plus"]),
                             "n": d["n"], "provisional": d["provisional"]})
        rows.sort(key=lambda r: -r["stuff"])
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        _stuff_cache.update(at=now, rows=rows, by_player=table, error=None)
    except Exception as e:                        # noqa: BLE001 - shown in the tab
        # Surfaced in the tab rather than 500ing the whole dashboard -- the
        # other four tabs don't depend on the model being loadable.
        _stuff_cache.update(at=now, rows=[], by_player={}, error=str(e))
    return _stuff_cache


@app.route("/")
def index():
    roster = data.pitchers(_engine())
    slug = request.args.get("p")
    chosen = next((p for p in roster if p["slug"] == slug), None) or \
        (roster[0] if roster else None)

    ctx = {"roster": roster, "player": chosen,
           "pitch_labels": metrics.PITCH_TYPE_LABELS}
    if not chosen:
        return render_template("index.html", empty=True, **ctx)

    pitch_list = data.pitches(_engine(), chosen["id"])
    ars = data.arsenal(pitch_list, chosen.get("height_in"))

    # Percentiles run on his most-thrown pitch by default -- ranking a pitcher on
    # a pitch he threw four times is noise dressed as a ranking.
    pt = request.args.get("pt")
    thrown = [a["pt"] for a in ars if a["pt"] != "UNK"]
    if pt not in thrown:
        pt = next((a["pt"] for a in ars if a["pt"] != "UNK"), None)

    st = _staff_stuff(_engine())
    stuff_mine = sorted([r for r in st["rows"] if r["player_id"] == chosen["id"]],
                        key=lambda r: -r["stuff"])

    return render_template(
        "index.html", empty=False,
        stuff_mine=stuff_mine,
        stuff_by_pt={r["pt"]: r for r in stuff_mine},
        stuff_staff=st["rows"],
        stuff_error=st["error"],
        arsenal=ars,
        sessions=data.sessions(pitch_list),
        pitches=[{k: v for k, v in p.items() if k != "session_id"}
                 for p in pitch_list],
        percentiles=data.percentiles(_engine(), chosen["id"],
                                     chosen["level"], pt) if pt else [],
        percentile_pitch=pt,
        thrown=thrown,
        n_pitches=len(pitch_list),
        # Slot inferred from fastball spin axis -- one number per pitcher, plus
        # the side, which the axis reads more reliably than release side does.
        slot=data.pitcher_slot(pitch_list),
        height_in=chosen.get("height_in"),
        **ctx)


@app.route("/api/health")
def health():
    """Railway healthcheck: report the database, not just that Flask is up."""
    try:
        roster = data.pitchers(_engine())
        return jsonify({"ok": True, "pitchers": len(roster)})
    except Exception as e:                       # noqa: BLE001 - surface the cause
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",
            port=int(os.environ.get("PORT", 5060)))
