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

from flask import Flask, jsonify, render_template, request

import data
import db
import metrics

APP_DIR = os.path.dirname(os.path.abspath(__file__))


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

    return render_template(
        "index.html", empty=False,
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
