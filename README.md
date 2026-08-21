# Moeller Rapsodo

The pitching dashboard, modelled on `Mine_Pitcher_app.R` — pitcher selector,
pitch-type filters, and four tabs: **Pitch Location · Percentile Rankings ·
Movement Profile · Velocity/Pitch**. Rewritten in Flask so it reads the
player-development Postgres instead of a CSV.

**The data arrives on its own.** `rapsodo-cron` pulls Rapsodo every morning at
5am ET into the same database. This app never calls Rapsodo and never writes —
it is read-only by design.

## Running it locally

```
DATABASE_URL="sqlite:///C:/Users/IDBac/OneDrive/Desktop/Moeller/Player_Dev_Hub/playerdev.db" \
  PORT=5060 python app.py
```

Without `DATABASE_URL` it looks for `playerdev.db` next to `app.py`, which is
empty — point it at the hub's copy, or at Postgres.

## Deploying

⚠ **It must be a service inside the `feisty-luck` project**, not a new Railway
project. Railway's private networking is per-project: the Postgres has no public
endpoint, so a service in a different project could not reach it without exposing
the database to the internet.

```
railway add --service rapsodo-app
railway variables --service rapsodo-app --set 'DATABASE_URL=${{Postgres.DATABASE_URL}}'
railway domain --service rapsodo-app
railway up --service rapsodo-app
```

`railway.json` sets the start command and a healthcheck on `/api/health`, which
reports the pitcher count rather than just "Flask is up" — a deploy that can't
reach the database fails the check instead of serving empty pages.

## The vendored schema

`db.py` and `metrics.py` are **copies** of the ones in `Player_Dev_Hub`. This app
is its own repo and its own deploy, so it can't import them.

`python test_drift.py` checksums the copies against the hub's and checks that
every column this app reads still exists. Run it before deploying. If it fails,
re-copy — do not edit one side only.

## Choices worth knowing

**Percentiles are same-level only.** A pitcher is ranked against other Moeller
arms on his own level — varsity vs varsity, JV vs JV. Ranking a freshman against
varsity measures age, not development. A pitch also needs 10+ reps from a pitcher
before he enters the ranking for it, and fewer than three qualifying arms shows no
percentile at all: a percentile out of two people is theatre.

**Extension is missing on purpose.** The R app ranks it; the Pitching 2.0 unit
does not measure it, and an empty column reads as a bug. Spin efficiency, total
break and tilt fill the space instead.

**Provisional pitches are labelled.** Rapsodo auto-classifies every pitch and
*nothing has ever been coach-validated* (`isValidatedByUser` is false on all 2,387
pitches). A pitch type under 10 reps is marked `provisional`, because three reps
of a "curveball" is the device guessing. If coaches start tagging pitch types in
Rapsodo, the nightly pull picks the corrections up automatically.

**Fixed plot scales.** Movement is locked to ±25in, release to a 4ft/3-7ft window,
location to the plate. Auto-scaling makes every pitcher's axes different, so a
tight cluster and a scattered one look identical.
