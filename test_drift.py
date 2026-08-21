"""
test_drift.py -- has the vendored schema drifted from the hub's?

db.py and metrics.py are COPIES of the ones in Player_Dev_Hub. That is a
deliberate trade: this app is its own repo and its own deploy, so it can't import
them. The cost is that they can silently fall out of step with the database the
hub actually writes, and the symptom would be this app reading columns that no
longer exist -- or worse, quietly missing ones that do.

So the copies are checksummed. If the hub's originals move, this fails and tells
you to re-copy rather than letting the two definitions diverge in the dark.

    python test_drift.py
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The hub sits next to this app under Moeller/. Override if it moves.
HUB = os.environ.get(
    "HUB_PATH", os.path.join(os.path.dirname(HERE), "Player_Dev_Hub"))

VENDORED = ["db.py", "metrics.py"]

FAILS = []


def check(label, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILS.append(label)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


print(f"hub: {HUB}")

if not os.path.isdir(HUB):
    print("\n  [skip] hub checkout not found -- drift can't be checked from here.")
    print("         Set HUB_PATH if Player_Dev_Hub lives somewhere else.")
    sys.exit(0)

for name in VENDORED:
    mine = os.path.join(HERE, name)
    theirs = os.path.join(HUB, name)
    if not os.path.exists(theirs):
        check(f"{name} exists in the hub", False, theirs)
        continue
    same = sha(mine) == sha(theirs)
    check(f"{name} matches the hub's copy", same,
          f"re-copy it: cp '{theirs}' '{mine}'")

# The columns this app actually reads. A schema change that removes one of these
# breaks the dashboard, and it should break the test first.
sys.path.insert(0, HERE)
import db  # noqa: E402

REQUIRED = {
    "players": ["id", "slug", "first_name", "last_name", "is_pitcher"],
    "player_seasons": ["player_id", "season", "level"],
    "sessions": ["id", "player_id", "session_date", "source"],
    "pitch_metrics": ["session_id", "player_id", "seq", "pitch_type",
                      "metric_key", "value"],
}
for table_name, cols in REQUIRED.items():
    table = db.metadata.tables.get(table_name)
    if table is None:
        check(f"table {table_name} exists", False)
        continue
    missing = [c for c in cols if c not in table.c]
    check(f"{table_name} has the columns this app reads", not missing,
          f"missing: {missing}")

print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED:")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("vendored schema is in step with the hub\n")
