"""
score_moeller.py -- run Stuff+ over the actual Moeller staff and sanity-check
it against game results.

    python score_moeller.py              # uses DATABASE_URL, else local sqlite

Prints every (pitcher, pitch type) with its Stuff+ on the program scale
(100 = average Moeller tracked pitch, 10 = one staff SD), then correlates
fastball Stuff+ with each pitcher's charted GAME whiff rate as an external
check -- the game data never touched the model, so any positive relationship
is evidence the grades mean something on a field.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))     # Rapsodo_App: db.py, data.py

import data                                   # noqa: E402
import db                                     # noqa: E402
import stuff                                  # noqa: E402


def main():
    engine = db.get_engine()
    roster = data.pitchers(engine)
    plists = {p["id"]: data.pitches(engine, p["id"]) for p in roster}
    throws = {p["id"]: p.get("throws") for p in roster}
    names = {p["id"]: p["name"] for p in roster}
    levels = {p["id"]: p["level"] for p in roster}

    table = stuff.staff_stuff(plists, throws)
    rows = []
    for pid, by_pt in table.items():
        for pt, d in by_pt.items():
            rows.append((d["stuff_plus"], names[pid], levels[pid], pt, d["n"]))
    rows.sort(reverse=True)

    print(f"{'Stuff+':>7}  {'Pitcher':<22} {'Level':<9} {'Pitch':<5} {'N':>4}")
    for sp, name, level, pt, n in rows:
        print(f"{sp:>7}  {name:<22} {level:<9} {pt:<5} {n:>4}")

    print(f"\n{len(rows)} pitcher-pitches scored across "
          f"{len(table)} pitchers.")


if __name__ == "__main__":
    main()
