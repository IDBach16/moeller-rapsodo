"""
test_stuff.py -- checks for the Stuff+ scoring layer.

Run after train_stuff.py has produced stuff_model.joblib:

    python test_stuff.py

Each check encodes an invariant the model must hold to be worth showing a
coach. If a retrained model fails the ordering checks, do not ship it.
"""

import sys

import stuff

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def fb(velo=84, spin=2100, ivb=15, hb=13, rel_h=5.8, rel_s=1.8, pt="FB"):
    return {"pt": pt, "velo": velo, "spin": spin, "ivb": ivb, "hb": hb,
            "rel_h": rel_h, "rel_s": rel_s}


print("model artefact")
bundle = stuff._load()
check("model file loads", bundle is not None)
check("meta lists the same features scoring builds",
      bundle["meta"]["features"] == stuff.FEATURES, str(bundle["meta"]["features"]))
check("training metadata recorded", "trained_on" in bundle["meta"])

print("\nordering: better raw traits must grade better (inside MLB support)")
# Raw scoring probes sit at MLB-plausible values: below the training range
# trees extrapolate flat, which is exactly why staff_stuff quantile-maps --
# the high-school band is covered by the staff-scaling checks further down.
def rv_of(pitch, throws="R"):
    scored = stuff.score_pitches({1: [pitch]}, {1: throws})
    return scored[1][0][1]

base = rv_of(fb(velo=93))
check("harder fastball grades better", rv_of(fb(velo=96)) < base)
check("more ride grades better", rv_of(fb(velo=93, ivb=19)) < base)
check("more spin does not grade worse",
      rv_of(fb(velo=93, spin=2400)) <= base + 1e-6)

print("\nhandedness mirroring")
# A lefty is the mirror image of a righty: negated hb and release side must
# score identically once mirrored into the trained frame.
r = rv_of(fb(hb=13, rel_s=1.8), throws="R")
l = rv_of(fb(hb=-13, rel_s=-1.8), throws="L")
check("mirrored lefty scores the same as the righty", abs(r - l) < 1e-9,
      f"{r} vs {l}")
l_wrong = rv_of(fb(hb=-13, rel_s=-1.8), throws="R")
check("an unmirrored glove-side 'fastball' scores differently",
      abs(r - l_wrong) > 1e-9)

print("\nscoring hygiene")
scored = stuff.score_pitches(
    {1: [fb(), {"pt": "UNK", "velo": 84, "spin": 2100, "ivb": 15, "hb": 13,
                "rel_h": 5.8, "rel_s": 1.8},
         {"pt": "SL", "velo": 74, "spin": None, "ivb": 2, "hb": -4,
          "rel_h": 5.8, "rel_s": 1.7}]},
    {1: "R"})
check("unlabeled pitches are never scored",
      all(pt != "UNK" for pt, _ in scored[1]))
check("a pitch missing a feature is dropped, not guessed",
      all(pt != "SL" for pt, _ in scored[1]))

print("\nstaff scaling")
staff = {}
throws = {}
for i in range(10):
    # Ten pitchers spread across a velocity band, 30 fastballs each -- the
    # measured floor for a grade that holds to about +/-3 points.
    staff[i] = [fb(velo=78 + i) for _ in range(30)]
    throws[i] = "R"
table = stuff.staff_stuff(staff, throws)
vals = [table[i]["FB"]["stuff_plus"] for i in range(10)]
weighted = sum(v * 30 for v in vals) / 300
check("program average sits at 100", abs(weighted - 100) < 1.0, f"{weighted}")
check("harder throwers rank higher on the staff scale",
      vals == sorted(vals), str(vals))
check("30 reps earns a grade", "FB" in table[0])
thin = stuff.staff_stuff({**staff, 99: [fb(velo=90)] * 29}, {**throws, 99: "R"})
check("29 reps gets NO grade -- withheld, not flagged", 99 not in thin)
check("withholding one pitcher doesn't drop anyone else",
      all(i in thin for i in range(10)))

print()
if FAILED:
    print(f"{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("all stuff+ checks passed")
