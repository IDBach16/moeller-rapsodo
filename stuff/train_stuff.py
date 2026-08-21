"""
train_stuff.py -- build the Stuff+ model from MLB Statcast data.

One-off training script; the app never runs this. It downloads a season of
Statcast pitches, converts them into the exact frame Rapsodo reports, trains a
gradient-boosted model to predict each pitch's run value from its physical
characteristics, and saves the fitted pipeline for stuff.py to score with.

Modelled on tjStuff+ (github.com/tnestico/tjstuff_plus, MIT): same target (run
value), same handedness mirroring, same fastball-differential features. Two
deliberate departures:

  * FEATURES ARE LIMITED TO WHAT RAPSODO MEASURES. tjStuff+ uses extension and
    accelerations; the Pitching 2.0 unit reports neither, so the model is
    trained without them -- velocity, spin, IVB, HB, release height/side, and
    each pitch's differentials off the pitcher's own fastball.
  * UNITS AND SIGNS ARE CONVERTED TO RAPSODO'S CONVENTIONS BEFORE TRAINING,
    so scoring needs no translation layer. Statcast reports movement and
    release from the catcher's view in feet; Rapsodo reports the pitcher's
    view in inches (movement) and feet (release). Verified empirically:
    RHP four-seamers average pfx_x = -0.63 ft (catcher view) and Rapsodo shows
    the same pitch as positive arm-side inches, so hb = -pfx_x * 12; release
    mirrors identically, rel_s = -release_pos_x.

Left-handers are mirrored into the right-handed frame (hb and rel_s negated),
exactly as tjStuff+ negates ax/x0 -- shape quality has no handedness.

    python train_stuff.py            # downloads (cached), trains, evaluates, saves

The download caches to statcast_2024.csv beside this file (~250MB) so re-runs
are free. Training data: April-August 2024; September is held out for the
evaluation numbers printed at the end.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "statcast_2024.csv")
MODEL_PATH = os.path.join(HERE, "stuff_model.joblib")

# The columns the model needs from Statcast.
RAW_COLS = ["game_date", "pitcher", "p_throws", "pitch_type",
            "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
            "release_pos_x", "release_pos_z", "delta_run_exp"]

# The features stuff.py must reproduce from Rapsodo data, in this order.
FEATURES = ["velo", "spin", "ivb", "hb", "rel_h", "rel_s",
            "velo_diff", "ivb_diff", "hb_diff"]

FASTBALLS = ["FF", "SI", "FC"]      # the family a pitcher's arsenal hangs off


def download():
    if os.path.exists(CACHE):
        print(f"using cached {CACHE}")
        return pd.read_csv(CACHE)
    from pybaseball import statcast
    frames = []
    months = [("2024-04-01", "2024-04-30"), ("2024-05-01", "2024-05-31"),
              ("2024-06-01", "2024-06-30"), ("2024-07-01", "2024-07-31"),
              ("2024-08-01", "2024-08-31"), ("2024-09-01", "2024-09-29")]
    for start, end in months:
        print(f"downloading {start}..{end}")
        df = statcast(start, end)
        frames.append(df[RAW_COLS])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(CACHE, index=False)
    return out


def to_rapsodo_frame(df):
    """Statcast catcher-view feet -> Rapsodo pitcher-view, lefties mirrored."""
    df = df.dropna(subset=["release_speed", "release_spin_rate", "pfx_x",
                           "pfx_z", "release_pos_x", "release_pos_z",
                           "delta_run_exp", "pitch_type", "p_throws"]).copy()
    df["velo"] = df["release_speed"]
    df["spin"] = df["release_spin_rate"]
    df["ivb"] = df["pfx_z"] * 12.0                       # ft -> inches
    df["hb"] = -df["pfx_x"] * 12.0                       # catcher -> pitcher view
    df["rel_h"] = df["release_pos_z"]
    df["rel_s"] = -df["release_pos_x"]
    lefty = df["p_throws"] == "L"
    df.loc[lefty, "hb"] = -df.loc[lefty, "hb"]           # mirror to RHP frame
    df.loc[lefty, "rel_s"] = -df.loc[lefty, "rel_s"]
    return df


def add_fastball_diffs(df, group_key="pitcher"):
    """Each pitch relative to the pitcher's own primary fastball -- an 82mph
    changeup is a different pitch behind a 95 fastball than behind an 84."""
    fb = df[df["pitch_type"].isin(FASTBALLS)]
    counts = fb.groupby([group_key, "pitch_type"]).size().rename("n").reset_index()
    primary = counts.sort_values("n", ascending=False) \
                    .drop_duplicates(group_key)[[group_key, "pitch_type"]]
    fb_primary = fb.merge(primary, on=[group_key, "pitch_type"])
    base = fb_primary.groupby(group_key).agg(
        fb_velo=("velo", "mean"), fb_ivb=("ivb", "mean"), fb_hb=("hb", "mean"))
    df = df.merge(base, on=group_key, how="left")
    # No fastball at all: fall back to his hardest pitch as the reference.
    hardest = df.groupby(group_key)["velo"].transform("max")
    df["fb_velo"] = df["fb_velo"].fillna(hardest)
    df["fb_ivb"] = df["fb_ivb"].fillna(df["ivb"])
    df["fb_hb"] = df["fb_hb"].fillna(df["hb"])
    df["velo_diff"] = df["velo"] - df["fb_velo"]
    df["ivb_diff"] = df["ivb"] - df["fb_ivb"]
    df["hb_diff"] = df["hb"] - df["fb_hb"]
    return df


def main():
    df = to_rapsodo_frame(download())
    df = add_fastball_diffs(df)
    df["month"] = pd.to_datetime(df["game_date"]).dt.month

    train = df[df["month"] <= 8]
    test = df[df["month"] > 8]
    print(f"train {len(train):,} pitches, holdout {len(test):,}")

    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.03, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=0.2, random_state=42,
        # Velocity is constrained monotone (harder = better, holding shape).
        # Trees are piecewise-constant and would otherwise wobble locally,
        # which turns into rank flips when scoring a staff.
        monotonic_cst=[-1, 0, 0, 0, 0, 0, 0, 0, 0])
    model.fit(train[FEATURES], train["delta_run_exp"])

    pred = model.predict(test[FEATURES])
    base = np.full(len(test), train["delta_run_exp"].mean())
    rmse = float(np.sqrt(np.mean((test["delta_run_exp"] - pred) ** 2)))
    rmse0 = float(np.sqrt(np.mean((test["delta_run_exp"] - base) ** 2)))
    print(f"holdout RMSE {rmse:.5f} vs mean-baseline {rmse0:.5f}")

    # Pitch-level RV is mostly noise (ball-in-play luck), so per-pitch RMSE
    # barely moves. The real test is ordering at the pitcher-pitch level:
    # aggregate predictions should correlate with aggregate outcomes.
    test = test.copy()
    test["pred"] = pred
    agg = test.groupby(["pitcher", "pitch_type"]).agg(
        pred=("pred", "mean"), actual=("delta_run_exp", "mean"),
        n=("pred", "size"))
    agg = agg[agg["n"] >= 50]
    r = float(np.corrcoef(agg["pred"], agg["actual"])[0, 1])
    print(f"pitcher-pitch level corr(pred RV, actual RV), n>=50: r={r:.3f} "
          f"({len(agg)} pitcher-pitches)")

    # Sanity: better raw traits must grade better, all else equal.
    probe = pd.DataFrame([
        dict(velo=94, spin=2300, ivb=16, hb=8, rel_h=5.8, rel_s=1.8,
             velo_diff=0, ivb_diff=0, hb_diff=0),
        dict(velo=97, spin=2300, ivb=16, hb=8, rel_h=5.8, rel_s=1.8,
             velo_diff=0, ivb_diff=0, hb_diff=0),
        dict(velo=94, spin=2300, ivb=20, hb=8, rel_h=5.8, rel_s=1.8,
             velo_diff=0, ivb_diff=0, hb_diff=0),
    ])
    rv = model.predict(probe[FEATURES])
    print(f"probe FB rv: base {rv[0]:.4f} | +3mph {rv[1]:.4f} | +4in ride {rv[2]:.4f}"
          f"  (lower = better for the pitcher)")
    assert rv[1] < rv[0], "harder fastball must grade better"
    assert rv[2] < rv[0], "more ride must grade better"

    # The scoring side needs the MLB reference distribution to anchor scale
    # direction; the Moeller display scale is computed locally in stuff.py.
    train_pred = model.predict(train[FEATURES])
    # Per-feature quantile grids: high-school pitches sit below the model's
    # training support, where trees extrapolate FLAT (an 78 and an 84 score
    # identically). stuff.py maps each staff feature onto the same percentile
    # of this MLB grid before predicting, so the model is used strictly
    # in-support as a shape-weighting function -- the program-relative
    # ordering is preserved and differences don't vanish off the support.
    qgrid = list(range(101))
    meta = {"features": FEATURES,
            "mlb_rv_mean": float(np.mean(train_pred)),
            "mlb_rv_sd": float(np.std(train_pred)),
            "mlb_quantiles": {f: [float(v) for v in
                                  np.percentile(train[f], qgrid)]
                              for f in FEATURES},
            "trained_on": "Statcast 2024 Apr-Aug",
            "holdout_rmse": rmse, "pitcher_pitch_corr": r}
    joblib.dump({"model": model, "meta": meta}, MODEL_PATH)
    print(f"saved {MODEL_PATH}")
    print(meta)


if __name__ == "__main__":
    main()
