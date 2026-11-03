"""Train baselines, Ridge, and XGBoost; tune on dev; evaluate on both test sets.

Writes everything the report uses to results/: metric tables, tuning grids,
feature importances, a random sample of dev errors, and figures.
"""

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 0

CAT_COLS = ["rarity", "category", "suffix", "stage", "serie", "trainer_type", "energy_type"]
NUM_COLS = ["age_years", "hp", "has_hp", "number_ratio", "is_secret", "log_reprint"]
XGB_EXTRA_CAT = ["type"]
XGB_EXTRA_NUM = ["dex_id", "is_chase", "te_illustrator", "te_set"]

# Species collectors chase regardless of rarity (proposal: Charizard, Pikachu,
# Eeveelutions), plus the mascot legendaries that carry the same premium.
CHASE = {
    "Charizard", "Pikachu", "Eevee", "Vaporeon", "Jolteon", "Flareon", "Espeon",
    "Umbreon", "Leafeon", "Glaceon", "Sylveon", "Mew", "Mewtwo", "Lugia",
    "Rayquaza", "Gengar",
}
TE_SMOOTHING = 10


# ---------- features ----------

def base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # read_csv turns the literal string "None" into NaN.
    for c in CAT_COLS + XGB_EXTRA_CAT + ["illustrator", "set_id"]:
        df[c] = df[c].fillna("none").astype(str)
    df["has_hp"] = df["hp"].notna().astype(int)
    df["hp"] = df["hp"].fillna(0)
    df["number_ratio"] = df["number_ratio"].fillna(df["number_ratio"].median()).clip(0, 3)
    df["log_reprint"] = np.log1p(df["reprint_count"])
    df["dex_id"] = df["dex_id"].fillna(-1)
    species = df["name"].str.replace(r"^(Radiant|Shining|Dark|Light|Mega|M)\s+", "", regex=True)
    df["is_chase"] = species.str.split().str[0].str.replace("'s", "").isin(CHASE).astype(int)
    return df


def target_encode(train: pd.DataFrame, other: pd.DataFrame, col: str, y="log_price"):
    """Smoothed mean target encoding. Train rows get out-of-fold values so a
    rare illustrator's own price does not leak into its feature."""
    prior = train[y].mean()

    def encode(fit, apply):
        stats = fit.groupby(col)[y].agg(["sum", "count"])
        te = (stats["sum"] + TE_SMOOTHING * prior) / (stats["count"] + TE_SMOOTHING)
        return apply[col].map(te).fillna(prior).to_numpy()

    oof = np.zeros(len(train))
    for fit_idx, apply_idx in KFold(5, shuffle=True, random_state=SEED).split(train):
        oof[apply_idx] = encode(train.iloc[fit_idx], train.iloc[apply_idx])
    return oof, encode(train, other)


def design(train, others, cat_cols, num_cols, te_cols=()):
    """One-hot categoricals with train-set levels; returns matrices for train
    and each frame in `others`."""
    train = train.copy()
    others = [o.copy() for o in others]
    for col in te_cols:
        name = "te_" + col.replace("_id", "")
        tr, _ = target_encode(train, train, col)
        train[name] = tr
        for o in others:
            o[name] = target_encode(train, o, col)[1]
    levels = {c: sorted(train[c].astype(str).unique()) for c in cat_cols}

    def mat(df):
        parts = [df[num_cols].astype(float)]
        for c in cat_cols:
            # Levels unseen in training become all-zero rows.
            cat = pd.Categorical(df[c].where(df[c].isin(levels[c])), categories=levels[c])
            parts.append(pd.get_dummies(cat, prefix=c, dtype=float).set_index(df.index))
        X = pd.concat(parts, axis=1)
        # XGBoost rejects feature names containing [, ] or <.
        X.columns = X.columns.str.replace(r"[\[\]<]", "_", regex=True)
        return X

    return mat(train), [mat(o) for o in others]


# ---------- models ----------

class MedianBaseline:
    """Median training log price per group, backing off to coarser groups."""

    def __init__(self, keys):
        self.keys = keys  # e.g. [["rarity", "serie"], ["rarity"]]

    def fit(self, df):
        self.tables = [df.groupby(k)["log_price"].median() for k in self.keys]
        self.global_ = df["log_price"].median()
        return self

    def predict(self, df):
        pred = pd.Series(np.nan, index=df.index)
        for k, table in zip(self.keys, self.tables):
            idx = pd.MultiIndex.from_frame(df[k]) if len(k) > 1 else df[k[0]]
            looked = pd.Series(table.reindex(idx).to_numpy(), index=df.index)
            pred = pred.fillna(looked)
        return pred.fillna(self.global_).to_numpy()


def fit_ridge(Xtr, ytr, alpha):
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    model = Ridge(alpha=alpha).fit((Xtr - mu) / sd, ytr)
    return lambda X: model.predict((X - mu) / sd)


def fit_xgb(Xtr, ytr, params):
    model = XGBRegressor(
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
        tree_method="hist", **params,
    ).fit(Xtr, ytr)
    return model


# ---------- metrics ----------

def metrics(y, pred):
    err = pred - y
    return {
        "rmse_log": float(np.sqrt(np.mean(err**2))),
        "mae_usd": float(np.mean(np.abs(10**pred - 10**y))),
        "spearman": float(spearmanr(y, pred).statistic),
        "within_2x": float(np.mean(np.abs(err) < np.log10(2))),
        "n": int(len(y)),
    }


def per_rarity(df, pred, min_n=30):
    out = {}
    for r, g in df.assign(pred=pred).groupby("rarity"):
        if len(g) >= min_n:
            out[r] = float(np.sqrt(np.mean((g["pred"] - g["log_price"]) ** 2)))
    return out


# ---------- main ----------

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    df = base_features(pd.read_csv(ROOT / "data" / "cards.csv", parse_dates=["release_date"]))
    tr, dv, te = (df[df.split_id == s] for s in ("train", "dev", "test"))

    xgb_cat = CAT_COLS + XGB_EXTRA_CAT
    xgb_num = NUM_COLS + XGB_EXTRA_NUM
    Xr_tr, (Xr_dv, Xr_te) = design(tr, [dv, te], CAT_COLS, NUM_COLS)
    Xx_tr, (Xx_dv, Xx_te) = design(tr, [dv, te], xgb_cat, xgb_num, te_cols=("illustrator", "set_id"))

    # Ridge alpha sweep on dev.
    ridge_grid = []
    for alpha in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        p = fit_ridge(Xr_tr, tr.log_price, alpha)
        ridge_grid.append({"alpha": alpha,
                           "train_rmse": metrics(tr.log_price.values, p(Xr_tr))["rmse_log"],
                           "dev_rmse": metrics(dv.log_price.values, p(Xr_dv))["rmse_log"]})
    ridge_grid = pd.DataFrame(ridge_grid)
    best_alpha = float(ridge_grid.loc[ridge_grid.dev_rmse.idxmin(), "alpha"])

    # XGBoost grid on dev.
    xgb_grid = []
    for depth, lr, n in itertools.product([3, 4, 6, 8], [0.03, 0.1], [200, 500, 1000]):
        params = {"max_depth": depth, "learning_rate": lr, "n_estimators": n}
        m = fit_xgb(Xx_tr, tr.log_price, params)
        xgb_grid.append({**params,
                         "train_rmse": metrics(tr.log_price.values, m.predict(Xx_tr))["rmse_log"],
                         "dev_rmse": metrics(dv.log_price.values, m.predict(Xx_dv))["rmse_log"]})
        print("xgb", params, round(xgb_grid[-1]["dev_rmse"], 4), flush=True)
    xgb_grid = pd.DataFrame(xgb_grid)
    best = xgb_grid.loc[xgb_grid.dev_rmse.idxmin()]
    best_xgb = {"max_depth": int(best.max_depth), "learning_rate": float(best.learning_rate),
                "n_estimators": int(best.n_estimators)}

    # Ablation: XGBoost on only the Ridge features, to separate the effect of
    # the model class from the effect of the extra features.
    xgb_plain = fit_xgb(Xr_tr, tr.log_price, best_xgb)

    def run_all(train, evals, X_r, X_x):
        """Fit every method on `train`, return predictions for each eval frame."""
        b1 = MedianBaseline([["rarity"]]).fit(train)
        b2 = MedianBaseline([["rarity", "serie"], ["rarity"]]).fit(train)
        ridge = fit_ridge(X_r[0], train.log_price, best_alpha)
        xgb = fit_xgb(X_x[0], train.log_price, best_xgb)
        preds = {}
        for name, frame, xr, xx in evals:
            preds[name] = {
                "Rarity median": b1.predict(frame),
                "Rarity $\\times$ era median": b2.predict(frame),
                "Ridge": ridge(xr),
                "XGBoost": xgb.predict(xx),
            }
        return preds, xgb

    preds, xgb_id = run_all(tr, [("dev", dv, Xr_dv, Xx_dv), ("test", te, Xr_te, Xx_te)],
                            [Xr_tr], [Xx_tr])
    preds["dev"]["XGBoost (Ridge features)"] = xgb_plain.predict(Xr_dv)
    preds["test"]["XGBoost (Ridge features)"] = xgb_plain.predict(Xr_te)

    # Out-of-distribution: train on everything released before the cutoff
    # (no dev set; hyperparameters are the ones chosen on the in-distribution dev set).
    otr, ote = df[df.split_ood == "train"], df[df.split_ood == "test"]
    Xr_otr, (Xr_ote,) = design(otr, [ote], CAT_COLS, NUM_COLS)
    Xx_otr, (Xx_ote,) = design(otr, [ote], xgb_cat, xgb_num, te_cols=("illustrator", "set_id"))
    ood_preds, _ = run_all(otr, [("ood", ote, Xr_ote, Xx_ote)], [Xr_otr], [Xx_otr])
    preds.update(ood_preds)

    frames = {"dev": dv, "test": te, "ood": ote}
    results = {split: {m: metrics(frames[split].log_price.values, p) for m, p in ps.items()}
               for split, ps in preds.items()}
    rarity_rmse = {split: {m: per_rarity(frames[split], p) for m, p in ps.items()}
                   for split, ps in preds.items()}

    json.dump({"results": results, "per_rarity": rarity_rmse, "best_alpha": best_alpha,
               "best_xgb": best_xgb}, open(OUT / "metrics.json", "w"), indent=2)
    ridge_grid.to_csv(OUT / "ridge_grid.csv", index=False)
    xgb_grid.to_csv(OUT / "xgb_grid.csv", index=False)
    imp = pd.Series(xgb_id.get_booster().get_score(importance_type="gain"))
    imp.sort_values(ascending=False).to_csv(OUT / "xgb_importance.csv", header=["gain"])

    # Error analysis: random sample of dev cards XGBoost misses by more than 2x.
    err = dv.assign(pred_log=preds["dev"]["XGBoost"])
    err["err"] = err.pred_log - err.log_price
    err["pred_price"] = 10**err.pred_log
    cols = ["id", "name", "set_name", "release_date", "rarity", "suffix", "category", "variant",
            "price", "pred_price", "err", "is_chase", "image"]
    wrong = err[err.err.abs() > np.log10(2)]
    wrong.sample(min(50, len(wrong)), random_state=SEED)[cols].to_csv(OUT / "dev_error_sample.csv", index=False)
    err[cols].to_csv(OUT / "dev_predictions.csv", index=False)
    print(f"dev: {len(wrong)}/{len(err)} XGBoost predictions off by >2x")

    make_figures(df, rarity_rmse)
    for split in results:
        print(f"\n== {split} ==")
        print(pd.DataFrame(results[split]).T.round(3).to_string())
    print("best alpha", best_alpha, "best xgb", best_xgb)


def make_figures(df, rarity_rmse):
    plt.rcParams.update({"font.size": 8, "figure.dpi": 200})

    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ax.hist(df.log_price, bins=60, color="#4C72B0")
    ax.set_xlabel(r"$\log_{10}$(TCGplayer market price, USD)")
    ax.set_ylabel("cards")
    ax.set_xticks(range(-2, 5))
    ax.set_xticklabels(["\\$0.01", "\\$0.10", "\\$1", "\\$10", "\\$100", "\\$1k", "\\$10k"])
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "price_hist.pdf")
    plt.close(fig)

    test = rarity_rmse["test"]
    rar = sorted(test["XGBoost"], key=lambda r: -df[df.rarity == r].shape[0])[:10]
    methods = ["Rarity $\\times$ era median", "Ridge", "XGBoost"]
    labels = ["Rarity×era median", "Ridge", "XGBoost"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    w = 0.27
    for i, (m, lab) in enumerate(zip(methods, labels)):
        ax.barh(np.arange(len(rar)) + (i - 1) * w, [test[m].get(r, np.nan) for r in rar], w, label=lab)
    ax.set_yticks(np.arange(len(rar)))
    ax.set_yticklabels(rar)
    ax.invert_yaxis()
    ax.set_xlabel(r"test RMSE ($\log_{10}$ price)")
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "rarity_rmse.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
