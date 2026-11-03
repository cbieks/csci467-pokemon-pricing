"""Flatten the raw TCGdex snapshot into data/cards.csv and assign splits.

One row per card. The target is log10 of the TCGplayer market price of the
card's primary printing. Splits:
  split_id  : 80/10/10 train/dev/test, grouped by card name so reprints of
              the same name never straddle train and test.
  split_ood : cards from sets released in the final 12 months before the
              snapshot are "test"; everything older is "train".
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXCLUDED_SERIES = {"tcgp"}
SEED = 0

# Proposal rule: holofoil only when the card has no normal printing. For
# WotC-era sets TCGplayer splits by edition instead; "unlimited" is the
# common printing and 1st Edition is a premium variant.
VARIANT_PRIORITY = [
    "normal",
    "holofoil",
    "unlimited",
    "unlimited-holofoil",
    "1st-edition",
    "1st-edition-holofoil",
    "reverse-holofoil",
]


def load_sets() -> dict:
    sets = {}
    for f in (RAW / "sets").glob("*.json"):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s is None or s.get("serie", {}).get("id") in EXCLUDED_SERIES:
            continue
        sets[s["id"]] = s
    return sets


def primary_price(tcgplayer: dict):
    for variant in VARIANT_PRIORITY:
        v = tcgplayer.get(variant)
        if isinstance(v, dict) and v.get("marketPrice"):
            return variant, float(v["marketPrice"])
    return None, None


def parse_number(local_id: str):
    # localIds are usually "001" but can be "TG05", "SV049", "H12", "!".
    m = re.search(r"\d+", local_id or "")
    return int(m.group()) if m else np.nan


def main():
    sets = load_sets()
    rows = []
    for f in (RAW / "cards").glob("*.json"):
        c = json.loads(f.read_text(encoding="utf-8"))
        if c is None or c["set"]["id"] not in sets:
            continue
        s = sets[c["set"]["id"]]
        tp = (c.get("pricing") or {}).get("tcgplayer") or {}
        variant, price = primary_price(tp)
        official = (s.get("cardCount") or {}).get("official") or np.nan
        number = parse_number(c.get("localId"))
        rows.append({
            "id": c["id"],
            "name": c["name"],
            "category": c.get("category"),
            "rarity": c.get("rarity") or "None",
            "suffix": c.get("suffix") or "none",
            "stage": c.get("stage") or "none",
            "trainer_type": c.get("trainerType") or "none",
            "energy_type": c.get("energyType") or "none",
            "type": (c.get("types") or ["none"])[0],
            "hp": c.get("hp"),
            "dex_id": (c.get("dexId") or [np.nan])[0],
            "illustrator": c.get("illustrator") or "unknown",
            "set_id": s["id"],
            "set_name": s["name"],
            "serie": s["serie"]["id"],
            "release_date": s.get("releaseDate"),
            "local_id": c.get("localId"),
            "number": number,
            "set_official": official,
            "variant": variant,
            "price": price,
            "price_updated": tp.get("updated"),
            "image": c.get("image"),
        })
    df = pd.DataFrame(rows)
    n_all = len(df)
    df = df[df["price"].notna() & (df["price"] > 0)].copy()
    print(f"{n_all} cards, {len(df)} with a TCGplayer market price")

    df["log_price"] = np.log10(df["price"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    snapshot = pd.to_datetime(df["price_updated"].str[:10]).max()
    df["age_years"] = (snapshot - df["release_date"]).dt.days / 365.25
    df["number_ratio"] = df["number"] / df["set_official"]
    df["is_secret"] = (df["number"] > df["set_official"]).astype(int)
    # Reprint count uses names only, never prices, so it is safe to compute
    # over the whole table.
    df["reprint_count"] = df.groupby("name")["id"].transform("count")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, rest_idx = next(gss.split(df, groups=df["name"]))
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    dev_idx, test_idx = next(gss2.split(rest, groups=rest["name"]))
    df["split_id"] = ""
    df.iloc[train_idx, df.columns.get_loc("split_id")] = "train"
    df.loc[rest.index[dev_idx], "split_id"] = "dev"
    df.loc[rest.index[test_idx], "split_id"] = "test"

    cutoff = snapshot - pd.DateOffset(months=12)
    df["split_ood"] = np.where(df["release_date"] >= cutoff, "test", "train")

    df.to_csv(ROOT / "data" / "cards.csv", index=False)
    print(f"snapshot date {snapshot.date()}, OOD cutoff {cutoff.date()}")
    print(df["split_id"].value_counts().to_string())
    print(df["split_ood"].value_counts().to_string())
    print(df["variant"].value_counts().to_string())


if __name__ == "__main__":
    main()
