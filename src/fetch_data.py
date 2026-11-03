"""Download set and card JSON from the TCGdex API into data/raw/.

Each response is cached to disk, so the script is resumable and the raw
snapshot (with its price timestamps) is what every later step reads.
pokemontcg.io was the proposal's primary source but returned intermittent
502s at snapshot time, so TCGdex (the planned backup) is used instead.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

API = "https://api.tcgdex.net/v2/en"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
# TCG Pocket is a separate digital game with no TCGplayer market.
EXCLUDED_SERIES = {"tcgp"}

session = requests.Session()
session.headers["User-Agent"] = "csci467-course-project (USC)"


def get_json(url: str, retries: int = 5):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def cached(path: Path, url: str):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = get_json(url)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def safe_name(card_id: str) -> str:
    # Card ids like "exu-!" or "exu-?" are not valid Windows filenames.
    return quote(card_id, safe="") + ".json"


def main():
    (RAW / "sets").mkdir(parents=True, exist_ok=True)
    (RAW / "cards").mkdir(parents=True, exist_ok=True)

    set_list = cached(RAW / "sets.json", f"{API}/sets")
    print(f"{len(set_list)} sets listed")

    with ThreadPoolExecutor(8) as pool:
        sets = list(pool.map(
            lambda s: cached(RAW / "sets" / safe_name(s["id"]), f"{API}/sets/{quote(s['id'], safe='')}"),
            set_list,
        ))

    card_ids = []
    for s in sets:
        if s is None or s.get("serie", {}).get("id") in EXCLUDED_SERIES:
            continue
        card_ids.extend(c["id"] for c in s.get("cards", []))
    print(f"{len(card_ids)} cards in included sets")

    todo = [c for c in card_ids if not (RAW / "cards" / safe_name(c)).exists()]
    print(f"{len(todo)} cards to download")
    done = 0
    failed = []
    with ThreadPoolExecutor(12) as pool:
        futures = {
            pool.submit(cached, RAW / "cards" / safe_name(c), f"{API}/cards/{quote(c, safe='')}"): c
            for c in todo
        }
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:  # keep going; failures are retried on the next run
                failed.append((futures[f], str(e)))
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    print(f"done, {len(failed)} failures")
    for c, e in failed[:20]:
        print("  FAIL", c, e)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
