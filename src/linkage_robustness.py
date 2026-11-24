"""
linkage_robustness.py
=======================
A controlled experiment answering a real research question, independent of
whether the underlying membership data is real or synthetic:

    "How robust is fuzzy name+zip record linkage to data-entry noise, and
    is there a better linkage strategy than exact-match on a normalized key?"

This is deliberately NOT about discovering something about emeritus faculty
(the synthetic data can't support that) — it's a methods experiment about
the record-linkage approach used in src/data_cleaning.py, which is exactly
the kind of question a data engineer actually has to answer before trusting
a fuzzy join in production.

Method
------
1. Generate N "true" synthetic identities (name + zip).
2. For each identity, create a "noisy" second-system record by applying
   realistic data-entry corruption (typos, transpositions, truncated/altered
   zip digits) at a controlled corruption probability p.
3. Compare two linkage strategies at each corruption level:
     - Strategy A (production): exact match on the normalized
       `last+first+zip` key, as implemented in src/data_cleaning.linkage_key.
       This is what the rest of the project actually uses.
     - Strategy B (similarity-threshold alternative): fuzzy name similarity
       (difflib ratio) among same-zip candidates, accepted above a
       threshold.
4. Measure RECALL (did the noisy record correctly link back to its true
   identity?) and the FALSE-MATCH RATE (did it incorrectly link to a
   *different* identity?) for both strategies across a range of corruption
   levels, revealing the precision/recall trade-off and the point at which
   each strategy breaks down.

Run directly to produce data/processed/linkage_robustness.csv:
    python -m src.linkage_robustness
"""
from __future__ import annotations

import difflib
import random
from pathlib import Path

import pandas as pd
from faker import Faker

from src.data_cleaning import normalize_text, normalize_zip, linkage_key

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

N_IDENTITIES = 400
CORRUPTION_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
N_TRIALS_PER_LEVEL = 5          # repeat with different random corruption draws
SIMILARITY_THRESHOLD = 0.82     # Strategy B acceptance threshold

fake = Faker()
Faker.seed(123)
random.seed(123)


def make_identities(n=N_IDENTITIES):
    identities = []
    for i in range(n):
        identities.append({
            "id": i,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "zip": f"85{random.randint(200, 399)}",
        })
    return identities


def corrupt_string(s: str, p: float) -> str:
    """Apply character-level noise (typo/transposition/deletion/substitution)
    to a string with per-character probability p, mimicking realistic
    data-entry error rates."""
    if not s or p <= 0:
        return s
    chars = list(s)
    i = 0
    out = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    while i < len(chars):
        if random.random() < p:
            op = random.choice(["sub", "del", "transpose", "dup"])
            if op == "sub":
                out.append(random.choice(alphabet))
                i += 1
            elif op == "del":
                i += 1  # drop this character
            elif op == "transpose" and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(chars[i])
                i += 2
            else:  # dup
                out.append(chars[i])
                out.append(chars[i])
                i += 1
        else:
            out.append(chars[i])
            i += 1
    return "".join(out)


def corrupt_zip(z: str, p: float) -> str:
    digits = list(z)
    for i in range(len(digits)):
        if random.random() < p:
            digits[i] = random.choice("0123456789")
    return "".join(digits)


def make_noisy_record(identity: dict, p: float) -> dict:
    return {
        "id": identity["id"],
        "first_name": corrupt_string(identity["first_name"], p),
        "last_name": corrupt_string(identity["last_name"], p),
        "zip": corrupt_zip(identity["zip"], p),
    }


# --- Strategy A: production exact-match linkage key -----------------------
def match_strategy_a(noisy: dict, clean_by_key: dict) -> int | None:
    key = linkage_key(noisy["first_name"], noisy["last_name"], noisy["zip"])
    return clean_by_key.get(key)


# --- Strategy B: zip-exact + name-similarity threshold ---------------------
def match_strategy_b(noisy: dict, clean_by_zip: dict) -> int | None:
    zip5 = normalize_zip(noisy["zip"])
    candidates = clean_by_zip.get(zip5, [])
    if not candidates:
        return None
    noisy_name = f"{normalize_text(noisy['first_name'])}{normalize_text(noisy['last_name'])}"
    best_id, best_score = None, 0.0
    for cand in candidates:
        cand_name = f"{normalize_text(cand['first_name'])}{normalize_text(cand['last_name'])}"
        score = difflib.SequenceMatcher(None, noisy_name, cand_name).ratio()
        if score > best_score:
            best_score, best_id = score, cand["id"]
    if best_score >= SIMILARITY_THRESHOLD:
        return best_id
    return None


def run_experiment() -> pd.DataFrame:
    identities = make_identities()
    clean_by_key = {
        linkage_key(ident["first_name"], ident["last_name"], ident["zip"]): ident["id"]
        for ident in identities
    }
    clean_by_zip: dict[str, list] = {}
    for ident in identities:
        clean_by_zip.setdefault(normalize_zip(ident["zip"]), []).append(ident)

    rows = []
    for p in CORRUPTION_LEVELS:
        for trial in range(N_TRIALS_PER_LEVEL):
            correct_a = false_a = miss_a = 0
            correct_b = false_b = miss_b = 0
            for ident in identities:
                noisy = make_noisy_record(ident, p)

                result_a = match_strategy_a(noisy, clean_by_key)
                if result_a is None:
                    miss_a += 1
                elif result_a == ident["id"]:
                    correct_a += 1
                else:
                    false_a += 1

                result_b = match_strategy_b(noisy, clean_by_zip)
                if result_b is None:
                    miss_b += 1
                elif result_b == ident["id"]:
                    correct_b += 1
                else:
                    false_b += 1

            n = len(identities)
            rows.append({
                "corruption_level": p, "trial": trial, "strategy": "A: exact normalized key",
                "recall": correct_a / n, "false_match_rate": false_a / n, "miss_rate": miss_a / n,
            })
            rows.append({
                "corruption_level": p, "trial": trial, "strategy": "B: zip-exact + name-similarity",
                "recall": correct_b / n, "false_match_rate": false_b / n, "miss_rate": miss_b / n,
            })

    return pd.DataFrame(rows)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = run_experiment()
    summary = df.groupby(["strategy", "corruption_level"])[["recall", "false_match_rate", "miss_rate"]].mean().reset_index()
    out_path = PROCESSED_DIR / "linkage_robustness.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote linkage robustness experiment results to {out_path}")
    print(summary.to_string(index=False))

    # Headline finding
    a_at_30 = summary[(summary.strategy.str.startswith("A")) & (summary.corruption_level == 0.3)]["recall"].values[0]
    b_at_30 = summary[(summary.strategy.str.startswith("B")) & (summary.corruption_level == 0.3)]["recall"].values[0]
    b_false_at_30 = summary[(summary.strategy.str.startswith("B")) & (summary.corruption_level == 0.3)]["false_match_rate"].values[0]
    print(f"\nAt 30% character-level corruption: Strategy A recall={a_at_30:.1%}, "
          f"Strategy B recall={b_at_30:.1%} (false-match rate={b_false_at_30:.1%})")


if __name__ == "__main__":
    main()
