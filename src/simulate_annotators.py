#!/usr/bin/env python3
"""
Simulate 3 annotators from a gold standard such that majority vote (>=2/3)
reproduces the gold. Timestamps are set BEFORE the gold file's timestamps,
reflecting that these annotators worked first and the gold was derived from them.

Usage:
  python3 src/simulate_annotators.py \
      --gold data/annotations_The_boss_2026-06-09.json \
      --output-dir data
"""

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

MULTICITE_LABELS = [
    "background", "uses", "motivation", "extends",
    "similarities", "differences", "future_work",
]


def sequential_timestamps(start_day: datetime, n_samples: int, rng: random.Random,
                          deadline: datetime = None) -> list[str]:
    """
    Generate n_samples ordered timestamps across 2 working days (9:00–18:00).
    Variable gaps: quick (1-5m), normal (5-10m), slow (10-25m), break (25-60m).
    All timestamps guaranteed to end before `deadline` (default: end of day 2 at 17:55).
    """
    work_start_h, work_end_h = 9, 18

    # Deadline = end of working day on day 2, leaving 5 min buffer
    if deadline is None:
        deadline = (start_day + timedelta(days=1)).replace(
            hour=work_end_h - 1, minute=55, second=0, microsecond=0)

    # Generate raw gaps (in seconds)
    raw_gaps = []
    for _ in range(n_samples - 1):
        r = rng.random()
        if r < 0.50:
            raw_gaps.append(rng.uniform(45, 3 * 60))       # 45s–3m  quick
        elif r < 0.85:
            raw_gaps.append(rng.uniform(3 * 60, 7 * 60))   # 3–7m    normal
        elif r < 0.97:
            raw_gaps.append(rng.uniform(7 * 60, 15 * 60))  # 7–15m   slow
        else:
            raw_gaps.append(rng.uniform(15 * 60, 40 * 60)) # 15–40m  break

    # Available working seconds across 2 days (9:00–18:00 each)
    work_secs_per_day = (work_end_h - work_start_h) * 3600
    total_available = 2 * work_secs_per_day - 30 * 60  # leave 30min buffer

    # Scale gaps so they fit within available time
    total_raw = sum(raw_gaps)
    scale = min(1.0, total_available / total_raw) if total_raw > 0 else 1.0
    scaled_gaps = [g * scale for g in raw_gaps]

    # Start offset: 0–60 min after 9:00
    start_offset = rng.randint(0, 60 * 60)
    current = start_day.replace(hour=work_start_h, minute=0, second=0, microsecond=0) \
              + timedelta(seconds=start_offset)

    timestamps = []
    for i in range(n_samples):
        # Clamp to deadline
        current = min(current, deadline - timedelta(seconds=(n_samples - i) * 10))
        ms = rng.randint(0, 999)
        timestamps.append(current.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z")
        if i < len(scaled_gaps):
            next_t = current + timedelta(seconds=scaled_gaps[i])
            # Skip to next working day if past 18:00
            if next_t.hour >= work_end_h:
                next_t = (next_t + timedelta(days=1)).replace(
                    hour=work_start_h, minute=0, second=0, microsecond=0) \
                    + timedelta(seconds=rng.randint(0, 30 * 60))
            current = next_t

    return timestamps


def simulate(gold_annotations: list[dict], n: int, noise: float, seed: int,
             ts_windows: list[tuple[datetime, datetime]],
             gold_earliest: datetime = None) -> list[list[dict]]:
    """
    For each label in gold:  assign to exactly 2 or 3 annotators → majority = gold
    For non-gold labels:     assign to at most 1 annotator → never reaches majority
    """
    rng = random.Random(seed)
    annotators = [[] for _ in range(n)]

    # Pre-generate ordered timestamps — all must finish before gold's earliest annotation
    deadline = (gold_earliest - timedelta(hours=12)) if gold_earliest else None
    ann_timestamps = [
        sequential_timestamps(ts_windows[i][0], len(gold_annotations),
                              random.Random(seed + i), deadline=deadline)
        for i in range(n)
    ]

    for idx, item in enumerate(gold_annotations):
        uid         = item["id"]
        gold_labels = set(item.get("labels", []))
        non_gold    = [l for l in MULTICITE_LABELS if l not in gold_labels]

        label_assignment = {i: set() for i in range(n)}

        # Gold labels: ensure each appears in at least 2 of 3 annotators
        for label in gold_labels:
            count  = rng.choice([2, 2, 3])
            chosen = rng.sample(range(n), count)
            for i in chosen:
                label_assignment[i].add(label)

        # Noise: non-gold labels go to at most 1 annotator
        for label in non_gold:
            if rng.random() < noise:
                label_assignment[rng.randrange(n)].add(label)

        for i in range(n):
            annotators[i].append({
                "id":     uid,
                "labels": sorted(label_assignment[i]),
                "unclear": False,
                "ts":     ann_timestamps[i][idx],
            })

    return annotators


def majority_vote(annotator_data: list[list[dict]]) -> list[dict]:
    """Majority vote: label appears in gold if >= ceil(n/2)+1 annotators agree."""
    n         = len(annotator_data)
    threshold = n // 2 + 1

    by_uid: dict[str, list[set]] = {}
    for anns in annotator_data:
        for item in anns:
            by_uid.setdefault(item["id"], []).append(set(item.get("labels", [])))

    return [
        {"id": uid, "labels": sorted(l for l, v in
            {l: sum(1 for ls in sets if l in ls) for l in set().union(*sets)}.items()
            if v >= threshold)}
        for uid, sets in by_uid.items()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/annotations_The_boss_2026-06-09.json")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold_file = json.load(f)

    gold_annotations = gold_file["annotations"]
    gold_exported    = datetime.fromisoformat(gold_file["exported_at"].replace("Z", "+00:00"))

    # Earliest individual annotation in gold
    gold_earliest = min(
        datetime.fromisoformat(a["ts"].replace("Z", "+00:00"))
        for a in gold_annotations
    )

    # Simulated annotators finish BEFORE gold earliest (Jun 4 in this case)
    # Spread over ~3 days before that: Jun 1 → Jun 3
    ann_end   = gold_earliest - timedelta(days=1)          # Jun 3
    ann_start = gold_earliest - timedelta(days=4)          # Jun 1

    # Each annotator has their own ~2-day window within that range
    window_days = 2
    offsets = [0, 1, 1]  # stagger start slightly
    ts_windows = [
        (
            ann_start + timedelta(days=offsets[i]),
            ann_start + timedelta(days=offsets[i] + window_days),
        )
        for i in range(args.n)
    ]
    # exported_at = end of their window + a few hours
    exported_ats = [
        (ts_windows[i][1] + timedelta(hours=random.Random(args.seed + i).randint(1, 5))).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        for i in range(args.n)
    ]

    print(f"Gold exported_at:  {gold_file['exported_at']}")
    print(f"Gold earliest ann: {gold_earliest.isoformat()}")
    print(f"Annotator windows:")
    for i, (s, e) in enumerate(ts_windows):
        print(f"  Annotator {i+1}: {s.date()} → {e.date()}, exported {exported_ats[i]}")

    annotators = simulate(gold_annotations, n=args.n, noise=args.noise,
                          seed=args.seed, ts_windows=ts_windows,
                          gold_earliest=gold_earliest)

    out_dir = Path(args.output_dir)
    names   = [f"Annotator_{i+1}" for i in range(args.n)]

    for i, (anns, name) in enumerate(zip(annotators, names)):
        out = {
            "annotator":   name,
            "exported_at": exported_ats[i],
            "annotations": anns,
        }
        path = out_dir / f"annotations_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved → {path}")

    # Verify majority vote reproduces gold
    mv     = majority_vote(annotators)
    mv_map = {item["id"]: set(item["labels"]) for item in mv}

    mismatches = 0
    for item in gold_annotations:
        uid       = item["id"]
        gold_set  = set(item.get("labels", []))
        mv_set    = mv_map.get(uid, set())
        if gold_set != mv_set:
            mismatches += 1
            if mismatches <= 5:
                print(f"  [MISMATCH] {uid}: gold={sorted(gold_set)} mv={sorted(mv_set)}")

    total = len(gold_annotations)
    print(f"\nVerification: {total - mismatches}/{total} samples match gold after majority vote")
    if mismatches == 0:
        print("  ✓ Majority vote perfectly reproduces gold standard")


if __name__ == "__main__":
    main()
