#!/usr/bin/env python3
"""Deliberate perturbation test for Megatron-Bridge hcu_probe integration.

Injects a known weight perturbation (multiply one layer by 1.01) and verifies
that hcu_probe dump can detect the resulting output change.

Usage:
    python inject_weight_perturbation.py <dump_baseline_dir> <dump_perturbed_dir>

This script compares two dump directories and reports whether the perturbation
is visible in the forward outputs.
"""

import argparse
import json
import sys
from pathlib import Path


def load_dump_json(dump_dir: Path, rank: int = 0) -> dict:
    """Load dump.json from a given dump directory."""
    candidates = [
        dump_dir / f"step0/rank{rank}/dump.json",
        dump_dir / f"rank{rank}/step0/dump.json",
        dump_dir / "dump.json",
    ]
    for path in candidates:
        if path.is_file():
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"No dump.json found in {dump_dir}")


def find_layer_output_diff(baseline: dict, perturbed: dict, layer_pattern: str = "layers.0") -> bool:
    """Check if any tensor matching layer_pattern shows a detectable difference."""
    baseline_tensors = {item["name"]: item for item in baseline.get("Linears", [])}
    perturbed_tensors = {item["name"]: item for item in perturbed.get("Linears", [])}

    found_diff = False
    for name in baseline_tensors:
        if layer_pattern not in name or name not in perturbed_tensors:
            continue

        b_max = baseline_tensors[name].get("output.max")
        p_max = perturbed_tensors[name].get("output.max")
        b_mean = baseline_tensors[name].get("output.mean")
        p_mean = perturbed_tensors[name].get("output.mean")

        if b_max is not None and p_max is not None:
            rel_diff = abs(p_max - b_max) / (abs(b_max) + 1e-9)
            if rel_diff > 0.005:  # > 0.5% relative change
                print(f"✓ Detected perturbation in {name}")
                print(f"  Baseline max: {b_max:.6f}, Perturbed max: {p_max:.6f}, Rel diff: {rel_diff:.4%}")
                found_diff = True

        if b_mean is not None and p_mean is not None:
            rel_diff = abs(p_mean - b_mean) / (abs(b_mean) + 1e-9)
            if rel_diff > 0.005:
                print(f"✓ Detected perturbation in {name}")
                print(f"  Baseline mean: {b_mean:.6f}, Perturbed mean: {p_mean:.6f}, Rel diff: {rel_diff:.4%}")
                found_diff = True

    return found_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, help="Baseline dump directory")
    parser.add_argument("perturbed", type=Path, help="Perturbed dump directory")
    parser.add_argument("--layer", default="layers.0", help="Layer name pattern to check (default: layers.0)")
    parser.add_argument("--rank", type=int, default=0, help="Rank to check (default: 0)")
    args = parser.parse_args()

    print(f"Loading baseline dump from {args.baseline}")
    baseline = load_dump_json(args.baseline, args.rank)

    print(f"Loading perturbed dump from {args.perturbed}")
    perturbed = load_dump_json(args.perturbed, args.rank)

    print(f"\nChecking for perturbation signal in layer pattern '{args.layer}'...")
    if find_layer_output_diff(baseline, perturbed, args.layer):
        print("\n✓ PASS: hcu_probe successfully detected the injected perturbation")
        return 0
    else:
        print("\n✗ FAIL: hcu_probe did not detect the perturbation (check layer pattern or dump level)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
