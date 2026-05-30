#!/usr/bin/env python3
"""Regenerate the Lloyd-Max codebook constants used by ggml-quants.c
TurboQuant types.

Spec: doc/specs/2026-05-30-turboquant-kv-cache.md §3.2 (in the engine workspace).

The CPU reference in ggml-quants.c uses Gaussian Lloyd-Max levels scaled
by 1/sqrt(QK_TURBO). For d = 128 the limiting distribution of normalised
coordinates after FWHT is well approximated by N(0, 1/d), and Gaussian
Lloyd-Max is within ~0.5 % of full Beta(0.5, (d-1)/2) Lloyd-Max while
being analytically tractable.

Run this when changing QK_TURBO or when refining against the true Beta
distribution:

    python tools/turboquant/gen_codebook.py --bits 3
    python tools/turboquant/gen_codebook.py --bits 2

Outputs C constants you can paste into ggml-quants.c.
"""
import argparse
import math
import sys


def lloyd_max_gaussian(n_levels: int, iters: int = 200) -> list[float]:
    """Symmetric Lloyd-Max for N(0, 1). Returns the n_levels centroids in
    ascending order. Uses the standard fixed-point iteration: alternate
    between computing region midpoints (decision thresholds) and the
    conditional mean per region.
    """
    # Initial centroids — uniform on [-3, 3] interior.
    levels = [(-3 + 6 * (i + 0.5) / n_levels) for i in range(n_levels)]

    # Use a fine grid for numerical integration of the pdf.
    GRID_N = 200_000
    GRID_LO, GRID_HI = -8.0, 8.0
    dx = (GRID_HI - GRID_LO) / GRID_N
    xs = [GRID_LO + (i + 0.5) * dx for i in range(GRID_N)]
    inv_sqrt_2pi = 1.0 / math.sqrt(2 * math.pi)
    pdf = [inv_sqrt_2pi * math.exp(-0.5 * x * x) for x in xs]

    for _ in range(iters):
        # Decision thresholds: midpoints between adjacent centroids.
        thr = [-float("inf")] + [(levels[i] + levels[i + 1]) * 0.5
                                  for i in range(n_levels - 1)] + [float("inf")]
        # Compute conditional means per region.
        new_levels = []
        for k in range(n_levels):
            num = 0.0
            den = 0.0
            for i, x in enumerate(xs):
                if thr[k] <= x < thr[k + 1]:
                    num += x * pdf[i]
                    den += pdf[i]
            new_levels.append(num / den if den > 0 else levels[k])
        # Check convergence (max |delta|).
        delta = max(abs(a - b) for a, b in zip(levels, new_levels))
        levels = new_levels
        if delta < 1e-8:
            break

    return sorted(levels)


def emit_c_codebook(bits: int, qk: int = 128) -> str:
    n_levels = 1 << bits
    levels = lloyd_max_gaussian(n_levels)
    inv_sqrt_qk = 1.0 / math.sqrt(qk)

    out = []
    out.append(f"// {bits}-bit Lloyd-Max optimal centroids for N(0, 1).")
    out.append(f"// QK = {qk}, inv_sqrt_qk = {inv_sqrt_qk:.17f}.")
    out.append(f"static const float TURBO{bits}_CODEBOOK[{n_levels}] = {{")
    for v in levels:
        out.append(f"    {v:+.4f}f * TURBOQ_INV_SQRT_QK,")
    out.append("};")
    out.append("")
    out.append(f"static const float TURBO{bits}_THRESHOLDS[{n_levels - 1}] = {{")
    for i in range(n_levels - 1):
        m = (levels[i] + levels[i + 1]) * 0.5
        out.append(f"    ({levels[i]:+.4f}f + {levels[i + 1]:+.4f}f) * 0.5f * TURBOQ_INV_SQRT_QK,")
    out.append("};")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=3, choices=(2, 3, 4))
    p.add_argument("--qk", type=int, default=128)
    args = p.parse_args()
    print(emit_c_codebook(args.bits, args.qk))
    return 0


if __name__ == "__main__":
    sys.exit(main())
