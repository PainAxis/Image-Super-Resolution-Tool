"""Reproducible wall-time and resource-estimate benchmark."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from sr_tool.fsr.pipeline import process_image
from sr_tool.utils.resources import estimate_peak_resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--scale", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--fxaa", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width < 1 or args.height < 1 or args.warmups < 0 or args.repetitions < 1:
        raise SystemExit(
            "dimensions/repetitions must be positive and warmups non-negative"
        )
    image = np.random.default_rng(20260824).random(
        (args.height, args.width, 3), dtype=np.float32
    )
    for _ in range(args.warmups):
        process_image(image, args.scale, antialias=args.fxaa)
    durations: list[float] = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        result = process_image(image, args.scale, antialias=args.fxaa)
        durations.append(time.perf_counter() - started)
    median = statistics.median(durations)
    output_megapixels = result.shape[0] * result.shape[1] / 1_000_000
    estimate = estimate_peak_resources(args.height, args.width, args.scale)
    print(
        f"input={args.width}x{args.height} scale={args.scale} fxaa={args.fxaa} "
        f"warmups={args.warmups} repetitions={args.repetitions} "
        f"median={median:.3f}s output={output_megapixels:.3f}MP "
        f"throughput={output_megapixels / median:.3f}MP/s "
        f"estimated_peak={estimate.peak_mib:.1f}MiB"
    )


if __name__ == "__main__":
    main()
