import argparse
import statistics
import time
from pathlib import Path

import requests


def p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def summarize(label, values):
    if not values:
        print(f"{label}: no samples")
        return
    print(
        f"{label}: "
        f"n={len(values)} "
        f"avg={statistics.mean(values):.2f}ms "
        f"min={min(values):.2f}ms "
        f"max={max(values):.2f}ms "
        f"p95={p95(values):.2f}ms"
    )


def benchmark_query(base_url, question, runs, timeout):
    latencies = []
    url = f"{base_url.rstrip('/')}/query"

    # Warm-up
    requests.get(url, params={"q": question}, timeout=timeout)

    for _ in range(runs):
        start = time.perf_counter()
        response = requests.get(url, params={"q": question}, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        latencies.append(elapsed_ms)
    summarize("Query latency", latencies)


def benchmark_upload(base_url, file_path, runs, timeout):
    latencies = []
    url = f"{base_url.rstrip('/')}/upload"
    file_name = Path(file_path).name

    for _ in range(runs):
        with open(file_path, "rb") as fh:
            files = {"file": (file_name, fh)}
            start = time.perf_counter()
            response = requests.post(url, files=files, timeout=timeout)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            latencies.append(elapsed_ms)
    summarize("Upload latency", latencies)


def main():
    parser = argparse.ArgumentParser(description="Benchmark local RAG API latency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="what is kinetic energy?")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--upload-file",
        default=None,
        help="Optional file path to benchmark /upload latency",
    )
    args = parser.parse_args()

    print(f"Benchmarking {args.base_url} with {args.runs} runs...")
    benchmark_query(args.base_url, args.question, args.runs, args.timeout)

    if args.upload_file:
        upload_path = Path(args.upload_file)
        if not upload_path.exists():
            raise FileNotFoundError(f"Upload file not found: {upload_path}")
        benchmark_upload(args.base_url, str(upload_path), args.runs, args.timeout)


if __name__ == "__main__":
    main()
