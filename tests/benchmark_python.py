import time
import os
import argparse
from ltp import LTP
import torch

def main():
    parser = argparse.ArgumentParser(description="Python LTP Parser Benchmark")
    parser.add_argument("--model", type=str, default="LTP/base2", help="LTP model name (e.g. LTP/base2, LTP/small, LTP/tiny)")
    parser.add_argument("--input", type=str, default="tests/sample.txt", help="Input file path")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run on (cpu or cuda)")
    args = parser.parse_args()

    print("========================================")
    print("Python LTP Runtime Benchmark")
    print("========================================")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Iterations: {args.iterations}")
    print(f"Requested Device: {args.device}")
    
    # Check CUDA availability
    actual_device = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to cpu.")
        actual_device = "cpu"
    print(f"Actual Device: {actual_device}")
    print("----------------------------------------")

    # 1. Measure Initialization Time
    init_start = time.perf_counter()
    ltp = LTP(args.model)
    ltp.to(actual_device)
    init_duration = time.perf_counter() - init_start
    print(f"Initialization time: {init_duration * 1000.0:.2f} ms")

    # 2. Read input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found at {args.input}")

    lines = []
    total_chars = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
                total_chars += len(line)

    num_sentences = len(lines)
    print(f"Loaded {num_sentences} sentences, {total_chars} characters.")
    print("----------------------------------------")

    # 3. Warmup
    print("Warming up... ", end="", flush=True)
    warmup_start = time.perf_counter()
    # LTP pipeline returns output dict
    _ = ltp.pipeline(lines, tasks=["cws", "pos", "dep"])
    warmup_duration = time.perf_counter() - warmup_start
    print(f"done in {warmup_duration * 1000.0:.2f} ms")

    # 4. Benchmark Loop
    print("Running benchmark...")
    bench_start = time.perf_counter()
    for _ in range(args.iterations):
        _ = ltp.pipeline(lines, tasks=["cws", "pos", "dep"])
    total_duration = time.perf_counter() - bench_start

    # 5. Calculate & Print Metrics
    total_ms = total_duration * 1000.0
    avg_iter_ms = total_ms / args.iterations
    avg_sentence_ms = avg_iter_ms / num_sentences
    
    total_sentences_processed = num_sentences * args.iterations
    total_chars_processed = total_chars * args.iterations
    
    sentences_per_sec = total_sentences_processed / total_duration
    chars_per_sec = total_chars_processed / total_duration

    print("----------------------------------------")
    print("Benchmark Results:")
    print(f"Total prediction time: {total_ms:.2f} ms (over {args.iterations} iterations)")
    print(f"Average time per iteration: {avg_iter_ms:.2f} ms")
    print(f"Average time per sentence: {avg_sentence_ms:.2f} ms")
    print(f"Throughput: {sentences_per_sec:.2f} sentences/sec")
    print(f"Throughput: {chars_per_sec:.2f} characters/sec")
    print("========================================")

if __name__ == "__main__":
    main()
