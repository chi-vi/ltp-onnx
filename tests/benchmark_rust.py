import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Rust ONNX Runtime LTP Parser Benchmark Wrapper")
    parser.add_argument("--model", type=str, default="./models/ltp_base2.onnx", help="Path to ONNX model file")
    parser.add_argument("--iterations", type=str, default="10", help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run on (cpu or cuda)")
    args = parser.parse_args()

    print("========================================")
    print("Running Rust ONNX Runtime LTP Parser Benchmark")
    print("========================================")
    
    # Run cargo run --release --bin benchmark_rust -- <model_path> <iterations> <device>
    cmd = ["cargo", "run", "--release", "--bin", "benchmark_rust", "--", args.model, args.iterations, args.device]
    
    try:
        # Run subprocess and stream stdout/stderr
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read stdout line by line
        for line in process.stdout:
            print(line, end="")
            
        # Wait for the process to complete and get stderr
        _, stderr_output = process.communicate()
        
        if process.returncode != 0:
            print("\nError running Rust benchmark:")
            print(stderr_output)
            sys.exit(process.returncode)
            
    except FileNotFoundError:
        print("Error: 'cargo' command not found. Make sure Rust is installed and in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing Rust benchmark: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
