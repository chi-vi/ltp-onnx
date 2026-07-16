# LTP ONNX Parser (Rust)

A fast, memory-efficient Rust library for running [LTP (Language Technology Platform)](https://github.com/HIT-SCIR/ltp) models using ONNX Runtime. It supports Chinese Word Segmentation (CWS), Part of Speech (POS) tagging, and Dependency Parsing (DEP) in a single unified execution.

## Features

- **High Performance**: Powered by [ONNX Runtime (ort)](https://github.com/pyort/ort) for accelerated CPU and GPU (CUDA) inference.
- **Memory Efficient**: Supports FP16 quantized models, significantly reducing memory footprint and model size.
- **Easy Integration**: Built-in Hugging Face Tokenizers (`tokenizers`) and Eisner decoding algorithm for dependency parsing.
- **Flexible Interface**: Support for parsing both raw text and pre-segmented (CWS) text.

---

## Getting Started

### 1. Installation

Add `ltp-onnx` and required dependencies to your `Cargo.toml`:

```toml
[dependencies]
ltp-onnx = { path = "." } # Adjust path to where this crate is located
ort = { version = "2.0.0-rc.12", features = ["cuda"] }
tokenizers = "0.21.0"
```

*Note: Ensure you have the ONNX Runtime dynamic library installed on your system path. If GPU support is enabled, CUDA and cuDNN must be configured.*

### 2. Basic Usage in Rust

Here is a complete example of using `ltp-onnx` to parse raw text:

```rust
use ltp_onnx::LtpParser;

fn main() {
    let model_path = "./models/ltp_base2.onnx";
    let tokenizer_path = "./models/tokenizer.json";

    // Initialize parser on CPU (use LtpParser::new_with_gpu for GPU support)
    let parser = LtpParser::new(model_path, tokenizer_path)
        .expect("Failed to initialize LtpParser");

    let inputs = vec!["他被深渊凝视着。", "天线宝宝非常可爱。"];
    let results = parser.parse_raw_text(&inputs)
        .expect("Failed to parse raw text");

    for (i, result) in results.iter().enumerate() {
        println!("Sentence {}:", i + 1);
        println!("  Words: {:?}", result.words);
        println!("  POS: {:?}", result.pos);
        println!("  DEP Heads: {:?}", result.dep.head);
        println!("  DEP Labels: {:?}", result.dep.label);
    }
}
```

If you already have segmented text (tab-separated words), use `parse_cws_text`:

```rust
let inputs = vec!["他\t被\t深渊\t凝视\t着\t。"];
let results = parser.parse_cws_text(&inputs)
    .expect("Failed to parse CWS text");

for result in &results {
    println!("  POS Tags: {:?}", result.pos);
    println!("  DEP Heads: {:?}", result.dep.head);
}
```

---

## Model Export & Quantization Pipeline

This repository includes a complete pipeline to export PyTorch LTP models to ONNX and quantize them.

### Prerequisites

Install the Python dependencies:

```bash
pip install torch ltp onnx onnxruntime onnxruntime-tools
```

### Step 1: Export LTP Model to ONNX

Use [convert-ltp.py](file:///hub/cvlab/ltp-onnx/convert-ltp.py) to export a HuggingFace LTP model (e.g. `LTP/tiny`, `LTP/small`, `LTP/base2`).

```bash
python convert-ltp.py
```

**What it does:**
1. Loads the standard PyTorch model (e.g., `LTP/small`).
2. Wraps it with a simplified wrapper class `LtpSimplifiedONNX` which exposes the relevant output logits in a single forward pass:
   - `cws_logits` (Word segmentation)
   - `pos_logits` (Part of Speech)
   - `dep_arc_scores` & `dep_rel_scores` (Dependency parsing)
3. Saves the model to `.onnx` with dynamic axes (for batch size and sequence length).
4. Generates a corresponding `.vocab` file containing vocabulary lists required by the Rust decoder.

### Step 2: Quantize to FP16

Quantize the exported model to float16 format using [quantize.py](file:///hub/cvlab/ltp-onnx/quantize.py).

```bash
python quantize.py
```

**What it does:**
1. Uses ONNX Runtime's BERT optimizer to fuse LayerNorm, Attention, and GeLU blocks (to prevent type-mismatch bugs in ONNX Runtime).
2. Converts weights and operations to `float16` (`FP16`), keeping input/output boundary types as `float32` for maximum compatibility.
3. Produces a `*_fp16.onnx` model and copies the corresponding `.vocab` files.

### Step 3: Verify the Quantized Model

Run [verify_quantized.py](file:///hub/cvlab/ltp-onnx/verify_quantized.py) to ensure the FP16 model runs successfully and produces correct results.

```bash
python verify_quantized.py
```

This verifies that the output shapes match and computes the Mean Absolute Error (MAE) compared to the base model.

---

## Benchmarking

We provide benchmark scripts to compare performance and throughput between the Python PyTorch implementation and the Rust ONNX implementation.

### Running Rust Benchmark

Run the Rust benchmark directly via cargo:

```bash
cargo run --release --bin benchmark_rust -- ./models/ltp_base2_fp16.onnx 10 cpu
```

Or run the Python benchmark wrapper:

```bash
python tests/benchmark_rust.py --model ./models/ltp_base2_fp16.onnx --iterations 10 --device cpu
```

### Running Python Benchmark

To benchmark the original PyTorch model:

```bash
python tests/benchmark_python.py --model LTP/base2 --device cpu
```
