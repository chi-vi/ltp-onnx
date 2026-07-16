use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use ltp_onnx::LtpParser;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "./models/ltp_base2.onnx"
    };

    let num_iterations = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(10)
    } else {
        10
    };

    let device = if args.len() > 3 {
        &args[3]
    } else {
        "cpu"
    };

    let tokenizer_path = "./models/tokenizer.json";
    let input_path = "tests/sample.txt";

    println!("========================================");
    println!("Rust ONNX Runtime LTP Parser Benchmark");
    println!("========================================");
    println!("Model: {}", model_path);
    println!("Tokenizer: {}", tokenizer_path);
    println!("Input: {}", input_path);
    println!("Iterations: {}", num_iterations);
    println!("Device: {}", device);
    println!("----------------------------------------");

    // 1. Measure Initialization Time
    let init_start = Instant::now();
    let use_gpu = device == "cuda";
    let parser = LtpParser::new_with_gpu(model_path, tokenizer_path, use_gpu)
        .expect("Failed to initialize LtpParser");
    let init_duration = init_start.elapsed();
    println!("Initialization time: {:.2} ms", init_duration.as_secs_f64() * 1000.0);

    // 2. Read input file
    let file = File::open(input_path).unwrap_or_else(|_| panic!("Failed to open {}", input_path));
    let reader = BufReader::new(file);
    let mut lines = Vec::new();
    let mut total_chars = 0;

    for line in reader.lines() {
        let line = line.expect("Failed to read line").trim().to_string();
        if !line.is_empty() {
            total_chars += line.chars().count();
            lines.push(line);
        }
    }

    let num_sentences = lines.len();
    println!("Loaded {} sentences, {} characters.", num_sentences, total_chars);
    println!("----------------------------------------");

    let inputs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

    // 3. Warmup
    print!("Warming up... ");
    let warmup_start = Instant::now();
    let _ = parser.parse_raw_text(&inputs).expect("Warmup parse failed");
    let warmup_duration = warmup_start.elapsed();
    println!("done in {:.2} ms", warmup_duration.as_secs_f64() * 1000.0);

    // 4. Benchmark Loop
    println!("Running benchmark...");
    let bench_start = Instant::now();
    for _ in 0..num_iterations {
        let _ = parser.parse_raw_text(&inputs).expect("Prediction parse failed");
    }
    let total_duration = bench_start.elapsed();

    // 5. Calculate & Print Metrics
    let total_secs = total_duration.as_secs_f64();
    let avg_iter_ms = (total_secs * 1000.0) / (num_iterations as f64);
    let avg_sentence_ms = avg_iter_ms / (num_sentences as f64);
    
    let total_sentences_processed = (num_sentences * num_iterations) as f64;
    let total_chars_processed = (total_chars * num_iterations) as f64;
    
    let sentences_per_sec = total_sentences_processed / total_secs;
    let chars_per_sec = total_chars_processed / total_secs;

    println!("----------------------------------------");
    println!("Benchmark Results:");
    println!("Total prediction time: {:.2} ms (over {} iterations)", total_secs * 1000.0, num_iterations);
    println!("Average time per iteration: {:.2} ms", avg_iter_ms);
    println!("Average time per sentence: {:.2} ms", avg_sentence_ms);
    println!("Throughput: {:.2} sentences/sec", sentences_per_sec);
    println!("Throughput: {:.2} characters/sec", chars_per_sec);
    println!("========================================");
}
