pub mod eisner;

use std::sync::Mutex;

const POS_VOCAB: &[&str] = &[
    "n", "v", "wp", "u", "d", "a", "m", "p", "r", "ns", "c", "q", "nt", "nh",
    "nd", "j", "i", "b", "ni", "nz", "nl", "z", "k", "ws", "o", "h", "e",
];

const DEP_VOCAB: &[&str] = &[
    "ATT", "WP", "ADV", "VOB", "SBV", "COO", "RAD", "HED", "POB", "CMP", "LAD", "FOB", "DBL", "IOB",
];

#[derive(thiserror::Error, Debug)]
pub enum LtpError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("ONNX Builder error: {0}")]
    OrtBuilder(String),
}

#[derive(Debug, Clone)]
pub struct DepResult {
    pub head: Vec<usize>,
    pub label: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub words: Vec<String>,
    pub pos: Vec<String>,
    pub dep: DepResult,
}

#[derive(Debug, Clone)]
pub struct CwsParseResult {
    pub pos: Vec<String>,
    pub dep: DepResult,
}

pub struct LtpParser {
    session: Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    cws_vocab: Vec<String>,
    pos_vocab: Vec<String>,
    dep_vocab: Vec<String>,
    use_gpu: bool,
}

fn default_vocabs() -> (Vec<String>, Vec<String>, Vec<String>) {
    let cws = vec!["B-W".to_string(), "I-W".to_string()];
    let pos = POS_VOCAB.iter().map(|&s| s.to_string()).collect();
    let dep = DEP_VOCAB.iter().map(|&s| s.to_string()).collect();
    (cws, pos, dep)
}

impl LtpParser {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self, LtpError> {
        Self::new_with_gpu(model_path, tokenizer_path, false)
    }

    pub fn new_with_gpu(model_path: &str, tokenizer_path: &str, use_gpu: bool) -> Result<Self, LtpError> {
        let mut builder = ort::session::Session::builder()?;
        if use_gpu {
            builder = builder.with_execution_providers([
                ort::ep::CUDA::default().build()
            ]).map_err(|e| LtpError::OrtBuilder(e.to_string()))?;
        }
        let session = builder.commit_from_file(model_path)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| LtpError::Tokenizer(e.to_string()))?;

        let vocab_path = model_path.replace(".onnx", ".vocab");
        let (cws_vocab, pos_vocab, dep_vocab) = if let Ok(content) = std::fs::read_to_string(&vocab_path) {
            let lines: Vec<&str> = content.lines().collect();
            if lines.len() >= 3 {
                let cws: Vec<String> = lines[0].split(',').map(|s| s.to_string()).collect();
                let pos: Vec<String> = lines[1].split(',').map(|s| s.to_string()).collect();
                let dep: Vec<String> = lines[2].split(',').map(|s| s.to_string()).collect();
                (cws, pos, dep)
            } else {
                default_vocabs()
            }
        } else {
            default_vocabs()
        };

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            cws_vocab,
            pos_vocab,
            dep_vocab,
            use_gpu,
        })
    }

    pub fn parse_raw_text(&self, inputs: &[&str]) -> Result<Vec<ParseResult>, LtpError> {
        if self.use_gpu {
            self.parse_raw_text_batched(inputs)
        } else {
            self.parse_raw_text_sequential(inputs)
        }
    }

    pub fn parse_cws_text(&self, inputs: &[&str]) -> Result<Vec<CwsParseResult>, LtpError> {
        if self.use_gpu {
            self.parse_cws_text_batched(inputs)
        } else {
            self.parse_cws_text_sequential(inputs)
        }
    }

    fn parse_raw_text_sequential(&self, inputs: &[&str]) -> Result<Vec<ParseResult>, LtpError> {
        let mut results = Vec::new();
        let mut session = self.session.lock().unwrap();
        for &input in inputs {
            let encoding = self.tokenizer.encode(input, true)
                .map_err(|e| LtpError::Tokenizer(e.to_string()))?;
            let input_ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            let seq_len = input_ids.len();

            let input_ids_vec: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
            let attention_mask_vec: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();

            let input_ids_val = ort::value::Value::from_array((vec![1usize, seq_len], input_ids_vec))?;
            let attention_mask_val = ort::value::Value::from_array((vec![1usize, seq_len], attention_mask_vec))?;

            let inputs_ort = ort::inputs![
                "input_ids" => &input_ids_val,
                "attention_mask" => &attention_mask_val,
            ];

            let outputs = session.run(inputs_ort)?;

            let cws_logits_extracted = outputs.get("cws_logits").unwrap().try_extract_tensor::<f32>()?;
            let cws_slice = cws_logits_extracted.1;

            let pos_logits_extracted = outputs.get("pos_logits").unwrap().try_extract_tensor::<f32>()?;
            let pos_slice = pos_logits_extracted.1;

            let dep_arc_extracted = outputs.get("dep_arc_scores").unwrap().try_extract_tensor::<f32>()?;
            let dep_arc_slice = dep_arc_extracted.1;

            let dep_rel_extracted = outputs.get("dep_rel_scores").unwrap().try_extract_tensor::<f32>()?;
            let dep_rel_slice = dep_rel_extracted.1;

            let offsets = encoding.get_offsets();
            let mut char_idx = Vec::new();
            let mut char_pos = Vec::new();
            let mut last = None;

            if seq_len > 2 {
                for idx in 1..(seq_len - 1) {
                    let (start, end) = offsets[idx];
                    if start == 0 && end == 0 {
                        break;
                    }
                    if start == end {
                        continue;
                    }
                    if Some((start, end)) != last {
                        char_idx.push(idx - 1);
                        char_pos.push(start);
                    }
                    last = Some((start, end));
                }
            }
            char_pos.push(input.len());

            let mut tags = Vec::new();
            for &c_idx in &char_idx {
                let token_idx = c_idx + 1;
                let logit_0 = cws_slice[token_idx * 2 + 0];
                let logit_1 = cws_slice[token_idx * 2 + 1];
                let tag = if logit_1 > logit_0 { 1 } else { 0 };
                tags.push(tag);
            }

            let b_w_idx = self.cws_vocab.iter().position(|x| x == "B-W").unwrap_or(0);
            let mut words = Vec::new();
            let mut word_indices = Vec::new();
            if !tags.is_empty() {
                let mut start = 0;
                for i in 1..tags.len() {
                    if tags[i] == b_w_idx {
                        words.push(input[char_pos[start]..char_pos[i]].to_string());
                        word_indices.push(char_idx[start] + 1);
                        start = i;
                    }
                }
                words.push(input[char_pos[start]..char_pos[tags.len()]].to_string());
                word_indices.push(char_idx[start] + 1);
            }

            let mut pos = Vec::new();
            for &w_tok_idx in &word_indices {
                let mut max_val = -f32::INFINITY;
                let mut max_idx = 0;
                for p_idx in 0..27 {
                    let val = pos_slice[w_tok_idx * 27 + p_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = p_idx;
                    }
                }
                pos.push(self.pos_vocab[max_idx].clone());
            }

            let mut dep_indices = vec![0];
            dep_indices.extend(word_indices.iter().map(|&x| x));
            let num_words = dep_indices.len();

            let mut gathered_arc = ndarray::Array2::zeros((num_words, num_words));
            for i in 0..num_words {
                let dep_idx = dep_indices[i];
                for j in 0..num_words {
                    let head_idx = dep_indices[j];
                    gathered_arc[[i, j]] = dep_arc_slice[dep_idx * seq_len + head_idx];
                }
            }

            gathered_arc.slice_mut(ndarray::s![0, 1..]).fill(-f32::INFINITY);
            for d in 0..num_words {
                gathered_arc[[d, d]] = -f32::INFINITY;
            }

            let heads = eisner::eisner(&gathered_arc, num_words);
            let mut head_results = Vec::new();
            let mut label_results = Vec::new();
            for t in 1..num_words {
                let h = heads[t];
                head_results.push(h);

                let dep_idx = dep_indices[t];
                let head_idx = dep_indices[h];
                let mut max_val = -f32::INFINITY;
                let mut max_idx = 0;
                for r_idx in 0..14 {
                    let val = dep_rel_slice[dep_idx * (seq_len * 14) + head_idx * 14 + r_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = r_idx;
                    }
                }
                label_results.push(self.dep_vocab[max_idx].clone());
            }

            results.push(ParseResult {
                words,
                pos,
                dep: DepResult {
                    head: head_results,
                    label: label_results,
                },
            });
        }
        Ok(results)
    }

    fn parse_raw_text_batched(&self, inputs: &[&str]) -> Result<Vec<ParseResult>, LtpError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self.tokenizer.encode_batch(inputs.to_vec(), true)
            .map_err(|e| LtpError::Tokenizer(e.to_string()))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        if max_len == 0 {
            return Ok(Vec::new());
        }

        let mut input_ids_vec = vec![0i64; batch_size * max_len];
        let mut attention_mask_vec = vec![0i64; batch_size * max_len];

        for (b, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = ids.len();
            for i in 0..len {
                input_ids_vec[b * max_len + i] = ids[i] as i64;
                attention_mask_vec[b * max_len + i] = mask[i] as i64;
            }
        }

        let mut session = self.session.lock().unwrap();
        let input_ids_val = ort::value::Value::from_array((vec![batch_size, max_len], input_ids_vec))?;
        let attention_mask_val = ort::value::Value::from_array((vec![batch_size, max_len], attention_mask_vec))?;

        let inputs_ort = ort::inputs![
            "input_ids" => &input_ids_val,
            "attention_mask" => &attention_mask_val,
        ];

        let outputs = session.run(inputs_ort)?;

        let cws_logits_extracted = outputs.get("cws_logits").unwrap().try_extract_tensor::<f32>()?;
        let cws_slice = cws_logits_extracted.1;

        let pos_logits_extracted = outputs.get("pos_logits").unwrap().try_extract_tensor::<f32>()?;
        let pos_slice = pos_logits_extracted.1;

        let dep_arc_extracted = outputs.get("dep_arc_scores").unwrap().try_extract_tensor::<f32>()?;
        let dep_arc_slice = dep_arc_extracted.1;

        let dep_rel_extracted = outputs.get("dep_rel_scores").unwrap().try_extract_tensor::<f32>()?;
        let dep_rel_slice = dep_rel_extracted.1;

        let mut results = Vec::new();
        for b in 0..batch_size {
            let encoding = &encodings[b];
            let seq_len = encoding.get_ids().len();
            let input = inputs[b];

            let offsets = encoding.get_offsets();
            let mut char_idx = Vec::new();
            let mut char_pos = Vec::new();
            let mut last = None;

            if seq_len > 2 {
                for idx in 1..(seq_len - 1) {
                    let (start, end) = offsets[idx];
                    if start == 0 && end == 0 {
                        break;
                    }
                    if start == end {
                        continue;
                    }
                    if Some((start, end)) != last {
                        char_idx.push(idx - 1);
                        char_pos.push(start);
                    }
                    last = Some((start, end));
                }
            }
            char_pos.push(input.len());

            let mut tags = Vec::new();
            for &c_idx in &char_idx {
                let token_idx = c_idx + 1;
                let base_idx = b * (max_len * 2) + token_idx * 2;
                let logit_0 = cws_slice[base_idx + 0];
                let logit_1 = cws_slice[base_idx + 1];
                let tag = if logit_1 > logit_0 { 1 } else { 0 };
                tags.push(tag);
            }

            let b_w_idx = self.cws_vocab.iter().position(|x| x == "B-W").unwrap_or(0);
            let mut words = Vec::new();
            let mut word_indices = Vec::new();
            if !tags.is_empty() {
                let mut start = 0;
                for i in 1..tags.len() {
                    if tags[i] == b_w_idx {
                        words.push(input[char_pos[start]..char_pos[i]].to_string());
                        word_indices.push(char_idx[start] + 1);
                        start = i;
                    }
                }
                words.push(input[char_pos[start]..char_pos[tags.len()]].to_string());
                word_indices.push(char_idx[start] + 1);
            }

            let mut pos = Vec::new();
            for &w_tok_idx in &word_indices {
                let mut max_val = -f32::INFINITY;
                let mut max_idx = 0;
                let base_idx = b * (max_len * 27) + w_tok_idx * 27;
                for p_idx in 0..27 {
                    let val = pos_slice[base_idx + p_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = p_idx;
                    }
                }
                pos.push(self.pos_vocab[max_idx].clone());
            }

            let mut dep_indices = vec![0];
            dep_indices.extend(word_indices.iter().map(|&x| x));
            let num_words = dep_indices.len();

            let mut gathered_arc = ndarray::Array2::zeros((num_words, num_words));
            for i in 0..num_words {
                let dep_idx = dep_indices[i];
                for j in 0..num_words {
                    let head_idx = dep_indices[j];
                    let base_idx = b * (max_len * max_len) + dep_idx * max_len + head_idx;
                    gathered_arc[[i, j]] = dep_arc_slice[base_idx];
                }
            }

            gathered_arc.slice_mut(ndarray::s![0, 1..]).fill(-f32::INFINITY);
            for d in 0..num_words {
                gathered_arc[[d, d]] = -f32::INFINITY;
            }

            let heads = eisner::eisner(&gathered_arc, num_words);
            let mut head_results = Vec::new();
            let mut label_results = Vec::new();
            for t in 1..num_words {
                let h = heads[t];
                head_results.push(h);

                let dep_idx = dep_indices[t];
                let head_idx = dep_indices[h];
                let mut max_val = -f32::INFINITY;
                let mut max_idx = 0;
                let base_idx = b * (max_len * max_len * 14) + dep_idx * (max_len * 14) + head_idx * 14;
                for r_idx in 0..14 {
                    let val = dep_rel_slice[base_idx + r_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = r_idx;
                    }
                }
                label_results.push(self.dep_vocab[max_idx].clone());
            }

            results.push(ParseResult {
                words,
                pos,
                dep: DepResult {
                    head: head_results,
                    label: label_results,
                },
            });
        }
        Ok(results)
    }

    fn parse_cws_text_sequential(&self, inputs: &[&str]) -> Result<Vec<CwsParseResult>, LtpError> {
        let mut results = Vec::new();
        let mut session = self.session.lock().unwrap();
        for &input in inputs {
            let words: Vec<&str> = input.split('\t').collect();
            if words.is_empty() {
                continue;
            }

            let encoding = self.tokenizer.encode(words.clone(), true)
                .map_err(|e| LtpError::Tokenizer(e.to_string()))?;
            let input_ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            let seq_len = input_ids.len();

            let input_ids_vec: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
            let attention_mask_vec: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();

            let input_ids_val = ort::value::Value::from_array((vec![1usize, seq_len], input_ids_vec))?;
            let attention_mask_val = ort::value::Value::from_array((vec![1usize, seq_len], attention_mask_vec))?;

            let inputs_ort = ort::inputs![
                "input_ids" => &input_ids_val,
                "attention_mask" => &attention_mask_val,
            ];

            let outputs = session.run(inputs_ort)?;

            let pos_logits_extracted = outputs.get("pos_logits").unwrap().try_extract_tensor::<f32>()?;
            let pos_slice = pos_logits_extracted.1;

            let dep_arc_extracted = outputs.get("dep_arc_scores").unwrap().try_extract_tensor::<f32>()?;
            let dep_arc_slice = dep_arc_extracted.1;

            let dep_rel_extracted = outputs.get("dep_rel_scores").unwrap().try_extract_tensor::<f32>()?;
            let dep_rel_slice = dep_rel_extracted.1;

            let mut word_indices = Vec::new();
            let mut last_word_idx = None;
            let word_ids = encoding.get_word_ids();

            for idx in 1..(seq_len - 1) {
                if let Some(word_idx) = word_ids[idx] {
                    if Some(word_idx) != last_word_idx {
                        word_indices.push(idx);
                        last_word_idx = Some(word_idx);
                    }
                }
            }

            let mut pos = Vec::new();
            for &w_tok_idx in &word_indices {
                let mut max_val = -f32::INFINITY;
                let mut max_idx = 0;
                for p_idx in 0..27 {
                    let val = pos_slice[w_tok_idx * 27 + p_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = p_idx;
                    }
                }
                pos.push(self.pos_vocab[max_idx].clone());
            }

            let mut dep_indices = vec![0];
            dep_indices.extend(word_indices.iter().map(|&x| x));
            let num_words = dep_indices.len();

            let mut gathered_arc = ndarray::Array2::zeros((num_words, num_words));
            for i in 0..num_words {
                let dep_idx = dep_indices[i];
                for j in 0..num_words {
                    let head_idx = dep_indices[j];
                    gathered_arc[[i, j]] = dep_arc_slice[dep_idx * seq_len + head_idx];
                }
            }

            gathered_arc.slice_mut(ndarray::s![0, 1..]).fill(-f32::INFINITY);
            for d in 0..num_words {
                gathered_arc[[d, d]] = -f32::INFINITY;
            }

            let heads = eisner::eisner(&gathered_arc, num_words);
            let mut head_results = Vec::new();
            let mut label_results = Vec::new();
            for t in 1..num_words {
                let h = heads[t];
                head_results.push(h);

                let dep_idx = dep_indices[t];
                let head_idx = dep_indices[h];
                let mut max_val = -f32::INFINITY;
                let mut max_idx = 0;
                for r_idx in 0..14 {
                    let val = dep_rel_slice[dep_idx * (seq_len * 14) + head_idx * 14 + r_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = r_idx;
                    }
                }
                label_results.push(self.dep_vocab[max_idx].clone());
            }

            results.push(CwsParseResult {
                pos,
                dep: DepResult {
                    head: head_results,
                    label: label_results,
                },
            });
        }
        Ok(results)
    }

    fn parse_cws_text_batched(&self, inputs: &[&str]) -> Result<Vec<CwsParseResult>, LtpError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut batched_words = Vec::new();
        for &input in inputs {
            let words: Vec<&str> = input.split('\t').collect();
            batched_words.push(words);
        }

        let encodings = self.tokenizer.encode_batch(batched_words, true)
            .map_err(|e| LtpError::Tokenizer(e.to_string()))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        if max_len == 0 {
            return Ok(Vec::new());
        }

        let mut input_ids_vec = vec![0i64; batch_size * max_len];
        let mut attention_mask_vec = vec![0i64; batch_size * max_len];

        for (b, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = ids.len();
            for i in 0..len {
                input_ids_vec[b * max_len + i] = ids[i] as i64;
                attention_mask_vec[b * max_len + i] = mask[i] as i64;
            }
        }

        let mut session = self.session.lock().unwrap();
        let input_ids_val = ort::value::Value::from_array((vec![batch_size, max_len], input_ids_vec))?;
        let attention_mask_val = ort::value::Value::from_array((vec![batch_size, max_len], attention_mask_vec))?;

        let inputs_ort = ort::inputs![
            "input_ids" => &input_ids_val,
            "attention_mask" => &attention_mask_val,
        ];

        let outputs = session.run(inputs_ort)?;

        let pos_logits_extracted = outputs.get("pos_logits").unwrap().try_extract_tensor::<f32>()?;
        let pos_slice = pos_logits_extracted.1;

        let dep_arc_extracted = outputs.get("dep_arc_scores").unwrap().try_extract_tensor::<f32>()?;
        let dep_arc_slice = dep_arc_extracted.1;

        let dep_rel_extracted = outputs.get("dep_rel_scores").unwrap().try_extract_tensor::<f32>()?;
        let dep_rel_slice = dep_rel_extracted.1;

        let mut results = Vec::new();
        for b in 0..batch_size {
            let encoding = &encodings[b];
            let seq_len = encoding.get_ids().len();
            if seq_len == 0 {
                continue;
            }

            let mut word_indices = Vec::new();
            let mut last_word_idx = None;
            let word_ids = encoding.get_word_ids();

            for idx in 1..(seq_len - 1) {
                if let Some(word_idx) = word_ids[idx] {
                    if Some(word_idx) != last_word_idx {
                        word_indices.push(idx);
                        last_word_idx = Some(word_idx);
                    }
                }
            }

            let mut pos = Vec::new();
            for &w_tok_idx in &word_indices {
                let mut max_val = -f32::INFINITY;
                let mut max_idx = 0;
                let base_idx = b * (max_len * 27) + w_tok_idx * 27;
                for p_idx in 0..27 {
                    let val = pos_slice[base_idx + p_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = p_idx;
                    }
                }
                pos.push(self.pos_vocab[max_idx].clone());
            }

            let mut dep_indices = vec![0];
            dep_indices.extend(word_indices.iter().map(|&x| x));
            let num_words = dep_indices.len();

            let mut gathered_arc = ndarray::Array2::zeros((num_words, num_words));
            for i in 0..num_words {
                let dep_idx = dep_indices[i];
                for j in 0..num_words {
                    let head_idx = dep_indices[j];
                    let base_idx = b * (max_len * max_len) + dep_idx * max_len + head_idx;
                    gathered_arc[[i, j]] = dep_arc_slice[base_idx];
                }
            }

            gathered_arc.slice_mut(ndarray::s![0, 1..]).fill(-f32::INFINITY);
            for d in 0..num_words {
                gathered_arc[[d, d]] = -f32::INFINITY;
            }

            let heads = eisner::eisner(&gathered_arc, num_words);
            let mut head_results = Vec::new();
            let mut label_results = Vec::new();
            for t in 1..num_words {
                let h = heads[t];
                head_results.push(h);

                let dep_idx = dep_indices[t];
                let head_idx = dep_indices[h];
                let mut max_val = -f32::INFINITY;
                let mut max_idx = 0;
                let base_idx = b * (max_len * max_len * 14) + dep_idx * (max_len * 14) + head_idx * 14;
                for r_idx in 0..14 {
                    let val = dep_rel_slice[base_idx + r_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = r_idx;
                    }
                }
                label_results.push(self.dep_vocab[max_idx].clone());
            }

            results.push(CwsParseResult {
                pos,
                dep: DepResult {
                    head: head_results,
                    label: label_results,
                },
            });
        }
        Ok(results)
    }
}
