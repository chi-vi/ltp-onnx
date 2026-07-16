use ltp_onnx::LtpParser;

#[test]
fn test_ltp_parser_raw() {
    let model_path = "./models/ltp_base2.onnx";
    let tokenizer_path = "./models/tokenizer.json";

    let parser = LtpParser::new(model_path, tokenizer_path).expect("Failed to initialize LtpParser");

    let inputs = vec!["他被深渊凝视着。", "天线宝宝非常可爱。"];
    let results = parser.parse_raw_text(&inputs).expect("Failed to parse raw text");

    assert_eq!(results.len(), 2);

    // Sentence 1
    assert_eq!(results[0].words, vec!["他", "被", "深渊", "凝视", "着", "。"]);
    assert_eq!(results[0].pos, vec!["r", "p", "n", "v", "u", "wp"]);
    assert_eq!(results[0].dep.head, vec![4, 4, 2, 0, 4, 4]);
    assert_eq!(results[0].dep.label, vec!["FOB", "ADV", "POB", "HED", "RAD", "WP"]);

    // Sentence 2
    assert_eq!(results[1].words, vec!["天线", "宝宝", "非常", "可爱", "。"]);
    assert_eq!(results[1].pos, vec!["n", "n", "d", "a", "wp"]);
    assert_eq!(results[1].dep.head, vec![2, 4, 4, 0, 4]);
    assert_eq!(results[1].dep.label, vec!["ATT", "SBV", "ADV", "HED", "WP"]);
}

#[test]
fn test_ltp_parser_cws() {
    let model_path = "./models/ltp_base2.onnx";
    let tokenizer_path = "./models/tokenizer.json";

    let parser = LtpParser::new(model_path, tokenizer_path).expect("Failed to initialize LtpParser");

    let inputs = vec!["他\t被\t深渊\t凝视\t着\t。", "天线\t宝宝\t非常\t可爱\t。"];
    let results = parser.parse_cws_text(&inputs).expect("Failed to parse cws text");

    assert_eq!(results.len(), 2);

    // Sentence 1
    assert_eq!(results[0].pos, vec!["r", "p", "n", "v", "u", "wp"]);
    assert_eq!(results[0].dep.head, vec![4, 4, 2, 0, 4, 4]);
    assert_eq!(results[0].dep.label, vec!["FOB", "ADV", "POB", "HED", "RAD", "WP"]);

    // Sentence 2
    assert_eq!(results[1].pos, vec!["n", "n", "d", "a", "wp"]);
    assert_eq!(results[1].dep.head, vec![2, 4, 4, 0, 4]);
    assert_eq!(results[1].dep.label, vec!["ATT", "SBV", "ADV", "HED", "WP"]);
}
