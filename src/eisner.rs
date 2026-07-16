pub fn eisner(scores: &ndarray::Array2<f32>, seq_len: usize) -> Vec<usize> {
    let mut s_i = vec![vec![-f32::INFINITY; seq_len]; seq_len];
    let mut s_c = vec![vec![-f32::INFINITY; seq_len]; seq_len];
    let mut p_i = vec![vec![0; seq_len]; seq_len];
    let mut p_c = vec![vec![0; seq_len]; seq_len];

    for i in 0..seq_len {
        s_c[i][i] = 0.0;
    }

    for w in 1..seq_len {
        for i in 0..(seq_len - w) {
            let j = i + w;

            // Incomplete Left/Right
            let mut max_il = -f32::INFINITY;
            let mut arg_il = 0;
            let mut max_ir = -f32::INFINITY;
            let mut arg_ir = 0;

            for r in i..j {
                let val_ilr = s_c[i][r] + s_c[j][r + 1];

                // Left incomplete: j -> i (head j, dependent i) -> scores[i, j]
                let val_il = val_ilr + scores[[i, j]];
                if val_il > max_il {
                    max_il = val_il;
                    arg_il = r;
                }

                // Right incomplete: i -> j (head i, dependent j) -> scores[j, i]
                let val_ir = val_ilr + scores[[j, i]];
                if val_ir > max_ir {
                    max_ir = val_ir;
                    arg_ir = r;
                }
            }

            s_i[j][i] = max_il;
            p_i[j][i] = arg_il;

            s_i[i][j] = max_ir;
            p_i[i][j] = arg_ir;

            // Complete Left
            let mut max_cl = -f32::INFINITY;
            let mut arg_cl = 0;
            for r in i..j {
                let val_cl = s_c[r][i] + s_i[j][r];
                if val_cl > max_cl {
                    max_cl = val_cl;
                    arg_cl = r;
                }
            }
            s_c[j][i] = max_cl;
            p_c[j][i] = arg_cl;

            // Complete Right
            let mut max_cr = -f32::INFINITY;
            let mut arg_cr = 0;
            for r in (i + 1)..=j {
                let val_cr = s_i[i][r] + s_c[r][j];
                if val_cr > max_cr {
                    max_cr = val_cr;
                    arg_cr = r;
                }
            }

            if i == 0 && w != seq_len - 1 {
                s_c[i][j] = -f32::INFINITY;
            } else {
                s_c[i][j] = max_cr;
                p_c[i][j] = arg_cr;
            }
        }
    }

    let mut heads = vec![1; seq_len];
    backtrack(&p_i, &p_c, &mut heads, 0, seq_len - 1, true);
    heads
}

fn backtrack(
    p_i: &[Vec<usize>],
    p_c: &[Vec<usize>],
    heads: &mut [usize],
    i: usize,
    j: usize,
    complete: bool,
) {
    if i == j {
        return;
    }
    if complete {
        let r = p_c[i][j];
        backtrack(p_i, p_c, heads, i, r, false);
        backtrack(p_i, p_c, heads, r, j, true);
    } else {
        let r = p_i[i][j];
        heads[j] = i;
        let (low, high) = if i < j { (i, j) } else { (j, i) };
        backtrack(p_i, p_c, heads, low, r, true);
        backtrack(p_i, p_c, heads, high, r + 1, true);
    }
}
