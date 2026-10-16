/// Criterion enum used by discretizers and heuristic.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Criterion {
    Gini,
    Entropy,
}

impl std::str::FromStr for Criterion {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "gini" => Ok(Criterion::Gini),
            "entropy" => Ok(Criterion::Entropy),
            _ => Err(format!(
                "Unknown criterion: '{}'. Expected 'gini' or 'entropy'.",
                s
            )),
        }
    }
}

/// Compute impurity from class counts.
/// counts[c] = number of samples with class c. n = total samples.
pub fn impurity(criterion: Criterion, counts: &[u32], n: u32) -> f64 {
    match criterion {
        Criterion::Gini => gini(counts, n),
        Criterion::Entropy => entropy(counts, n),
    }
}

/// Gini impurity: 1 - sum(p_c^2)
pub fn gini(counts: &[u32], n: u32) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let mut sum_sq = 0.0;
    for &c in counts {
        let p = c as f64 / n_f;
        sum_sq += p * p;
    }
    1.0 - sum_sq
}

/// Entropy: -sum(p_c * log2(p_c))
pub fn entropy(counts: &[u32], n: u32) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let mut ent = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / n_f;
            ent -= p * p.log2();
        }
    }
    ent
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    // ── Gini tests ───────────────────────────────────────────────
    #[test]
    fn gini_pure_node() {
        assert_eq!(gini(&[10, 0], 10), 0.0);
        assert_eq!(gini(&[0, 5], 5), 0.0);
    }

    #[test]
    fn gini_uniform_binary() {
        let g = gini(&[5, 5], 10);
        assert!((g - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gini_uniform_three_class() {
        let g = gini(&[10, 10, 10], 30);
        assert!((g - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn gini_asymmetric() {
        let g = gini(&[9, 1], 10);
        assert!((g - 0.18).abs() < 1e-12);
    }

    #[test]
    fn gini_single_sample() {
        assert_eq!(gini(&[1, 0], 1), 0.0);
    }

    #[test]
    fn gini_zero_samples() {
        assert_eq!(gini(&[0, 0], 0), 0.0);
    }

    // ── Entropy tests ────────────────────────────────────────────
    #[test]
    fn entropy_pure_node() {
        assert_eq!(entropy(&[10, 0], 10), 0.0);
    }

    #[test]
    fn entropy_uniform_binary() {
        let e = entropy(&[5, 5], 10);
        assert!((e - 1.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_uniform_three_class() {
        let e = entropy(&[10, 10, 10], 30);
        assert!((e - 3.0_f64.log2()).abs() < 1e-12);
    }

    #[test]
    fn entropy_single_sample() {
        assert_eq!(entropy(&[1, 0], 1), 0.0);
    }

    #[test]
    fn entropy_zero_samples() {
        assert_eq!(entropy(&[0, 0], 0), 0.0);
    }

    // ── Criterion dispatch ───────────────────────────────────────
    #[test]
    fn criterion_dispatch_gini() {
        let g = impurity(Criterion::Gini, &[5, 5], 10);
        assert!((g - 0.5).abs() < 1e-12);
    }

    #[test]
    fn criterion_dispatch_entropy() {
        let e = impurity(Criterion::Entropy, &[5, 5], 10);
        assert!((e - 1.0).abs() < 1e-12);
    }

    #[test]
    fn criterion_from_str() {
        assert_eq!(Criterion::from_str("gini").unwrap(), Criterion::Gini);
        assert_eq!(Criterion::from_str("entropy").unwrap(), Criterion::Entropy);
        assert!(Criterion::from_str("bad").is_err());
    }
}
