use crate::impurity::{impurity as compute_impurity, Criterion};
use crate::presorted::PresortedRows;
use std::cmp::Ordering;
use std::ops::Range;

const MIN_GAIN: f64 = 1e-15;

struct FeatureVal {
    x: f64,
    y: u8,
}

struct SplitResult {
    feature_idx: usize,
    threshold: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

struct CartResult {
    misclassified: u32,
    decision_nodes: u32,
    witness: Option<CartWitness>,
}

#[derive(Clone, Debug)]
pub struct CartSubtreeResult {
    pub local_indices: Vec<usize>,
    pub remaining_depth: usize,
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub enum CartWitness {
    Leaf {
        local_indices: Vec<usize>,
        remaining_depth: usize,
        pred_class: u8,
        cost: f64,
    },
    Split {
        local_indices: Vec<usize>,
        remaining_depth: usize,
        feature_idx: usize,
        threshold: f64,
        cost: f64,
        left: Box<CartWitness>,
        right: Box<CartWitness>,
    },
}

/// Find the best split across all features for given sample indices.
/// Returns the left/right partition if a valid split is found.
fn find_best_split(
    x: &[f64],
    y: &[u8],
    indices: &[usize],
    n_features: usize,
    n_classes: usize,
    parent_impurity: f64,
    criterion: Criterion,
) -> Option<SplitResult> {
    let n_samples = indices.len();
    if n_samples < 2 {
        return None;
    }

    let mut total_counts = vec![0u32; n_classes];
    for &idx in indices {
        total_counts[y[idx] as usize] += 1;
    }

    let mut best_gain: f64 = 0.0;
    let mut best_feature: usize = 0;
    let mut best_threshold: f64 = 0.0;

    let mut data_buffer: Vec<FeatureVal> = Vec::with_capacity(n_samples);

    for feat_idx in 0..n_features {
        data_buffer.clear();
        for &idx in indices {
            data_buffer.push(FeatureVal {
                x: x[idx * n_features + feat_idx],
                y: y[idx],
            });
        }
        data_buffer.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));

        let mut left_counts = vec![0u32; n_classes];
        let mut right_counts = total_counts.clone();
        let mut n_left: u32 = 0;
        let mut n_right = n_samples as u32;

        for i in 0..(n_samples - 1) {
            let yc = data_buffer[i].y as usize;
            left_counts[yc] += 1;
            right_counts[yc] -= 1;
            n_left += 1;
            n_right -= 1;

            if data_buffer[i].x == data_buffer[i + 1].x {
                continue;
            }

            let imp_l = compute_impurity(criterion, &left_counts, n_left);
            let imp_r = compute_impurity(criterion, &right_counts, n_right);
            let w_imp = (n_left as f64 / n_samples as f64) * imp_l
                + (n_right as f64 / n_samples as f64) * imp_r;
            let gain = parent_impurity - w_imp;

            if gain > best_gain && gain > MIN_GAIN {
                best_gain = gain;
                best_feature = feat_idx;
                best_threshold = (data_buffer[i].x + data_buffer[i + 1].x) / 2.0;
            }
        }
    }

    if best_gain > MIN_GAIN {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &idx in indices {
            if x[idx * n_features + best_feature] <= best_threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }
        Some(SplitResult {
            feature_idx: best_feature,
            threshold: best_threshold,
            left_indices,
            right_indices,
        })
    } else {
        None
    }
}

/// Recursive depth-limited CART.
fn cart_recurse(
    x: &[f64],
    y: &[u8],
    indices: &[usize],
    n_features: usize,
    n_classes: usize,
    depth: usize,
    lambda_: f64,
    criterion: Criterion,
    mut completed: Option<&mut Vec<CartSubtreeResult>>,
) -> CartResult {
    let n_samples = indices.len();

    // Compute class counts and majority count
    let mut counts = vec![0u32; n_classes];
    let mut max_count: u32 = 0;
    let mut pred_class: u8 = 0;
    for &idx in indices {
        let c = y[idx] as usize;
        counts[c] += 1;
        if counts[c] > max_count {
            max_count = counts[c];
            pred_class = c as u8;
        }
    }

    let misclassified_here = n_samples as u32 - max_count;
    let collect_witness = completed.is_some();

    let result = if depth == 0 || n_samples < 2 || misclassified_here == 0 {
        CartResult {
            misclassified: misclassified_here,
            decision_nodes: 0,
            witness: collect_witness.then(|| CartWitness::Leaf {
                local_indices: indices.to_vec(),
                remaining_depth: depth,
                pred_class,
                cost: misclassified_here as f64,
            }),
        }
    } else {
        let parent_impurity = compute_impurity(criterion, &counts, n_samples as u32);
        match find_best_split(
            x,
            y,
            indices,
            n_features,
            n_classes,
            parent_impurity,
            criterion,
        ) {
            None => CartResult {
                misclassified: misclassified_here,
                decision_nodes: 0,
                witness: collect_witness.then(|| CartWitness::Leaf {
                    local_indices: indices.to_vec(),
                    remaining_depth: depth,
                    pred_class,
                    cost: misclassified_here as f64,
                }),
            },
            Some(s) => {
                let left = cart_recurse(
                    x,
                    y,
                    &s.left_indices,
                    n_features,
                    n_classes,
                    depth - 1,
                    lambda_,
                    criterion,
                    completed.as_deref_mut(),
                );
                let right = cart_recurse(
                    x,
                    y,
                    &s.right_indices,
                    n_features,
                    n_classes,
                    depth - 1,
                    lambda_,
                    criterion,
                    completed.as_deref_mut(),
                );
                let misclassified = left.misclassified + right.misclassified;
                let decision_nodes = 1 + left.decision_nodes + right.decision_nodes;
                let cost = misclassified as f64 + lambda_ * decision_nodes as f64;
                CartResult {
                    misclassified,
                    decision_nodes,
                    witness: collect_witness.then(|| CartWitness::Split {
                        local_indices: indices.to_vec(),
                        remaining_depth: depth,
                        feature_idx: s.feature_idx,
                        threshold: s.threshold,
                        cost,
                        left: Box::new(left.witness.unwrap()),
                        right: Box::new(right.witness.unwrap()),
                    }),
                }
            }
        }
    };

    if let Some(completed) = completed {
        completed.push(CartSubtreeResult {
            local_indices: indices.to_vec(),
            remaining_depth: depth,
            cost: result.misclassified as f64 + lambda_ * result.decision_nodes as f64,
        });
    }

    result
}

fn cart_recurse_presorted(
    x: &[f64],
    y: &[u8],
    rows: &mut PresortedRows,
    range: Range<usize>,
    n_features: usize,
    n_classes: usize,
    depth: usize,
    lambda_: f64,
    criterion: Criterion,
    mut completed: Option<&mut Vec<CartSubtreeResult>>,
) -> CartResult {
    let n_samples = range.len();
    // Descendant partitions mutate this range, so retain its canonical order first.
    let node_indices = completed
        .is_some()
        .then(|| rows.local_order(range.clone()).to_vec());
    let mut counts = vec![0u32; n_classes];
    let mut max_count = 0;
    let mut pred_class = 0;
    for &idx in rows.local_order(range.clone()) {
        let class = y[idx] as usize;
        counts[class] += 1;
        if counts[class] > max_count {
            max_count = counts[class];
            pred_class = class as u8;
        }
    }

    let misclassified_here = n_samples as u32 - max_count;
    let collect_witness = completed.is_some();
    let result = if depth == 0 || n_samples < 2 || misclassified_here == 0 {
        CartResult {
            misclassified: misclassified_here,
            decision_nodes: 0,
            witness: collect_witness.then(|| CartWitness::Leaf {
                local_indices: node_indices.clone().unwrap(),
                remaining_depth: depth,
                pred_class,
                cost: misclassified_here as f64,
            }),
        }
    } else {
        let parent_impurity = compute_impurity(criterion, &counts, n_samples as u32);
        match rows.best_split(
            x,
            y,
            range.clone(),
            n_features,
            n_classes,
            criterion,
            parent_impurity,
        ) {
            None => CartResult {
                misclassified: misclassified_here,
                decision_nodes: 0,
                witness: collect_witness.then(|| CartWitness::Leaf {
                    local_indices: node_indices.clone().unwrap(),
                    remaining_depth: depth,
                    pred_class,
                    cost: misclassified_here as f64,
                }),
            },
            Some(split) => {
                let middle = rows.partition(
                    x,
                    range.clone(),
                    n_features,
                    split.feature_idx,
                    split.threshold,
                );
                let left = cart_recurse_presorted(
                    x,
                    y,
                    rows,
                    range.start..middle,
                    n_features,
                    n_classes,
                    depth - 1,
                    lambda_,
                    criterion,
                    completed.as_deref_mut(),
                );
                let right = cart_recurse_presorted(
                    x,
                    y,
                    rows,
                    middle..range.end,
                    n_features,
                    n_classes,
                    depth - 1,
                    lambda_,
                    criterion,
                    completed.as_deref_mut(),
                );
                let misclassified = left.misclassified + right.misclassified;
                let decision_nodes = 1 + left.decision_nodes + right.decision_nodes;
                let cost = misclassified as f64 + lambda_ * decision_nodes as f64;
                CartResult {
                    misclassified,
                    decision_nodes,
                    witness: collect_witness.then(|| CartWitness::Split {
                        local_indices: node_indices.clone().unwrap(),
                        remaining_depth: depth,
                        feature_idx: split.feature_idx,
                        threshold: split.threshold,
                        cost,
                        left: Box::new(left.witness.unwrap()),
                        right: Box::new(right.witness.unwrap()),
                    }),
                }
            }
        }
    };

    if let Some(completed) = completed {
        completed.push(CartSubtreeResult {
            local_indices: node_indices.unwrap(),
            remaining_depth: depth,
            cost: result.misclassified as f64 + lambda_ * result.decision_nodes as f64,
        });
    }
    result
}

/// Fit a depth-limited greedy CART tree and return penalized cost.
///
/// Returns: misclassified_count + lambda_ * decision_node_count
///
/// # Arguments
/// * `x` - Feature matrix, flattened row-major (n_samples * n_features)
/// * `y` - Class labels (0..n_classes-1), length n_samples
/// * `n_features` - Number of features
/// * `lambda_` - Penalty per decision node
/// * `max_depth` - Maximum tree depth (0 = leaf only)
/// * `criterion` - Gini or Entropy
pub fn cart_heuristic(
    x: &[f64],
    y: &[u8],
    n_features: usize,
    lambda_: f64,
    max_depth: usize,
    criterion: Criterion,
) -> f64 {
    let n_samples = y.len();
    if n_samples == 0 {
        return 0.0;
    }

    // Determine n_classes from data
    let n_classes = y.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

    let result = match (max_depth > 1)
        .then(|| PresortedRows::new(x, n_samples, n_features))
        .flatten()
    {
        Some(mut rows) => cart_recurse_presorted(
            x,
            y,
            &mut rows,
            0..n_samples,
            n_features,
            n_classes,
            max_depth,
            lambda_,
            criterion,
            None,
        ),
        None => {
            let indices: Vec<usize> = (0..n_samples).collect();
            cart_recurse(
                x, y, &indices, n_features, n_classes, max_depth, lambda_, criterion, None,
            )
        }
    };

    result.misclassified as f64 + lambda_ * result.decision_nodes as f64
}

pub fn cart_heuristic_with_subtrees(
    x: &[f64],
    y: &[u8],
    n_features: usize,
    lambda_: f64,
    max_depth: usize,
    criterion: Criterion,
) -> (f64, Vec<CartSubtreeResult>, Option<CartWitness>) {
    if y.is_empty() {
        return (0.0, Vec::new(), None);
    }
    let n_classes = y.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;
    let mut completed = Vec::new();
    let result = match (max_depth > 1)
        .then(|| PresortedRows::new(x, y.len(), n_features))
        .flatten()
    {
        Some(mut rows) => cart_recurse_presorted(
            x,
            y,
            &mut rows,
            0..y.len(),
            n_features,
            n_classes,
            max_depth,
            lambda_,
            criterion,
            Some(&mut completed),
        ),
        None => {
            let indices: Vec<usize> = (0..y.len()).collect();
            cart_recurse(
                x,
                y,
                &indices,
                n_features,
                n_classes,
                max_depth,
                lambda_,
                criterion,
                Some(&mut completed),
            )
        }
    };
    (
        result.misclassified as f64 + lambda_ * result.decision_nodes as f64,
        completed,
        result.witness,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impurity::Criterion;

    #[test]
    fn depth_zero_is_leaf_only() {
        // depth=0 → majority vote, no splits
        // Labels [0,0,0,1,1] → 2 misclassified, 0 decision nodes
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1];
        let cost = cart_heuristic(&x, &y, 1, 1.0, 0, Criterion::Gini);
        assert!((cost - 2.0).abs() < 1e-12);
    }

    #[test]
    fn perfect_separation_depth_1() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let cost = cart_heuristic(&x, &y, 1, 0.0, 1, Criterion::Gini);
        assert!((cost - 0.0).abs() < 1e-12);
    }

    #[test]
    fn lambda_penalty_applied() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        // lambda=5.0 → cost = 0 + 5.0 * 1 = 5.0
        let cost = cart_heuristic(&x, &y, 1, 5.0, 1, Criterion::Gini);
        assert!((cost - 5.0).abs() < 1e-12);
    }

    #[test]
    fn single_class_no_splits() {
        let x = vec![1.0, 2.0, 3.0];
        let y: Vec<u8> = vec![0, 0, 0];
        let cost = cart_heuristic(&x, &y, 1, 1.0, 5, Criterion::Gini);
        assert!((cost - 0.0).abs() < 1e-12);
    }

    #[test]
    fn single_sample() {
        let x = vec![3.0];
        let y: Vec<u8> = vec![1];
        let cost = cart_heuristic(&x, &y, 1, 1.0, 3, Criterion::Gini);
        assert!((cost - 0.0).abs() < 1e-12);
    }

    #[test]
    fn multiclass() {
        // 3 classes perfectly separable in 1D with depth 2
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cost = cart_heuristic(&x, &y, 1, 0.0, 2, Criterion::Gini);
        assert!((cost - 0.0).abs() < 1e-12);
    }

    #[test]
    fn multi_feature_picks_best() {
        // Feature 0: informative, Feature 1: noise
        let x = vec![1.0, 5.0, 2.0, 3.0, 3.0, 7.0, 4.0, 2.0, 5.0, 8.0, 6.0, 1.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let cost = cart_heuristic(&x, &y, 2, 0.0, 1, Criterion::Gini);
        assert!((cost - 0.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_criterion() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let cost = cart_heuristic(&x, &y, 1, 0.0, 1, Criterion::Entropy);
        assert!((cost - 0.0).abs() < 1e-12);
    }

    #[test]
    fn deeper_tree_fewer_misclassifications() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y: Vec<u8> = vec![0, 1, 0, 1, 1, 0, 1, 0];
        let cost_d1 = cart_heuristic(&x, &y, 1, 0.0, 1, Criterion::Gini);
        let cost_d3 = cart_heuristic(&x, &y, 1, 0.0, 3, Criterion::Gini);
        assert!(cost_d3 <= cost_d1 + 1e-12);
    }

    #[test]
    fn subtree_and_witness_indices_keep_local_order() {
        let x = vec![2.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 3.0, 5.0, 4.0, 4.0];
        let y = vec![0, 0, 1, 1, 0, 1];
        let (cost, completed, witness) =
            cart_heuristic_with_subtrees(&x, &y, 2, 0.25, 2, Criterion::Gini);

        assert_eq!(cost, 1.5);
        assert_eq!(
            completed
                .iter()
                .map(|subtree| (
                    subtree.local_indices.clone(),
                    subtree.remaining_depth,
                    subtree.cost
                ))
                .collect::<Vec<_>>(),
            vec![
                (vec![4, 5], 0, 1.0),
                (vec![2, 3], 0, 0.0),
                (vec![2, 3, 4, 5], 1, 1.25),
                (vec![0, 1], 1, 0.0),
                (vec![0, 1, 2, 3, 4, 5], 2, 1.5),
            ]
        );
        match witness.unwrap() {
            CartWitness::Split {
                local_indices,
                left,
                right,
                ..
            } => {
                assert_eq!(local_indices, vec![0, 1, 2, 3, 4, 5]);
                assert_eq!(witness_indices(&left), &[2, 3, 4, 5]);
                assert_eq!(witness_indices(&right), &[0, 1]);
            }
            CartWitness::Leaf { .. } => panic!("expected split witness"),
        }
    }

    #[test]
    fn preserves_legacy_nan_cart_result() {
        let x = vec![f64::NAN, 0.0, 1.0, 2.0, 3.0];
        let y = vec![0, 0, 1, 1, 0];
        let (cost, completed, witness) =
            cart_heuristic_with_subtrees(&x, &y, 1, 0.25, 2, Criterion::Gini);

        assert_eq!(cost, 2.5);
        assert_eq!(completed.last().unwrap().local_indices, vec![0, 1, 2, 3, 4]);
        match witness.unwrap() {
            CartWitness::Split { threshold, .. } => assert_eq!(threshold, 0.5),
            CartWitness::Leaf { .. } => panic!("expected split witness"),
        }
    }

    #[test]
    fn presorted_matches_legacy_cost_and_witness_for_finite_inputs() {
        let mut seed = 11u64;
        for criterion in [Criterion::Gini, Criterion::Entropy] {
            for n_samples in 2..24 {
                let mut x = Vec::with_capacity(n_samples * 4);
                let mut y = Vec::with_capacity(n_samples);
                for _ in 0..n_samples {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    y.push(((seed >> 32) % 3) as u8);
                    for _ in 0..4 {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        x.push(((seed >> 32) % 7) as f64);
                    }
                }

                let (actual_cost, actual_subtrees, actual_witness) =
                    cart_heuristic_with_subtrees(&x, &y, 4, 0.25, 3, criterion);
                let indices: Vec<usize> = (0..n_samples).collect();
                let mut expected_subtrees = Vec::new();
                let expected = cart_recurse(
                    &x,
                    &y,
                    &indices,
                    4,
                    3,
                    3,
                    0.25,
                    criterion,
                    Some(&mut expected_subtrees),
                );
                let expected_cost =
                    expected.misclassified as f64 + 0.25 * expected.decision_nodes as f64;
                assert_eq!(actual_cost, expected_cost);
                assert_subtrees_eq(&actual_subtrees, &expected_subtrees);
                assert_witness_eq(
                    actual_witness.as_ref().unwrap(),
                    expected.witness.as_ref().unwrap(),
                );
            }
        }
    }

    fn witness_indices(witness: &CartWitness) -> &[usize] {
        match witness {
            CartWitness::Leaf { local_indices, .. } | CartWitness::Split { local_indices, .. } => {
                local_indices
            }
        }
    }

    fn assert_subtrees_eq(actual: &[CartSubtreeResult], expected: &[CartSubtreeResult]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.local_indices, expected.local_indices);
            assert_eq!(actual.remaining_depth, expected.remaining_depth);
            assert_eq!(actual.cost, expected.cost);
        }
    }

    fn assert_witness_eq(actual: &CartWitness, expected: &CartWitness) {
        match (actual, expected) {
            (
                CartWitness::Leaf {
                    local_indices: actual_indices,
                    remaining_depth: actual_depth,
                    pred_class: actual_class,
                    cost: actual_cost,
                },
                CartWitness::Leaf {
                    local_indices: expected_indices,
                    remaining_depth: expected_depth,
                    pred_class: expected_class,
                    cost: expected_cost,
                },
            ) => {
                assert_eq!(actual_indices, expected_indices);
                assert_eq!(actual_depth, expected_depth);
                assert_eq!(actual_class, expected_class);
                assert_eq!(actual_cost, expected_cost);
            }
            (
                CartWitness::Split {
                    local_indices: actual_indices,
                    remaining_depth: actual_depth,
                    feature_idx: actual_feature,
                    threshold: actual_threshold,
                    cost: actual_cost,
                    left: actual_left,
                    right: actual_right,
                },
                CartWitness::Split {
                    local_indices: expected_indices,
                    remaining_depth: expected_depth,
                    feature_idx: expected_feature,
                    threshold: expected_threshold,
                    cost: expected_cost,
                    left: expected_left,
                    right: expected_right,
                },
            ) => {
                assert_eq!(actual_indices, expected_indices);
                assert_eq!(actual_depth, expected_depth);
                assert_eq!(actual_feature, expected_feature);
                assert_eq!(actual_threshold, expected_threshold);
                assert_eq!(actual_cost, expected_cost);
                assert_witness_eq(actual_left, expected_left);
                assert_witness_eq(actual_right, expected_right);
            }
            _ => panic!("witness variants differ"),
        }
    }
}
