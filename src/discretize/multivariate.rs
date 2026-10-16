use crate::impurity::{impurity as compute_impurity, Criterion};
use crate::presorted::{BestSplit as PresortedSplit, PresortedRows};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::ops::Range;

/// Per-feature discretization result extracted from a global multivariate CART tree.
#[derive(Debug, Clone)]
pub struct MultivarDiscretizeResult {
    /// feature_index -> thresholds (sorted bin boundaries)
    pub feature_thresholds: BTreeMap<usize, Vec<f64>>,
    /// feature_index -> per-bin sample indices
    pub feature_bin_indices: BTreeMap<usize, Vec<Vec<usize>>>,
    /// feature_index -> per-bin class distributions
    pub feature_bin_distributions: BTreeMap<usize, Vec<Vec<u32>>>,
}

impl MultivarDiscretizeResult {
    /// Conservative retained heap accounting for the exact-result cache.
    pub fn allocated_bytes(&self) -> usize {
        const BTREE_ENTRY_OVERHEAD: usize = 256;
        let mut bytes = std::mem::size_of::<Self>();
        for thresholds in self.feature_thresholds.values() {
            bytes = bytes.saturating_add(BTREE_ENTRY_OVERHEAD);
            bytes = bytes.saturating_add(thresholds.capacity() * std::mem::size_of::<f64>());
        }
        for bins in self.feature_bin_indices.values() {
            bytes = bytes.saturating_add(BTREE_ENTRY_OVERHEAD);
            bytes = bytes.saturating_add(bins.capacity() * std::mem::size_of::<Vec<usize>>());
            for bin in bins {
                bytes = bytes.saturating_add(bin.capacity() * std::mem::size_of::<usize>());
            }
        }
        for distributions in self.feature_bin_distributions.values() {
            bytes = bytes.saturating_add(BTREE_ENTRY_OVERHEAD);
            bytes =
                bytes.saturating_add(distributions.capacity() * std::mem::size_of::<Vec<u32>>());
            for distribution in distributions {
                bytes = bytes.saturating_add(distribution.capacity() * std::mem::size_of::<u32>());
            }
        }
        bytes
    }
}

// ── Internal types ──────────────────────────────────────────────────────────

struct FeatureVal {
    x: f64,
    y: u8,
}

struct SplitInfo {
    feature_idx: usize,
    threshold: f64,
    gain: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
    left_impurity: f64,
    right_impurity: f64,
}

struct Node {
    _id: usize, // unique creation ID; used via HeapEntry.creation_id for tie-breaking
    _indices: Vec<usize>, // sample indices at this node; kept for debugging
    _impurity: f64, // node impurity; kept for debugging/future use
    is_leaf: bool,
    best_split: Option<SplitInfo>,
}

struct HeapEntry {
    weighted_gain: f64,
    node_id: usize,
    creation_id: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.weighted_gain == other.weighted_gain && self.creation_id == other.creation_id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        if (self.weighted_gain - other.weighted_gain).abs() < 1e-12 {
            other.creation_id.cmp(&self.creation_id)
        } else {
            self.weighted_gain
                .partial_cmp(&other.weighted_gain)
                .unwrap_or(Ordering::Equal)
        }
    }
}

const MIN_GAIN: f64 = 1e-15;

/// Find the best split across all features for a set of sample indices.
fn find_best_split_multivariate(
    x: &[f64], // flattened row-major
    y: &[u8],
    indices: &[usize],
    n_features: usize,
    n_classes: usize,
    criterion: Criterion,
    parent_impurity: f64,
) -> Option<SplitInfo> {
    let n_samples = indices.len();
    if n_samples < 2 {
        return None;
    }

    // Compute total counts
    let mut total_counts = vec![0u32; n_classes];
    for &idx in indices {
        total_counts[y[idx] as usize] += 1;
    }

    let mut best_gain: f64 = 0.0;
    let mut best_feature: usize = 0;
    let mut best_threshold: f64 = 0.0;
    let mut best_left_imp: f64 = 0.0;
    let mut best_right_imp: f64 = 0.0;

    // For each feature, sort samples by that feature and scan for best split
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
                best_left_imp = imp_l;
                best_right_imp = imp_r;
            }
        }
    }

    if best_gain > MIN_GAIN {
        // Partition indices based on the best split
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &idx in indices {
            if x[idx * n_features + best_feature] <= best_threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }
        Some(SplitInfo {
            feature_idx: best_feature,
            threshold: best_threshold,
            gain: best_gain,
            left_indices,
            right_indices,
            left_impurity: best_left_imp,
            right_impurity: best_right_imp,
        })
    } else {
        None
    }
}

/// Fit a best-first multivariate CART tree and extract per-feature discretization.
///
/// # Arguments
/// * `x` - Feature matrix, flattened row-major (n_samples * n_features)
/// * `y` - Class labels (0..n_classes-1), length n_samples
/// * `n_features` - Number of features
/// * `n_classes` - Number of distinct classes
/// * `max_leaf_nodes` - Maximum number of leaves
/// * `criterion` - Gini or Entropy
fn fit_multivariate_cart_legacy(
    x: &[f64],
    y: &[u8],
    n_features: usize,
    n_classes: usize,
    max_leaf_nodes: usize,
    criterion: Criterion,
) -> MultivarDiscretizeResult {
    let n_samples = y.len();

    // 1. Root node
    let mut root_counts = vec![0u32; n_classes];
    for i in 0..n_samples {
        root_counts[y[i] as usize] += 1;
    }
    let root_impurity = compute_impurity(criterion, &root_counts, n_samples as u32);
    let root_indices: Vec<usize> = (0..n_samples).collect();

    let root_split = find_best_split_multivariate(
        x,
        y,
        &root_indices,
        n_features,
        n_classes,
        criterion,
        root_impurity,
    );

    let mut nodes: Vec<Node> = Vec::new();
    let mut next_id: usize = 0;
    let mut split_records: Vec<(usize, f64)> = Vec::new();

    nodes.push(Node {
        _id: next_id,
        _indices: root_indices,
        _impurity: root_impurity,
        is_leaf: true,
        best_split: root_split,
    });
    next_id += 1;

    // 2. Priority queue
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    if nodes[0].best_split.is_some() {
        let wg = nodes[0].best_split.as_ref().unwrap().gain * (n_samples as f64);
        heap.push(HeapEntry {
            weighted_gain: wg,
            node_id: 0,
            creation_id: 0,
        });
    }

    let mut current_leaf_count: usize = 1;

    // 3. Growth loop — record splits as we go
    while current_leaf_count < max_leaf_nodes {
        let entry = match heap.pop() {
            Some(e) => e,
            None => break,
        };

        let parent_id = entry.node_id;
        let split = nodes[parent_id].best_split.take().unwrap();
        nodes[parent_id].is_leaf = false;

        // Record this split
        split_records.push((split.feature_idx, split.threshold));

        // Left child
        let left_imp = split.left_impurity;
        let left_indices = split.left_indices;
        let left_split = find_best_split_multivariate(
            x,
            y,
            &left_indices,
            n_features,
            n_classes,
            criterion,
            left_imp,
        );
        let left_id = next_id;
        next_id += 1;
        let left_n = left_indices.len();
        nodes.push(Node {
            _id: left_id,
            _indices: left_indices,
            _impurity: left_imp,
            is_leaf: true,
            best_split: left_split,
        });

        // Right child
        let right_imp = split.right_impurity;
        let right_indices = split.right_indices;
        let right_split = find_best_split_multivariate(
            x,
            y,
            &right_indices,
            n_features,
            n_classes,
            criterion,
            right_imp,
        );
        let right_id = next_id;
        next_id += 1;
        let right_n = right_indices.len();
        nodes.push(Node {
            _id: right_id,
            _indices: right_indices,
            _impurity: right_imp,
            is_leaf: true,
            best_split: right_split,
        });

        // Push to heap
        if nodes[left_id].best_split.is_some() {
            let wg = nodes[left_id].best_split.as_ref().unwrap().gain * (left_n as f64);
            heap.push(HeapEntry {
                weighted_gain: wg,
                node_id: left_id,
                creation_id: left_id,
            });
        }
        if nodes[right_id].best_split.is_some() {
            let wg = nodes[right_id].best_split.as_ref().unwrap().gain * (right_n as f64);
            heap.push(HeapEntry {
                weighted_gain: wg,
                node_id: right_id,
                creation_id: right_id,
            });
        }

        current_leaf_count += 1;
    }

    build_result(x, y, n_features, n_classes, n_samples, split_records)
}

struct PresortedNode {
    range: Range<usize>,
    best_split: Option<PresortedSplit>,
}

/// Fit a best-first multivariate CART tree and extract per-feature discretization.
pub fn fit_multivariate_cart(
    x: &[f64],
    y: &[u8],
    n_features: usize,
    n_classes: usize,
    max_leaf_nodes: usize,
    criterion: Criterion,
) -> MultivarDiscretizeResult {
    let n_samples = y.len();
    let Some(mut rows) = PresortedRows::new(x, n_samples, n_features) else {
        return fit_multivariate_cart_legacy(
            x,
            y,
            n_features,
            n_classes,
            max_leaf_nodes,
            criterion,
        );
    };

    let mut root_counts = vec![0u32; n_classes];
    for &class in y {
        root_counts[class as usize] += 1;
    }
    let root_impurity = compute_impurity(criterion, &root_counts, n_samples as u32);
    let root_range = 0..n_samples;
    let root_split = rows.best_split(
        x,
        y,
        root_range.clone(),
        n_features,
        n_classes,
        criterion,
        root_impurity,
    );

    let mut nodes = vec![PresortedNode {
        range: root_range,
        best_split: root_split,
    }];
    let mut split_records = Vec::new();
    let mut heap = BinaryHeap::new();
    if let Some(split) = &nodes[0].best_split {
        heap.push(HeapEntry {
            weighted_gain: split.gain * n_samples as f64,
            node_id: 0,
            creation_id: 0,
        });
    }

    let mut leaf_count = 1;
    while leaf_count < max_leaf_nodes {
        let Some(entry) = heap.pop() else { break };
        let parent_id = entry.node_id;
        let split = nodes[parent_id].best_split.take().unwrap();
        let parent_range = nodes[parent_id].range.clone();
        split_records.push((split.feature_idx, split.threshold));
        let middle = rows.partition(
            x,
            parent_range.clone(),
            n_features,
            split.feature_idx,
            split.threshold,
        );

        for (range, impurity) in [
            (parent_range.start..middle, split.left_impurity),
            (middle..parent_range.end, split.right_impurity),
        ] {
            let best_split = rows.best_split(
                x,
                y,
                range.clone(),
                n_features,
                n_classes,
                criterion,
                impurity,
            );
            let node_id = nodes.len();
            let sample_count = range.len();
            nodes.push(PresortedNode { range, best_split });
            if let Some(child_split) = &nodes[node_id].best_split {
                heap.push(HeapEntry {
                    weighted_gain: child_split.gain * sample_count as f64,
                    node_id,
                    creation_id: node_id,
                });
            }
        }
        leaf_count += 1;
    }

    build_result(x, y, n_features, n_classes, n_samples, split_records)
}

fn build_result(
    x: &[f64],
    y: &[u8],
    n_features: usize,
    n_classes: usize,
    n_samples: usize,
    split_records: Vec<(usize, f64)>,
) -> MultivarDiscretizeResult {
    // Build per-feature thresholds from split records
    let mut feat_thresholds: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for &(feat_idx, thr) in &split_records {
        feat_thresholds.entry(feat_idx).or_default().push(thr);
    }

    // Sort thresholds per feature
    for thresholds in feat_thresholds.values_mut() {
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    }

    // Bin all samples in original local order.
    let mut feature_bin_indices: BTreeMap<usize, Vec<Vec<usize>>> = BTreeMap::new();
    let mut feature_bin_distributions: BTreeMap<usize, Vec<Vec<u32>>> = BTreeMap::new();

    for (&feat_idx, thresholds) in &feat_thresholds {
        let n_bins = thresholds.len() + 1;
        let mut bin_indices: Vec<Vec<usize>> = vec![Vec::new(); n_bins];
        let mut bin_dists: Vec<Vec<u32>> = vec![vec![0u32; n_classes]; n_bins];

        for i in 0..n_samples {
            let val = x[i * n_features + feat_idx];
            let y_val = y[i] as usize;

            // Find bin: bin 0 if val <= threshold[0], etc.
            let mut bin = thresholds.len(); // default to last bin
            for (k, &t) in thresholds.iter().enumerate() {
                if val <= t {
                    bin = k;
                    break;
                }
            }

            bin_indices[bin].push(i);
            bin_dists[bin][y_val] += 1;
        }

        feature_bin_indices.insert(feat_idx, bin_indices);
        feature_bin_distributions.insert(feat_idx, bin_dists);
    }

    MultivarDiscretizeResult {
        feature_thresholds: feat_thresholds,
        feature_bin_indices,
        feature_bin_distributions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impurity::Criterion;

    #[test]
    fn two_features_picks_best() {
        // Feature 0: [1,2,3,4], Feature 1: [1,1,1,1] (uninformative)
        // Labels: [0,0,1,1] → should split on feature 0
        let x = vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0];
        let y: Vec<u8> = vec![0, 0, 1, 1];
        let result = fit_multivariate_cart(&x, &y, 2, 2, 2, Criterion::Gini);
        assert!(result.feature_thresholds.contains_key(&0));
        assert_eq!(result.feature_thresholds[&0].len(), 1);
    }

    #[test]
    fn single_feature_matches_univariate() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let result = fit_multivariate_cart(&x, &y, 1, 2, 2, Criterion::Gini);
        assert!(result.feature_thresholds.contains_key(&0));
        assert_eq!(result.feature_thresholds[&0].len(), 1);
        assert_eq!(result.feature_bin_indices[&0].len(), 2);
    }

    #[test]
    fn respects_max_leaf_nodes() {
        // 2 features, 3 classes
        let x = vec![
            1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0,
        ];
        let y: Vec<u8> = vec![0, 0, 1, 1, 2, 2];
        let result = fit_multivariate_cart(&x, &y, 2, 3, 2, Criterion::Gini);
        // At most 2 leaves → 1 split → 1 feature with 1 threshold
        let total_thresholds: usize = result.feature_thresholds.values().map(|t| t.len()).sum();
        assert_eq!(total_thresholds, 1);
    }

    #[test]
    fn pure_data_no_splits() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<u8> = vec![0, 0, 0, 0];
        let result = fit_multivariate_cart(&x, &y, 1, 1, 4, Criterion::Gini);
        assert!(result.feature_thresholds.is_empty());
    }

    #[test]
    fn all_bins_cover_all_samples() {
        let x = vec![1.0, 5.0, 2.0, 4.0, 3.0, 3.0, 4.0, 2.0, 5.0, 1.0];
        let y: Vec<u8> = vec![0, 0, 1, 1, 1];
        let result = fit_multivariate_cart(&x, &y, 2, 2, 3, Criterion::Gini);
        for bins in result.feature_bin_indices.values() {
            let mut all: Vec<usize> = bins.iter().flatten().cloned().collect();
            all.sort();
            all.dedup();
            assert_eq!(all.len(), 5);
        }
    }

    #[test]
    fn entropy_criterion() {
        let x = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let y: Vec<u8> = vec![0, 0, 1, 1];
        let result = fit_multivariate_cart(&x, &y, 2, 2, 2, Criterion::Entropy);
        let has_thresholds = result.feature_thresholds.values().any(|t| !t.is_empty());
        assert!(has_thresholds);
    }

    #[test]
    fn deterministic() {
        let x = vec![3.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0, 4.0];
        let y: Vec<u8> = vec![0, 1, 0, 1];
        let r1 = fit_multivariate_cart(&x, &y, 2, 2, 3, Criterion::Gini);
        let r2 = fit_multivariate_cart(&x, &y, 2, 2, 3, Criterion::Gini);
        assert_eq!(r1.feature_thresholds, r2.feature_thresholds);
    }

    #[test]
    fn distributions_sum_correctly() {
        let x = vec![1.0, 5.0, 2.0, 4.0, 3.0, 3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 1, 1, 0, 1];
        let result = fit_multivariate_cart(&x, &y, 2, 2, 3, Criterion::Gini);
        for bins in result.feature_bin_indices.values() {
            let total: usize = bins.iter().map(|b| b.len()).sum();
            assert_eq!(total, 6);
        }
        for (feat, dists) in &result.feature_bin_distributions {
            let bins = &result.feature_bin_indices[feat];
            for (i, dist) in dists.iter().enumerate() {
                let sum: u32 = dist.iter().sum();
                assert_eq!(sum as usize, bins[i].len());
            }
        }
    }

    #[test]
    fn preserves_original_sample_order_with_duplicate_values() {
        let x = vec![2.0, 1.0, 2.0, 4.0, 3.0, 4.0];
        let y = vec![0, 0, 0, 1, 1, 1];
        let result = fit_multivariate_cart(&x, &y, 1, 2, 2, Criterion::Gini);

        assert_eq!(result.feature_thresholds[&0], vec![2.5]);
        assert_eq!(
            result.feature_bin_indices[&0],
            vec![vec![0, 1, 2], vec![3, 4, 5]]
        );
        assert_eq!(
            result.feature_bin_distributions[&0],
            vec![vec![3, 0], vec![0, 3]]
        );
    }

    #[test]
    fn preserves_legacy_nan_ordering() {
        let x = vec![f64::NAN, 0.0, 1.0, 2.0, 3.0];
        let y = vec![0, 0, 1, 1, 0];
        let result = fit_multivariate_cart(&x, &y, 1, 2, 3, Criterion::Gini);

        let thresholds = &result.feature_thresholds[&0];
        assert_eq!(thresholds.len(), 2);
        assert_eq!(thresholds[0], 0.5);
        assert!(thresholds[1].is_nan());
        assert_eq!(
            result.feature_bin_indices[&0],
            vec![vec![1], vec![], vec![0, 2, 3, 4]]
        );
    }

    #[test]
    fn presorted_matches_legacy_for_finite_inputs() {
        let mut seed = 7u64;
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

                let actual = fit_multivariate_cart(&x, &y, 4, 3, 5, criterion);
                let expected = fit_multivariate_cart_legacy(&x, &y, 4, 3, 5, criterion);
                assert_eq!(actual.feature_thresholds, expected.feature_thresholds);
                assert_eq!(actual.feature_bin_indices, expected.feature_bin_indices);
                assert_eq!(
                    actual.feature_bin_distributions,
                    expected.feature_bin_distributions
                );
            }
        }
    }
}
