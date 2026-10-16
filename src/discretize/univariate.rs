use crate::impurity::{impurity as compute_impurity, Criterion};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Result of univariate discretization on a single feature.
#[derive(Debug, Clone)]
pub struct DiscretizeResult {
    /// Sorted threshold values (bin boundaries). Length = n_bins - 1.
    pub thresholds: Vec<f64>,
    /// Per-bin sample indices (indices into the input arrays). Length = n_bins.
    pub bin_indices: Vec<Vec<usize>>,
    /// Per-bin class count distributions. bin_distributions[b][c] = count of class c in bin b.
    pub bin_distributions: Vec<Vec<u32>>,
}

// ── Internal types ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct Sample {
    x: f64,
    y: u8,
    original_idx: usize,
}

struct SplitInfo {
    split_index: usize, // index in sorted samples where left ends (exclusive)
    gain: f64,
    _threshold: f64, // stored for documentation; extracted from sorted samples later
    left_counts: Vec<u32>,
    right_counts: Vec<u32>,
    left_impurity: f64,
    right_impurity: f64,
}

struct Node {
    _id: usize, // unique creation ID; used via HeapEntry.creation_id for tie-breaking
    start: usize,
    end: usize,
    _impurity: f64, // node impurity; kept for debugging/future use
    counts: Vec<u32>,
    is_leaf: bool,
    best_split: Option<SplitInfo>,
    left: Option<usize>, // index into nodes vec (not used for heap, just tree structure)
    right: Option<usize>,
}

/// Wrapper for the BinaryHeap. Rust's BinaryHeap is a max-heap by default,
/// which matches the C++ priority_queue behavior (max weighted gain at top).
struct HeapEntry {
    weighted_gain: f64,
    node_id: usize,
    /// Lower creation_id = higher priority on ties (older nodes first).
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
        // Max-heap by weighted_gain. On tie, prefer smaller creation_id (older node).
        let gain_cmp = self
            .weighted_gain
            .partial_cmp(&other.weighted_gain)
            .unwrap_or(Ordering::Equal);
        if (self.weighted_gain - other.weighted_gain).abs() < 1e-12 {
            // Tie: smaller creation_id = higher priority = Greater in max-heap
            other.creation_id.cmp(&self.creation_id)
        } else {
            gain_cmp
        }
    }
}

const MIN_GAIN: f64 = 1e-7;

/// Find the best split point within a sorted range of samples.
fn find_best_split(
    samples: &[Sample],
    start: usize,
    end: usize,
    total_counts: &[u32],
    parent_impurity: f64,
    n_classes: usize,
    criterion: Criterion,
) -> Option<SplitInfo> {
    let n_total = end - start;
    if n_total < 2 {
        return None;
    }

    let mut left_counts = vec![0u32; n_classes];
    let mut right_counts = total_counts.to_vec();

    let mut n_left: u32 = 0;
    let mut n_right = n_total as u32;

    let mut best: Option<SplitInfo> = None;
    let mut best_gain: f64 = 0.0;

    for i in start..(end - 1) {
        let y = samples[i].y as usize;
        left_counts[y] += 1;
        right_counts[y] -= 1;
        n_left += 1;
        n_right -= 1;

        // Skip if same feature value as next (can't split between equal values)
        if samples[i].x == samples[i + 1].x {
            continue;
        }

        let imp_l = compute_impurity(criterion, &left_counts, n_left);
        let imp_r = compute_impurity(criterion, &right_counts, n_right);

        let w_imp =
            (n_left as f64 / n_total as f64) * imp_l + (n_right as f64 / n_total as f64) * imp_r;
        let gain = parent_impurity - w_imp;

        if gain > best_gain && gain > MIN_GAIN {
            best_gain = gain;
            best = Some(SplitInfo {
                split_index: i + 1, // first index of right side
                gain,
                _threshold: (samples[i].x + samples[i + 1].x) / 2.0,
                left_counts: left_counts.clone(),
                right_counts: right_counts.clone(),
                left_impurity: imp_l,
                right_impurity: imp_r,
            });
        }
    }

    best
}

/// Fit a best-first CART tree on a single feature column and return bin structure.
///
/// # Arguments
/// * `x` - Feature values, length n_samples
/// * `y` - Class labels (0..n_classes-1), length n_samples
/// * `n_classes` - Number of distinct classes
/// * `max_leaf_nodes` - Maximum number of leaves (bins)
/// * `criterion` - Gini or Entropy
pub fn discretize(
    x: &[f64],
    y: &[u8],
    n_classes: usize,
    max_leaf_nodes: usize,
    criterion: Criterion,
) -> DiscretizeResult {
    let n_samples = x.len();

    // 1. Create and sort samples
    let mut samples: Vec<Sample> = (0..n_samples)
        .map(|i| Sample {
            x: x[i],
            y: y[i],
            original_idx: i,
        })
        .collect();
    samples.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));

    // 2. Compute root counts and impurity
    let mut root_counts = vec![0u32; n_classes];
    for s in &samples {
        root_counts[s.y as usize] += 1;
    }
    let root_impurity = compute_impurity(criterion, &root_counts, n_samples as u32);

    // 3. Create root node
    let mut nodes: Vec<Node> = Vec::new();
    let mut next_id: usize = 0;

    let root_split = find_best_split(
        &samples,
        0,
        n_samples,
        &root_counts,
        root_impurity,
        n_classes,
        criterion,
    );

    nodes.push(Node {
        _id: next_id,
        start: 0,
        end: n_samples,
        _impurity: root_impurity,
        counts: root_counts,
        is_leaf: true,
        best_split: root_split,
        left: None,
        right: None,
    });
    next_id += 1;

    // 4. Priority queue for best-first growth
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

    // 5. Growth loop
    while current_leaf_count < max_leaf_nodes {
        let entry = match heap.pop() {
            Some(e) => e,
            None => break,
        };

        let parent_id = entry.node_id;

        // Extract split info from parent
        let split = nodes[parent_id].best_split.take().unwrap();
        nodes[parent_id].is_leaf = false;

        let mid = split.split_index;
        let parent_start = nodes[parent_id].start;
        let parent_end = nodes[parent_id].end;

        // Create left child
        let left_split = find_best_split(
            &samples,
            parent_start,
            mid,
            &split.left_counts,
            split.left_impurity,
            n_classes,
            criterion,
        );
        let left_id = next_id;
        next_id += 1;
        nodes.push(Node {
            _id: left_id,
            start: parent_start,
            end: mid,
            _impurity: split.left_impurity,
            counts: split.left_counts,
            is_leaf: true,
            best_split: left_split,
            left: None,
            right: None,
        });

        // Create right child
        let right_split = find_best_split(
            &samples,
            mid,
            parent_end,
            &split.right_counts,
            split.right_impurity,
            n_classes,
            criterion,
        );
        let right_id = next_id;
        next_id += 1;
        nodes.push(Node {
            _id: right_id,
            start: mid,
            end: parent_end,
            _impurity: split.right_impurity,
            counts: split.right_counts,
            is_leaf: true,
            best_split: right_split,
            left: None,
            right: None,
        });

        nodes[parent_id].left = Some(left_id);
        nodes[parent_id].right = Some(right_id);

        // Push children to heap if they have valid splits
        if nodes[left_id].best_split.is_some() {
            let n_left = mid - parent_start;
            let wg = nodes[left_id].best_split.as_ref().unwrap().gain * (n_left as f64);
            heap.push(HeapEntry {
                weighted_gain: wg,
                node_id: left_id,
                creation_id: left_id,
            });
        }
        if nodes[right_id].best_split.is_some() {
            let n_right = parent_end - mid;
            let wg = nodes[right_id].best_split.as_ref().unwrap().gain * (n_right as f64);
            heap.push(HeapEntry {
                weighted_gain: wg,
                node_id: right_id,
                creation_id: right_id,
            });
        }

        current_leaf_count += 1;
    }

    // 6. Collect leaves, sort by start position
    let mut leaves: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.is_leaf)
        .map(|(i, _)| i)
        .collect();
    leaves.sort_by_key(|&i| nodes[i].start);

    // 7. Build output
    let mut thresholds: Vec<f64> = Vec::new();
    let mut bin_indices: Vec<Vec<usize>> = Vec::new();
    let mut bin_distributions: Vec<Vec<u32>> = Vec::new();

    for (leaf_pos, &leaf_id) in leaves.iter().enumerate() {
        let node = &nodes[leaf_id];

        // Collect original indices for this bin
        let indices: Vec<usize> = (node.start..node.end)
            .map(|i| samples[i].original_idx)
            .collect();
        bin_indices.push(indices);

        // Class distribution
        bin_distributions.push(node.counts.clone());

        // Threshold between this leaf and the next
        if leaf_pos < leaves.len() - 1 {
            let val_last = samples[node.end - 1].x;
            let val_next = samples[node.end].x;
            thresholds.push((val_last + val_next) / 2.0);
        }
    }

    DiscretizeResult {
        thresholds,
        bin_indices,
        bin_distributions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impurity::Criterion;

    #[test]
    fn single_threshold_separable() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let result = discretize(&x, &y, 2, 2, Criterion::Gini);
        assert_eq!(result.thresholds.len(), 1);
        assert_eq!(result.bin_indices.len(), 2);
        assert_eq!(result.bin_indices[0].len(), 3);
        assert_eq!(result.bin_indices[1].len(), 3);
        assert_eq!(result.bin_distributions[0], vec![3, 0]);
        assert_eq!(result.bin_distributions[1], vec![0, 3]);
    }

    #[test]
    fn max_leaf_nodes_one_returns_single_bin() {
        let x = vec![1.0, 2.0, 3.0];
        let y: Vec<u8> = vec![0, 1, 0];
        let result = discretize(&x, &y, 2, 1, Criterion::Gini);
        assert_eq!(result.thresholds.len(), 0);
        assert_eq!(result.bin_indices.len(), 1);
        assert_eq!(result.bin_indices[0].len(), 3);
    }

    #[test]
    fn pure_node_no_split() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<u8> = vec![0, 0, 0, 0];
        let result = discretize(&x, &y, 1, 4, Criterion::Gini);
        assert_eq!(result.thresholds.len(), 0);
        assert_eq!(result.bin_indices.len(), 1);
    }

    #[test]
    fn multiple_thresholds() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let result = discretize(&x, &y, 3, 4, Criterion::Gini);
        assert!(result.thresholds.len() >= 2);
        assert!(result.bin_indices.len() >= 3);
    }

    #[test]
    fn tie_breaking_deterministic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<u8> = vec![0, 1, 0, 1];
        let r1 = discretize(&x, &y, 2, 3, Criterion::Gini);
        let r2 = discretize(&x, &y, 2, 3, Criterion::Gini);
        assert_eq!(r1.thresholds, r2.thresholds);
    }

    #[test]
    fn single_sample() {
        let x = vec![5.0];
        let y: Vec<u8> = vec![0];
        let result = discretize(&x, &y, 1, 4, Criterion::Gini);
        assert_eq!(result.thresholds.len(), 0);
        assert_eq!(result.bin_indices.len(), 1);
    }

    #[test]
    fn two_samples_different_classes() {
        let x = vec![1.0, 2.0];
        let y: Vec<u8> = vec![0, 1];
        let result = discretize(&x, &y, 2, 2, Criterion::Gini);
        assert_eq!(result.thresholds.len(), 1);
        assert_eq!(result.bin_indices.len(), 2);
    }

    #[test]
    fn entropy_criterion_works() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let result = discretize(&x, &y, 2, 2, Criterion::Entropy);
        assert_eq!(result.thresholds.len(), 1);
        assert_eq!(result.bin_indices.len(), 2);
    }

    #[test]
    fn respects_max_leaf_nodes_cap() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let result = discretize(&x, &y, 3, 2, Criterion::Gini);
        assert_eq!(result.bin_indices.len(), 2);
        assert_eq!(result.thresholds.len(), 1);
    }

    #[test]
    fn identical_feature_values_different_labels() {
        let x = vec![3.0, 3.0, 3.0, 3.0];
        let y: Vec<u8> = vec![0, 1, 0, 1];
        let result = discretize(&x, &y, 2, 4, Criterion::Gini);
        assert_eq!(result.thresholds.len(), 0);
        assert_eq!(result.bin_indices.len(), 1);
    }

    #[test]
    fn bin_indices_are_disjoint_and_cover_all() {
        let x = vec![5.0, 3.0, 1.0, 4.0, 2.0, 6.0];
        let y: Vec<u8> = vec![1, 0, 0, 1, 0, 1];
        let result = discretize(&x, &y, 2, 3, Criterion::Gini);
        let mut all_indices: Vec<usize> = result.bin_indices.iter().flatten().cloned().collect();
        all_indices.sort();
        assert_eq!(all_indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn thresholds_are_sorted() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let result = discretize(&x, &y, 3, 4, Criterion::Gini);
        for i in 1..result.thresholds.len() {
            assert!(result.thresholds[i] > result.thresholds[i - 1]);
        }
    }

    #[test]
    fn distributions_sum_to_bin_sizes() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 1, 1, 0, 1];
        let result = discretize(&x, &y, 2, 3, Criterion::Gini);
        for (i, dist) in result.bin_distributions.iter().enumerate() {
            let sum: u32 = dist.iter().sum();
            assert_eq!(sum as usize, result.bin_indices[i].len());
        }
    }
}
