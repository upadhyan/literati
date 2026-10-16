use crate::impurity::{impurity as compute_impurity, Criterion};
use std::cmp::Ordering;
use std::ops::Range;

const MIN_GAIN: f64 = 1e-15;

pub(crate) struct BestSplit {
    pub feature_idx: usize,
    pub threshold: f64,
    pub gain: f64,
    pub left_impurity: f64,
    pub right_impurity: f64,
}

pub(crate) struct PresortedRows {
    feature_orders: Vec<Vec<usize>>,
    local_order: Vec<usize>,
    scratch: Vec<usize>,
}

impl PresortedRows {
    pub fn new(x: &[f64], n_samples: usize, n_features: usize) -> Option<Self> {
        if x.iter().any(|value| value.is_nan()) {
            return None;
        }

        let mut feature_orders = Vec::with_capacity(n_features);
        for feature_idx in 0..n_features {
            let mut order: Vec<usize> = (0..n_samples).collect();
            order.sort_by(|&a, &b| {
                x[a * n_features + feature_idx]
                    .partial_cmp(&x[b * n_features + feature_idx])
                    .unwrap_or(Ordering::Equal)
            });
            feature_orders.push(order);
        }

        Some(Self {
            feature_orders,
            local_order: (0..n_samples).collect(),
            scratch: Vec::with_capacity(n_samples),
        })
    }

    pub fn local_order(&self, range: Range<usize>) -> &[usize] {
        &self.local_order[range]
    }

    pub fn best_split(
        &self,
        x: &[f64],
        y: &[u8],
        range: Range<usize>,
        n_features: usize,
        n_classes: usize,
        criterion: Criterion,
        parent_impurity: f64,
    ) -> Option<BestSplit> {
        let n_samples = range.len();
        if n_samples < 2 {
            return None;
        }

        let mut total_counts = vec![0u32; n_classes];
        for &idx in &self.local_order[range.clone()] {
            total_counts[y[idx] as usize] += 1;
        }

        let mut best: Option<BestSplit> = None;
        let mut left_counts = vec![0u32; n_classes];
        for (feature_idx, order) in self.feature_orders.iter().enumerate() {
            left_counts.fill(0);
            let mut right_counts = total_counts.clone();
            let mut n_left = 0u32;
            let mut n_right = n_samples as u32;
            let rows = &order[range.clone()];

            for pair_idx in 0..(n_samples - 1) {
                let idx = rows[pair_idx];
                let class = y[idx] as usize;
                left_counts[class] += 1;
                right_counts[class] -= 1;
                n_left += 1;
                n_right -= 1;

                let value = x[idx * n_features + feature_idx];
                let next_value = x[rows[pair_idx + 1] * n_features + feature_idx];
                if value == next_value {
                    continue;
                }

                let left_impurity = compute_impurity(criterion, &left_counts, n_left);
                let right_impurity = compute_impurity(criterion, &right_counts, n_right);
                let weighted_impurity = (n_left as f64 / n_samples as f64) * left_impurity
                    + (n_right as f64 / n_samples as f64) * right_impurity;
                let gain = parent_impurity - weighted_impurity;
                if gain > best.as_ref().map_or(0.0, |split| split.gain) && gain > MIN_GAIN {
                    best = Some(BestSplit {
                        feature_idx,
                        threshold: (value + next_value) / 2.0,
                        gain,
                        left_impurity,
                        right_impurity,
                    });
                }
            }
        }
        best
    }

    pub fn partition(
        &mut self,
        x: &[f64],
        range: Range<usize>,
        n_features: usize,
        feature_idx: usize,
        threshold: f64,
    ) -> usize {
        let start = range.start;
        let predicate = |idx: usize| x[idx * n_features + feature_idx] <= threshold;
        let left_len = Self::stable_partition(
            &mut self.local_order[range.clone()],
            &mut self.scratch,
            predicate,
        );
        for order in &mut self.feature_orders {
            Self::stable_partition(&mut order[range.clone()], &mut self.scratch, predicate);
        }
        start + left_len
    }

    fn stable_partition(
        rows: &mut [usize],
        scratch: &mut Vec<usize>,
        predicate: impl Fn(usize) -> bool,
    ) -> usize {
        scratch.clear();
        scratch.extend(rows.iter().copied().filter(|&idx| predicate(idx)));
        let left_len = scratch.len();
        scratch.extend(rows.iter().copied().filter(|&idx| !predicate(idx)));
        rows.copy_from_slice(scratch);
        left_len
    }
}
