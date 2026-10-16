use super::{FlatBins, ParentRef, ShapeFnChildRef, ShapeFnStateId};
use ordered_float::OrderedFloat;

/// OR node — represents a candidate shape function split on a specific feature.
/// Chooses between applying this split (AndState child) or refining to higher order.
pub struct ShapeFunctionState {
    pub id: ShapeFnStateId,
    pub sample_indices: Option<Vec<u32>>,
    pub depth_budget: u8,
    pub parent: ParentRef,
    pub feature_idx: u16,
    pub feature_name: String,
    pub thresholds: Vec<f64>,
    pub bin_sample_indices: Option<FlatBins>,
    pub split: Vec<u8>,         // 0 or 1 per bin
    pub k_set: Option<Vec<u8>>, // flip point indices (max ~5)
    pub order_budget: u8,
    pub left_local_indices: Option<Vec<u32>>,
    pub right_local_indices: Option<Vec<u32>>,
    pub success: bool,
    pub dual: f64,
    pub primal: f64,
    pub primal_best_child: Option<ShapeFnChildRef>,
    pub heuristic_value: f64,
    pub solved: bool,
    pub expanded: bool,
    // dual_queue: (dual_value, heuristic_value, child_ref)
    pub dual_queue: Vec<(OrderedFloat<f64>, OrderedFloat<f64>, ShapeFnChildRef)>,
    // heuristic_queue: (heuristic_value, child_ref)
    pub heuristic_queue: Vec<(OrderedFloat<f64>, ShapeFnChildRef)>,
}

impl ShapeFunctionState {
    pub fn is_solved(&self) -> bool {
        self.solved
    }
}
