use super::{DataChildRef, DataStateId, ParentRef};
use ordered_float::OrderedFloat;

/// OR node — represents a subset of samples at a tree node.
/// Chooses between being a leaf or adopting one of its ShapeFunctionState children.
pub struct DataState {
    pub id: DataStateId,
    pub sample_indices: Option<Vec<u32>>, // freed after expansion
    pub depth_budget: u8,
    pub parent: ParentRef,
    pub n_samples_local: u32,
    pub pred_class: i32,
    pub leaf_error: f64,
    pub dual: f64,
    pub primal: f64,
    pub primal_best_child: Option<DataChildRef>,
    pub heuristic_value: f64,
    pub init_heuristic_value: f64,
    pub solved: bool,
    pub expanded: bool,
    // Queues: sorted ascending by first element (best = index 0)
    // dual_queue: (dual_value, heuristic_value, child_ref)
    pub dual_queue: Vec<(OrderedFloat<f64>, OrderedFloat<f64>, DataChildRef)>,
    // heuristic_queue: (heuristic_value, child_ref)
    pub heuristic_queue: Vec<(OrderedFloat<f64>, DataChildRef)>,
    // Subproblem cache fields (not freed on expansion — needed at solve time)
    pub sample_bitset: Vec<u64>,
    pub cached_optimal: Option<f64>,
    pub from_cache: bool,
    /// Exact CART cost matched the dual; its witness is materialized after search.
    pub cart_completed: bool,
}

impl DataState {
    pub fn is_solved(&self) -> bool {
        self.solved
    }

    pub fn is_leaf(&self) -> bool {
        self.primal_best_child.is_none()
    }
}
