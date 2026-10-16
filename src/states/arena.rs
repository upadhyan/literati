use super::{
    and_state::AndState, data_state::DataState, shape_fn_state::ShapeFunctionState, AndStateId,
    DataStateId, FlatBins, ParentRef, ShapeFnStateId,
};
use slab::Slab;

/// Arena allocator for the AO* search graph.
/// All states are stored in Slabs, referenced by typed indices.
/// Slab enables O(1) insertion and removal with stable keys.
pub struct Arena {
    pub data_states: Slab<DataState>,
    pub shape_fn_states: Slab<ShapeFunctionState>,
    pub and_states: Slab<AndState>,
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl Arena {
    pub fn new() -> Self {
        Arena {
            data_states: Slab::new(),
            shape_fn_states: Slab::new(),
            and_states: Slab::new(),
        }
    }

    /// Allocate a new DataState with essential fields.
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_data_state(
        &mut self,
        sample_indices: Vec<u32>,
        depth_budget: u8,
        parent: ParentRef,
        n_samples_local: u32,
        pred_class: i32,
        leaf_error: f64,
        dual: f64,
        primal: f64,
        heuristic_value: f64,
        sample_bitset: Vec<u64>,
    ) -> DataStateId {
        let entry = self.data_states.vacant_entry();
        let key = entry.key();
        let id = DataStateId(key as u32);
        entry.insert(DataState {
            id,
            sample_indices: Some(sample_indices),
            depth_budget,
            parent,
            n_samples_local,
            pred_class,
            leaf_error,
            dual,
            primal,
            primal_best_child: None,
            heuristic_value,
            init_heuristic_value: heuristic_value,
            solved: false,
            expanded: false,
            dual_queue: Vec::new(),
            heuristic_queue: Vec::new(),
            sample_bitset,
            cached_optimal: None,
            from_cache: false,
            cart_completed: false,
        });
        id
    }

    /// Allocate a new ShapeFunctionState.
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_shape_fn_state(
        &mut self,
        sample_indices: Vec<u32>,
        depth_budget: u8,
        parent: ParentRef,
        feature_idx: u16,
        feature_name: String,
        thresholds: Vec<f64>,
        bin_sample_indices: FlatBins,
        split: Vec<u8>,
        k_set: Vec<u8>,
        order_budget: u8,
        left_local_indices: Vec<u32>,
        right_local_indices: Vec<u32>,
        dual: f64,
        primal: f64,
        heuristic_value: f64,
    ) -> ShapeFnStateId {
        let entry = self.shape_fn_states.vacant_entry();
        let key = entry.key();
        let id = ShapeFnStateId(key as u32);
        entry.insert(ShapeFunctionState {
            id,
            sample_indices: Some(sample_indices),
            depth_budget,
            parent,
            feature_idx,
            feature_name,
            thresholds,
            bin_sample_indices: Some(bin_sample_indices),
            split,
            k_set: Some(k_set),
            order_budget,
            left_local_indices: Some(left_local_indices),
            right_local_indices: Some(right_local_indices),
            success: true,
            dual,
            primal,
            primal_best_child: None,
            heuristic_value,
            solved: false,
            expanded: false,
            dual_queue: Vec::new(),
            heuristic_queue: Vec::new(),
        });
        id
    }

    /// Allocate a new AndState.
    pub fn alloc_and_state(
        &mut self,
        left: DataStateId,
        right: DataStateId,
        parent: ShapeFnStateId,
    ) -> AndStateId {
        let entry = self.and_states.vacant_entry();
        let key = entry.key();
        let id = AndStateId(key as u32);
        entry.insert(AndState {
            id,
            left,
            right,
            parent,
            choose_left: true,
        });
        id
    }

    // ── Accessors ────────────────────────────────────────────────────────

    pub fn ds(&self, id: DataStateId) -> &DataState {
        &self.data_states[id.0 as usize]
    }

    pub fn ds_mut(&mut self, id: DataStateId) -> &mut DataState {
        &mut self.data_states[id.0 as usize]
    }

    pub fn sf(&self, id: ShapeFnStateId) -> &ShapeFunctionState {
        &self.shape_fn_states[id.0 as usize]
    }

    pub fn sf_mut(&mut self, id: ShapeFnStateId) -> &mut ShapeFunctionState {
        &mut self.shape_fn_states[id.0 as usize]
    }

    pub fn as_(&self, id: AndStateId) -> &AndState {
        &self.and_states[id.0 as usize]
    }

    pub fn as_mut(&mut self, id: AndStateId) -> &mut AndState {
        &mut self.and_states[id.0 as usize]
    }

    // ── Computed properties ──────────────────────────────────────────────

    /// Compute AndState dual as sum of left.dual + right.dual
    pub fn and_dual(&self, id: AndStateId) -> f64 {
        let a = self.as_(id);
        debug_assert!(
            self.data_states.contains(a.left.0 as usize),
            "and_dual: left child {} already removed",
            a.left.0
        );
        debug_assert!(
            self.data_states.contains(a.right.0 as usize),
            "and_dual: right child {} already removed",
            a.right.0
        );
        self.ds(a.left).dual + self.ds(a.right).dual
    }

    /// Compute AndState primal as sum of left.primal + right.primal
    pub fn and_primal(&self, id: AndStateId) -> f64 {
        let a = self.as_(id);
        debug_assert!(
            self.data_states.contains(a.left.0 as usize),
            "and_primal: left child {} already removed",
            a.left.0
        );
        debug_assert!(
            self.data_states.contains(a.right.0 as usize),
            "and_primal: right child {} already removed",
            a.right.0
        );
        self.ds(a.left).primal + self.ds(a.right).primal
    }

    /// Check if AndState is solved (both children solved)
    pub fn and_is_solved(&self, id: AndStateId) -> bool {
        let a = self.as_(id);
        debug_assert!(
            self.data_states.contains(a.left.0 as usize),
            "and_is_solved: left child {} already removed",
            a.left.0
        );
        debug_assert!(
            self.data_states.contains(a.right.0 as usize),
            "and_is_solved: right child {} already removed",
            a.right.0
        );
        self.ds(a.left).is_solved() && self.ds(a.right).is_solved()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_retrieve_data_state() {
        let mut arena = Arena::new();
        let id = arena.alloc_data_state(
            vec![0u32, 1, 2],
            3,
            ParentRef::None,
            3,
            0,
            1.0,
            1.0,
            1.0,
            1.0,
            Vec::new(),
        );
        assert_eq!(id, DataStateId(0));
        assert_eq!(arena.ds(id).n_samples_local, 3);
        assert_eq!(arena.ds(id).pred_class, 0);
        assert!(!arena.ds(id).expanded);
    }

    #[test]
    fn alloc_and_retrieve_shape_fn_state() {
        let mut arena = Arena::new();
        let id = arena.alloc_shape_fn_state(
            vec![0u32, 1],
            2,
            ParentRef::DataState(DataStateId(0)),
            0,
            "feat_0".to_string(),
            vec![2.5],
            FlatBins::from_bins(&[vec![0u32], vec![1u32]]),
            vec![1, 0],
            vec![0u8],
            3,
            vec![0u32],
            vec![1u32],
            0.0,
            1.0,
            0.5,
        );
        assert_eq!(id, ShapeFnStateId(0));
        assert_eq!(arena.sf(id).feature_name, "feat_0");
        assert_eq!(arena.sf(id).split, vec![1, 0]);
    }

    #[test]
    fn alloc_and_retrieve_and_state() {
        let mut arena = Arena::new();
        let left = arena.alloc_data_state(
            vec![0u32],
            2,
            ParentRef::None,
            1,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            Vec::new(),
        );
        let right = arena.alloc_data_state(
            vec![1u32],
            2,
            ParentRef::None,
            1,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            Vec::new(),
        );
        let and_id = arena.alloc_and_state(left, right, ShapeFnStateId(0));
        assert_eq!(and_id, AndStateId(0));
        assert_eq!(arena.as_(and_id).left, left);
        assert_eq!(arena.as_(and_id).right, right);
        assert!(arena.as_(and_id).choose_left);
    }

    #[test]
    fn multiple_allocs_get_unique_ids() {
        let mut arena = Arena::new();
        let id1 = arena.alloc_data_state(
            vec![0u32],
            1,
            ParentRef::None,
            1,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            Vec::new(),
        );
        let id2 = arena.alloc_data_state(
            vec![1u32],
            1,
            ParentRef::None,
            1,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            Vec::new(),
        );
        assert_ne!(id1, id2);
        assert_eq!(id1, DataStateId(0));
        assert_eq!(id2, DataStateId(1));
    }

    #[test]
    fn and_dual_primal_computed() {
        let mut arena = Arena::new();
        let left = arena.alloc_data_state(
            vec![0u32],
            2,
            ParentRef::None,
            1,
            0,
            3.0,
            2.0,
            3.0,
            3.0,
            Vec::new(),
        );
        let right = arena.alloc_data_state(
            vec![1u32],
            2,
            ParentRef::None,
            1,
            1,
            5.0,
            4.0,
            5.0,
            5.0,
            Vec::new(),
        );
        let and_id = arena.alloc_and_state(left, right, ShapeFnStateId(0));
        assert!((arena.and_dual(and_id) - 6.0).abs() < 1e-12);
        assert!((arena.and_primal(and_id) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn and_is_solved_logic() {
        let mut arena = Arena::new();
        let left = arena.alloc_data_state(
            vec![0u32],
            2,
            ParentRef::None,
            1,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            Vec::new(),
        );
        let right = arena.alloc_data_state(
            vec![1u32],
            2,
            ParentRef::None,
            1,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            Vec::new(),
        );
        let and_id = arena.alloc_and_state(left, right, ShapeFnStateId(0));

        assert!(!arena.and_is_solved(and_id));

        arena.ds_mut(left).solved = true;
        assert!(!arena.and_is_solved(and_id));

        arena.ds_mut(right).solved = true;
        assert!(arena.and_is_solved(and_id));
    }

    #[test]
    fn ds_mut_modifies_state() {
        let mut arena = Arena::new();
        let id = arena.alloc_data_state(
            vec![0u32, 1],
            3,
            ParentRef::None,
            2,
            0,
            1.0,
            1.0,
            1.0,
            1.0,
            Vec::new(),
        );
        arena.ds_mut(id).dual = 0.5;
        assert!((arena.ds(id).dual - 0.5).abs() < 1e-12);
    }
}
