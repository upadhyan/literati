use super::{AndStateId, DataStateId, ShapeFnStateId};

/// AND node — represents the obligation to solve both left and right subtrees.
pub struct AndState {
    pub id: AndStateId,
    pub left: DataStateId,
    pub right: DataStateId,
    pub parent: ShapeFnStateId,
    pub choose_left: bool, // for round_robin selection alternation
}
