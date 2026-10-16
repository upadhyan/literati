use crate::config::Config;
use crate::states::{arena::Arena, DataChildRef, DataStateId, ShapeFnChildRef};

/// Predict class for a single sample by traversing the primal best tree.
fn predict_single(arena: &Arena, config: &Config, x_row: &[f64], ds_id: DataStateId) -> i32 {
    let ds = arena.ds(ds_id);

    // If leaf (no best child), return predicted class
    let child_ref = match &ds.primal_best_child {
        None => return ds.pred_class,
        Some(c) => *c,
    };

    // Follow primal_best_child chain
    match child_ref {
        DataChildRef::ShapeFn(sf_id) => predict_from_sf(arena, config, x_row, sf_id),
    }
}

/// Follow a ShapeFunctionState chain to find the AndState, then route left/right.
fn predict_from_sf(
    arena: &Arena,
    config: &Config,
    x_row: &[f64],
    sf_id: crate::states::ShapeFnStateId,
) -> i32 {
    let sf = arena.sf(sf_id);

    let best_child = match &sf.primal_best_child {
        None => {
            // No best child — go up to find a DataState
            // This shouldn't normally happen in a well-formed tree
            return sf.split[0] as i32; // fallback
        }
        Some(c) => *c,
    };

    match best_child {
        ShapeFnChildRef::And(and_id) => {
            // Determine bin for sample
            let feature_idx = sf.feature_idx as usize;
            let x_val = x_row[feature_idx];
            let thresholds = &sf.thresholds;
            let split = &sf.split;

            // Find bin
            let bin = get_sample_bin(x_val, thresholds);
            let bin = bin.min(split.len() - 1);

            let a = arena.as_(and_id);
            if split[bin] == 1 {
                predict_single(arena, config, x_row, a.left)
            } else {
                predict_single(arena, config, x_row, a.right)
            }
        }
        ShapeFnChildRef::ShapeFn(child_sf_id) => {
            // Higher-order refinement — recurse
            predict_from_sf(arena, config, x_row, child_sf_id)
        }
    }
}

/// Determine which bin a feature value falls into.
fn get_sample_bin(x_val: f64, thresholds: &[f64]) -> usize {
    if thresholds.is_empty() {
        return 0;
    }
    if x_val <= thresholds[0] {
        return 0;
    }
    for i in 1..thresholds.len() {
        if thresholds[i - 1] < x_val && x_val <= thresholds[i] {
            return i;
        }
    }
    thresholds.len()
}

/// Batch predict for all samples in x (flattened row-major).
pub fn batch_predict(
    arena: &Arena,
    config: &Config,
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    root_id: DataStateId,
) -> Vec<i32> {
    let mut predictions = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let row = &x[i * n_features..(i + 1) * n_features];
        predictions.push(predict_single(arena, config, row, root_id));
    }
    predictions
}
