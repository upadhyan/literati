use crate::config::Config;
use crate::states::{arena::Arena, DataChildRef, DataStateId, ShapeFnChildRef};
use std::collections::HashMap;

/// A node descriptor for visualization — matches what plot_literati_tree expects.
#[derive(Debug, Clone)]
pub struct NodeDescriptor {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub side: Option<String>, // "left", "right", or None
    pub depth: usize,
    pub is_leaf: bool,
    pub pred_class: i32,
    pub n_samples: u32,
    pub children: Vec<usize>,
    // Internal node fields
    pub feature: Option<String>,
    pub thresholds: Option<Vec<f64>>,
    pub split: Option<Vec<u8>>,
    pub feature_idx: Option<u16>,
}

/// Follow primal_best_child through potentially nested ShapeFunctionStates
/// to find the AndState. Returns (and_state_id, final_sf_id) or None.
fn follow_primal_to_and(
    arena: &Arena,
    sf_id: crate::states::ShapeFnStateId,
) -> Option<(crate::states::AndStateId, crate::states::ShapeFnStateId)> {
    let sf = arena.sf(sf_id);
    match &sf.primal_best_child {
        None => None,
        Some(ShapeFnChildRef::And(and_id)) => Some((*and_id, sf_id)),
        Some(ShapeFnChildRef::ShapeFn(child_sf_id)) => follow_primal_to_and(arena, *child_sf_id),
    }
}

/// Export the primal best tree as a list of node descriptors.
pub fn export_tree(arena: &Arena, config: &Config, root_id: DataStateId) -> Vec<NodeDescriptor> {
    let mut nodes = Vec::new();
    let mut id_counter: usize = 0;
    collect_nodes(
        arena,
        config,
        root_id,
        None,
        None,
        0,
        &mut nodes,
        &mut id_counter,
    );
    nodes
}

#[allow(clippy::only_used_in_recursion)] // config is needed at every recursion level for feature names
#[allow(clippy::too_many_arguments)]
fn collect_nodes(
    arena: &Arena,
    config: &Config,
    ds_id: DataStateId,
    parent_id: Option<usize>,
    side: Option<&str>,
    depth: usize,
    nodes: &mut Vec<NodeDescriptor>,
    id_counter: &mut usize,
) {
    let node_id = *id_counter;
    *id_counter += 1;

    let ds = arena.ds(ds_id);
    let n_samples = ds.n_samples_local;
    let pred_class = ds.pred_class;

    // Check if leaf
    let is_leaf = ds.primal_best_child.is_none();
    let and_info = if !is_leaf {
        match &ds.primal_best_child {
            Some(DataChildRef::ShapeFn(sf_id)) => follow_primal_to_and(arena, *sf_id),
            None => None,
        }
    } else {
        None
    };

    if is_leaf || and_info.is_none() {
        // Leaf node
        nodes.push(NodeDescriptor {
            id: node_id,
            parent_id,
            side: side.map(|s| s.to_string()),
            depth,
            is_leaf: true,
            pred_class,
            n_samples,
            children: vec![],
            feature: None,
            thresholds: None,
            split: None,
            feature_idx: None,
        });
        return;
    }

    let (and_id, final_sf_id) = and_info.unwrap();
    let sf = arena.sf(final_sf_id);
    let a = arena.as_(and_id);

    // Internal node
    nodes.push(NodeDescriptor {
        id: node_id,
        parent_id,
        side: side.map(|s| s.to_string()),
        depth,
        is_leaf: false,
        pred_class,
        n_samples,
        children: vec![], // filled below
        feature: Some(sf.feature_name.clone()),
        thresholds: Some(sf.thresholds.clone()),
        split: Some(sf.split.clone()),
        feature_idx: Some(sf.feature_idx),
    });

    let left_child_start = *id_counter;
    collect_nodes(
        arena,
        config,
        a.left,
        Some(node_id),
        Some("left"),
        depth + 1,
        nodes,
        id_counter,
    );
    let right_child_start = *id_counter;
    collect_nodes(
        arena,
        config,
        a.right,
        Some(node_id),
        Some("right"),
        depth + 1,
        nodes,
        id_counter,
    );

    // Update children list
    nodes[node_id].children = vec![left_child_start, right_child_start];
}

/// Convert tree export to a list of Python-compatible dicts.
/// Returns Vec of HashMaps that can be easily converted to Python dicts.
pub fn export_tree_as_dicts(
    arena: &Arena,
    config: &Config,
    root_id: DataStateId,
) -> Vec<HashMap<String, ExportValue>> {
    let descriptors = export_tree(arena, config, root_id);
    let mut result = Vec::new();

    for desc in &descriptors {
        let mut d: HashMap<String, ExportValue> = HashMap::new();
        d.insert("id".to_string(), ExportValue::Int(desc.id as i64));
        d.insert(
            "parent_id".to_string(),
            match desc.parent_id {
                Some(pid) => ExportValue::Int(pid as i64),
                None => ExportValue::None,
            },
        );
        d.insert(
            "side".to_string(),
            match &desc.side {
                Some(s) => ExportValue::Str(s.clone()),
                None => ExportValue::None,
            },
        );
        d.insert("depth".to_string(), ExportValue::Int(desc.depth as i64));
        d.insert("is_leaf".to_string(), ExportValue::Bool(desc.is_leaf));
        d.insert(
            "pred_class".to_string(),
            ExportValue::Int(desc.pred_class as i64),
        );
        d.insert(
            "n_samples".to_string(),
            ExportValue::Int(desc.n_samples as i64),
        );
        d.insert(
            "children".to_string(),
            ExportValue::IntList(desc.children.iter().map(|&c| c as i64).collect()),
        );

        if let Some(ref f) = desc.feature {
            d.insert("feature".to_string(), ExportValue::Str(f.clone()));
        }
        if let Some(ref t) = desc.thresholds {
            d.insert("thresholds".to_string(), ExportValue::FloatList(t.clone()));
        }
        if let Some(ref s) = desc.split {
            d.insert(
                "split".to_string(),
                ExportValue::IntList(s.iter().map(|&v| v as i64).collect()),
            );
        }
        if let Some(fi) = desc.feature_idx {
            d.insert("feature_idx".to_string(), ExportValue::Int(fi as i64));
        }

        result.push(d);
    }

    result
}

/// Value type for export to Python.
#[derive(Debug, Clone)]
pub enum ExportValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    None,
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use crate::engine::run_ao_star;
    use crate::impurity::Criterion;

    fn make_config(x: Vec<f64>, y: Vec<u8>, n_features: u16, n_classes: u8) -> Config {
        let n_samples = y.len() as u32;
        let nf = n_features as usize;
        let mut feature_dict = Vec::new();
        for f in 0..nf {
            feature_dict.push((format!("f{}", f), vec![f]));
        }
        Config {
            x,
            y,
            n_samples,
            bitset_words: (n_samples as usize).div_ceil(64),
            n_features,
            n_classes,
            max_depth: 2,
            lambda_: 0.0,
            alpha_: 0.0,
            max_order: 1,
            n_candidates_list: vec![3; 2],
            criterion: Criterion::Gini,
            discretizer: Discretizer::Univariate,
            eager_expansion: false,
            use_cache: true,
            heuristic: Heuristic::Primal,
            selection_scheme: SelectionScheme::RoundRobin,
            anytime: false,
            time_limit: 60.0,
            max_iter: -1,
            memory_byte_limit: -1,
            verbose: false,
            feature_names: (0..nf).map(|f| format!("f{}", f)).collect(),
            feature_dict,
            cart_heuristic_nanos: std::sync::atomic::AtomicU64::new(0),
        }
    }

    #[test]
    fn export_leaf_tree() {
        let x = vec![1.0, 2.0, 3.0];
        let y: Vec<u8> = vec![0, 0, 0];
        let config = make_config(x, y, 1, 1);
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        let nodes = export_tree(&result.arena, &config, result.root_id);
        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_leaf);
    }

    #[test]
    fn export_split_tree() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let mut config = make_config(x, y, 1, 2);
        config.max_depth = 1;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        let nodes = export_tree(&result.arena, &config, result.root_id);
        assert_eq!(nodes.len(), 3); // root + 2 leaves
        assert!(!nodes[0].is_leaf);
        assert!(nodes[1].is_leaf);
        assert!(nodes[2].is_leaf);
    }
}
