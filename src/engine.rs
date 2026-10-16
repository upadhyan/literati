use ordered_float::OrderedFloat;
use rand::Rng;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::config::{
    CartCacheKey, Config, Discretizer, DiscretizerCacheKey, Heuristic, SearchState, SelectionScheme,
};

use crate::discretize::multivariate::fit_multivariate_cart;
use crate::discretize::univariate::discretize;
use crate::heuristic::cart_heuristic::{cart_heuristic, cart_heuristic_with_subtrees, CartWitness};
use crate::impurity::{impurity as compute_impurity, Criterion};
use crate::states::{
    arena::Arena, AndStateId, DataChildRef, DataStateId, FlatBins, ParentRef, ShapeFnChildRef,
    ShapeFnStateId,
};

use rayon::prelude::*;

/// Minimum number of samples at a node for parallel expansion to be worthwhile.
/// Below this threshold, Rayon scheduling overhead exceeds parallelism benefit.
const PARALLEL_EXPANSION_THRESHOLD: usize = 2500;

// ─────────────────────────────────────────────────────────────────────────────
// Selection target
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum SelectionTarget {
    DataState(DataStateId),
    ShapeFnState(ShapeFnStateId),
}

// ─────────────────────────────────────────────────────────────────────────────
// Search result
// ─────────────────────────────────────────────────────────────────────────────

/// How often (in iterations) to check process RSS against the memory limit.
const MEMORY_CHECK_INTERVAL: usize = 1000;

/// Read the current process RSS in bytes from /proc/self/status (Linux only).
/// Returns None if the file cannot be read or parsed (non-Linux, permissions, etc.).
fn read_rss_bytes() -> Option<i64> {
    let contents = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in contents.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: i64 = parts[1].parse().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

/// A single entry in the optimization history, recorded whenever dual or primal improves.
#[derive(Clone, Debug)]
pub struct TrackEntry {
    pub iteration: usize,
    pub dual: f64,
    pub primal: f64,
    pub gap: f64,
    pub time: f64,
}

pub struct SearchResult {
    pub root_id: DataStateId,
    pub arena: Arena,
    pub iterations_run: usize,
    pub track: Vec<TrackEntry>,
    pub terminated_by_memory: bool,
    pub memory_guard_active: bool,
    /// Total wall-clock runtime of the AO* search loop, in seconds.
    pub total_time: f64,
    /// Cumulative time spent computing the CART heuristic, in seconds.
    /// Aggregate across threads (see `Config::cart_heuristic_nanos`); can
    /// exceed `total_time` when expansion runs in parallel.
    pub cart_heuristic_time: f64,
    /// Number of subproblem cache hits during the search.
    pub cache_hits: u64,
    /// Number of subproblem cache stores during the search.
    pub cache_stores: u64,
    /// Number of partial (unsolved) cache hits that tightened bounds.
    pub cache_partial_hits: u64,
    pub cart_cache_hits: u64,
    pub cart_cache_misses: u64,
    pub cart_cache_stores: u64,
    pub discretizer_cache_hits: u64,
    pub discretizer_cache_misses: u64,
    pub discretizer_cache_stores: u64,
    pub exact_cache_cap_rejections: u64,
    pub exact_cache_bytes: usize,
    pub discretizer_time: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Prepared state structs for parallel expansion
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed DataState fields. Parent ref assigned during sequential allocation.
struct PreparedDataState {
    sample_indices: Vec<u32>,
    depth_budget: u8,
    n_samples_local: u32,
    pred_class: i32,
    leaf_error: f64,
    dual: f64,
    primal: f64,
    heuristic_value: f64,
    sample_bitset: Vec<u64>,
    cart_completed: bool,
}

/// Precomputed eager expansion data (two child DataStates).
struct PreparedEagerData {
    left: PreparedDataState,
    right: PreparedDataState,
}

/// Complete precomputed data for one ShapeFnState child + its eager expansion.
struct PreparedShapeFnChild {
    sample_indices: Vec<u32>,
    depth_budget: u8,
    feature_idx: u16,
    feature_name: String,
    thresholds: Vec<f64>,
    flat_bins: FlatBins,
    split: Vec<u8>,
    k_set: Vec<u8>,
    order_budget: u8,
    left_local_indices: Vec<u32>,
    right_local_indices: Vec<u32>,
    sf_dual: f64,
    sf_primal: f64,
    sf_heuristic: f64,
    eager_data: Option<PreparedEagerData>,
}

// ─────────────────────────────────────────────────────────────────────────────
// DataState initialization
// ─────────────────────────────────────────────────────────────────────────────

fn sample_bitset(config: &Config, sample_indices: &[u32]) -> Vec<u64> {
    if !config.use_cache {
        return Vec::new();
    }
    let mut bitset = vec![0u64; config.bitset_words];
    for &idx in sample_indices {
        bitset[(idx / 64) as usize] |= 1u64 << (idx % 64);
    }
    bitset
}

/// Apply cache lookup to a freshly allocated DataState and set a true leaf.
fn apply_cache_and_leaf(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    id: DataStateId,
) -> bool {
    let ds = arena.ds(id);
    let depth_budget = ds.depth_budget;
    let leaf_error = ds.leaf_error;
    if config.use_cache {
        let entry = search_state
            .cache
            .get(ds.n_samples_local as usize, &ds.sample_bitset, depth_budget)
            .cloned();
        if let Some(entry) = entry {
            search_state.cache.hits += 1;

            if entry.solved {
                // ── Solved cache hit (existing behavior) ──
                let optimal_cost = entry.dual; // solved entry: dual == primal == optimal_cost
                let heuristic = entry.heuristic;

                let ds = arena.ds_mut(id);
                ds.cached_optimal = Some(optimal_cost);
                ds.from_cache = true;
                ds.dual = optimal_cost;
                ds.primal = ds.primal.min(optimal_cost);
                ds.heuristic_value = heuristic;

                // Cache-hit node with optimal_cost < leaf_error has dual == primal
                // (both set to optimal_cost), but is solved via SPLITTING, not as a leaf.
                // set_data_state_leaf would clobber the cached bounds with leaf_error.
                // So only set leaf when depth_budget == 0 (cache irrelevant there).
                if depth_budget == 0 {
                    set_data_state_leaf(arena, id);
                }
            } else {
                // ── Unsolved (partial) cache hit ──
                // Tighten bounds but do NOT set from_cache or cached_optimal.
                // Do NOT set solved — only backprop/set_data_state_leaf may do that.
                search_state.cache.partial_hits += 1;

                let ds = arena.ds_mut(id);
                ds.dual = ds.dual.max(entry.dual);
                ds.dual = ds.dual.min(ds.primal); // clamp: dual <= primal
                ds.heuristic_value = ds.heuristic_value.min(entry.heuristic);

                // Re-check leaf condition after tightening
                let is_leaf = depth_budget == 0 || ds.dual >= leaf_error - 1e-12;
                if is_leaf {
                    set_data_state_leaf(arena, id);
                }
            }
        } else {
            // No cache hit: check if should immediately be a leaf.
            let ds = arena.ds(id);
            let is_leaf = depth_budget == 0 || ds.dual == ds.primal;
            if is_leaf {
                set_data_state_leaf(arena, id);
            }
        }
    } else {
        // Cache disabled: check if should immediately be a leaf.
        let ds = arena.ds(id);
        let is_leaf = depth_budget == 0 || ds.dual == ds.primal;
        if is_leaf {
            set_data_state_leaf(arena, id);
        }
    }
    false
}

/// Create a DataState with proper initialization (class counts, bounds, heuristic).
/// Looks up the subproblem cache and injects `cached_optimal`/`dual`/`heuristic`
/// on hit.
pub fn create_data_state(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    sample_indices: Vec<u32>,
    depth_budget: u8,
    parent: ParentRef,
) -> DataStateId {
    let n_samples = sample_indices.len();

    // Compute class counts
    let mut class_counts = vec![0u32; config.n_classes as usize];
    for &idx in &sample_indices {
        class_counts[config.y[idx as usize] as usize] += 1;
    }

    // Find pred_class and leaf_error
    let mut max_count: u32 = 0;
    let mut pred_class: i32 = 0;
    for (c, &count) in class_counts.iter().enumerate() {
        if count > max_count {
            max_count = count;
            pred_class = c as i32;
        }
    }
    let leaf_error = n_samples as f64 - max_count as f64;

    // Compute dual and primal
    let dual = leaf_error.min(config.lambda_);
    let primal = leaf_error;

    // Cache-off returns an empty, non-allocating identity.
    let sample_bitset = sample_bitset(config, &sample_indices);

    let heuristic_value = compute_data_state_heuristic(
        config,
        Some(search_state),
        &sample_indices,
        &class_counts,
        n_samples,
        depth_budget,
        dual,
        primal,
        leaf_error,
    );
    let cart_completed = config.memory_byte_limit <= 0
        && config.heuristic == Heuristic::Cart
        && heuristic_value == dual
        && dual < leaf_error;

    let id = arena.alloc_data_state(
        sample_indices,
        depth_budget,
        parent,
        n_samples as u32,
        pred_class,
        leaf_error,
        dual,
        primal,
        heuristic_value,
        sample_bitset,
    );

    apply_cache_and_leaf(arena, config, search_state, id);
    if cart_completed {
        mark_data_state_cart_completed(arena, id);
    }

    id
}

#[allow(clippy::too_many_arguments)]
fn cached_cart_heuristic(
    config: &Config,
    search_state: &SearchState,
    sample_indices: &[u32],
    depth: u8,
) -> f64 {
    if config.use_cache {
        let key = CartCacheKey {
            sample_indices: sample_indices.to_vec(),
            depth,
            criterion: config.criterion,
            lambda_bits: config.lambda_.to_bits(),
            n_features: config.n_features,
        };
        if let Some(value) = {
            let mut cache = search_state.cart_cache.lock().unwrap();
            if let Some(&value) = cache.map.get(&key) {
                cache.hits += 1;
                Some(value)
            } else {
                cache.misses += 1;
                None
            }
        } {
            return value;
        }

        let (value, completed, _) =
            compute_cart_heuristic_with_subtrees(config, sample_indices, depth);
        let mut cache = search_state.cart_cache.lock().unwrap();
        if let Some(&existing) = cache.map.get(&key) {
            cache.hits += 1;
            existing
        } else {
            for (completed_key, completed_value) in completed.into_iter().rev() {
                if cache.map.contains_key(&completed_key) {
                    continue;
                }
                if !search_state.reserve_exact_cache_entry(
                    SearchState::cart_cache_entry_bytes(&completed_key),
                    crate::config::EXACT_CACHE_ENTRY_CAP,
                    crate::config::EXACT_CACHE_BYTE_BUDGET,
                ) {
                    continue;
                }
                cache.map.insert(completed_key, completed_value);
                cache.stores += 1;
            }
            value
        }
    } else {
        // Cache-off deliberately avoids key allocation, locking, and insertion.
        compute_cart_heuristic(config, sample_indices, depth)
    }
}

fn compute_cart_heuristic(config: &Config, sample_indices: &[u32], depth: u8) -> f64 {
    let nf = config.n_features as usize;
    let t0 = Instant::now();
    let y_local: Vec<u8> = sample_indices
        .iter()
        .map(|&idx| config.y[idx as usize])
        .collect();
    let mut x_local = Vec::with_capacity(sample_indices.len() * nf);
    for &idx in sample_indices {
        let idx = idx as usize;
        x_local.extend_from_slice(&config.x[idx * nf..(idx + 1) * nf]);
    }
    let out = cart_heuristic(
        &x_local,
        &y_local,
        nf,
        config.lambda_,
        depth as usize,
        config.criterion,
    );
    config
        .cart_heuristic_nanos
        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

fn compute_cart_heuristic_with_subtrees(
    config: &Config,
    sample_indices: &[u32],
    depth: u8,
) -> (f64, Vec<(CartCacheKey, f64)>, Option<CartWitness>) {
    let nf = config.n_features as usize;
    let t0 = Instant::now();
    let y_local: Vec<u8> = sample_indices
        .iter()
        .map(|&idx| config.y[idx as usize])
        .collect();
    let mut x_local = Vec::with_capacity(sample_indices.len() * nf);
    for &idx in sample_indices {
        let idx = idx as usize;
        x_local.extend_from_slice(&config.x[idx * nf..(idx + 1) * nf]);
    }
    let (value, completed, witness) = cart_heuristic_with_subtrees(
        &x_local,
        &y_local,
        nf,
        config.lambda_,
        depth as usize,
        config.criterion,
    );
    let completed = completed
        .into_iter()
        .map(|subtree| {
            (
                CartCacheKey {
                    sample_indices: subtree
                        .local_indices
                        .into_iter()
                        .map(|idx| sample_indices[idx])
                        .collect(),
                    depth: subtree.remaining_depth as u8,
                    criterion: config.criterion,
                    lambda_bits: config.lambda_.to_bits(),
                    n_features: config.n_features,
                },
                subtree.cost,
            )
        })
        .collect();
    config
        .cart_heuristic_nanos
        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    (value, completed, witness)
}

fn compute_data_state_heuristic(
    config: &Config,
    search_state: Option<&SearchState>,
    sample_indices: &[u32],
    class_counts: &[u32],
    n_samples: usize,
    depth_budget: u8,
    dual: f64,
    primal: f64,
    leaf_error: f64,
) -> f64 {
    match config.heuristic {
        Heuristic::Dual => dual,
        Heuristic::Primal => primal,
        Heuristic::Gini => {
            n_samples as f64 * compute_impurity(Criterion::Gini, class_counts, n_samples as u32)
        }
        Heuristic::Entropy => {
            n_samples as f64 * compute_impurity(Criterion::Entropy, class_counts, n_samples as u32)
        }
        Heuristic::Cart => {
            if depth_budget == 0 || n_samples <= 1 {
                return leaf_error;
            }
            match search_state {
                Some(search_state) => {
                    cached_cart_heuristic(config, search_state, sample_indices, depth_budget)
                }
                None => compute_cart_heuristic(config, sample_indices, depth_budget),
            }
        }
    }
}

/// Compute all DataState fields without arena access.
/// Used by the parallel expansion phase to precompute child DataState data.
fn prepare_data_state(
    config: &Config,
    search_state: Option<&SearchState>,
    sample_indices: &[u32],
    class_counts: Option<&[u32]>,
    depth_budget: u8,
) -> PreparedDataState {
    let n_samples = sample_indices.len();

    let computed_counts;
    let class_counts = match class_counts {
        Some(counts) => counts,
        None => {
            computed_counts = {
                let mut counts = vec![0u32; config.n_classes as usize];
                for &idx in sample_indices {
                    counts[config.y[idx as usize] as usize] += 1;
                }
                counts
            };
            &computed_counts
        }
    };

    let mut max_count: u32 = 0;
    let mut pred_class: i32 = 0;
    for (c, &count) in class_counts.iter().enumerate() {
        if count > max_count {
            max_count = count;
            pred_class = c as i32;
        }
    }
    let leaf_error = n_samples as f64 - max_count as f64;

    let dual = leaf_error.min(config.lambda_);
    let primal = leaf_error;

    let heuristic_value = compute_data_state_heuristic(
        config,
        search_state,
        sample_indices,
        class_counts,
        n_samples,
        depth_budget,
        dual,
        primal,
        leaf_error,
    );
    let cart_completed = config.memory_byte_limit <= 0
        && config.heuristic == Heuristic::Cart
        && heuristic_value == dual
        && dual < leaf_error;

    let sample_bitset = sample_bitset(config, sample_indices);

    PreparedDataState {
        sample_indices: sample_indices.to_vec(),
        depth_budget,
        n_samples_local: n_samples as u32,
        pred_class,
        leaf_error,
        dual,
        primal,
        heuristic_value,
        sample_bitset,
        cart_completed,
    }
}

/// Compute eager expansion data (two child DataStates)
/// without arena access. Used by the parallel expansion phase.
fn prepare_eager_data(
    config: &Config,
    search_state: Option<&SearchState>,
    sample_indices: &[u32],
    left_local_indices: &[u32],
    right_local_indices: &[u32],
    left_class_counts: Option<&[u32]>,
    right_class_counts: Option<&[u32]>,
    depth_budget: u8,
) -> PreparedEagerData {
    let left_sample_indices: Vec<u32> = left_local_indices
        .iter()
        .map(|&li| sample_indices[li as usize])
        .collect();
    let right_sample_indices: Vec<u32> = right_local_indices
        .iter()
        .map(|&ri| sample_indices[ri as usize])
        .collect();

    let child_depth = depth_budget.saturating_sub(1);
    let left = prepare_data_state(
        config,
        search_state,
        &left_sample_indices,
        left_class_counts,
        child_depth,
    );
    let right = prepare_data_state(
        config,
        search_state,
        &right_sample_indices,
        right_class_counts,
        child_depth,
    );

    PreparedEagerData { left, right }
}

/// Compute all ShapeFnState child data without arena access.
/// Returns None if the split is invalid (empty left or right side).
#[allow(clippy::too_many_arguments)]
fn prepare_shape_fn_child(
    config: &Config,
    search_state: Option<&SearchState>,
    sample_indices: &[u32],
    depth_budget: u8,
    feature_idx: u16,
    feature_name: String,
    thresholds: Vec<f64>,
    bin_sample_indices: &[Vec<u32>],
    bin_distributions: Option<&[Vec<u32>]>,
    split: Vec<u8>,
    k_set: Vec<u8>,
    order_budget: u8,
) -> Option<PreparedShapeFnChild> {
    let mut left_local_indices: Vec<u32> = Vec::new();
    let mut right_local_indices: Vec<u32> = Vec::new();
    let mut left_class_counts = vec![0u32; config.n_classes as usize];
    let mut right_class_counts = vec![0u32; config.n_classes as usize];
    for (b, &side) in split.iter().enumerate() {
        let (indices, counts) = if side == 1 {
            (&mut left_local_indices, &mut left_class_counts)
        } else {
            (&mut right_local_indices, &mut right_class_counts)
        };
        indices.extend_from_slice(&bin_sample_indices[b]);
        if let Some(distributions) = bin_distributions {
            for (total, &count) in counts.iter_mut().zip(&distributions[b]) {
                *total += count;
            }
        }
    }

    if left_local_indices.is_empty() || right_local_indices.is_empty() {
        return None;
    }

    let sf_dual = 0.0;
    let sf_primal = sample_indices.len() as f64;

    let (sf_heuristic, eager_data) = if config.eager_expansion {
        let eager = prepare_eager_data(
            config,
            search_state,
            sample_indices,
            &left_local_indices,
            &right_local_indices,
            bin_distributions.map(|_| left_class_counts.as_slice()),
            bin_distributions.map(|_| right_class_counts.as_slice()),
            depth_budget,
        );
        (f64::INFINITY, Some(eager))
    } else {
        let h = compute_sf_heuristic(
            config,
            search_state,
            sample_indices,
            &left_local_indices,
            &right_local_indices,
            depth_budget,
            sf_dual,
            sf_primal,
        );
        (h, None)
    };

    let flat_bins = FlatBins::from_bins(bin_sample_indices);

    Some(PreparedShapeFnChild {
        sample_indices: sample_indices.to_vec(),
        depth_budget,
        feature_idx,
        feature_name,
        thresholds,
        flat_bins,
        split,
        k_set,
        order_budget,
        left_local_indices,
        right_local_indices,
        sf_dual,
        sf_primal,
        sf_heuristic,
        eager_data,
    })
}

/// Allocate a PreparedShapeFnChild into the arena (sequential phase).
/// Creates the SF state and, if eager data is present, also creates
/// left/right DataStates and the AndState, wiring up all references.
fn allocate_prepared_sf_child(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    prepared: PreparedShapeFnChild,
    parent: ParentRef,
) -> Option<ShapeFnStateId> {
    let order_budget = prepared.order_budget;
    let sf_id = arena.alloc_shape_fn_state(
        prepared.sample_indices,
        prepared.depth_budget,
        parent,
        prepared.feature_idx,
        prepared.feature_name,
        prepared.thresholds,
        prepared.flat_bins,
        prepared.split,
        prepared.k_set,
        prepared.order_budget,
        prepared.left_local_indices,
        prepared.right_local_indices,
        prepared.sf_dual,
        prepared.sf_primal,
        prepared.sf_heuristic,
    );

    if let Some(eager) = prepared.eager_data {
        let left_parent = ParentRef::ShapeFnState(sf_id);
        let left_cart_completed = eager.left.cart_completed;
        let left_id = arena.alloc_data_state(
            eager.left.sample_indices,
            eager.left.depth_budget,
            left_parent,
            eager.left.n_samples_local,
            eager.left.pred_class,
            eager.left.leaf_error,
            eager.left.dual,
            eager.left.primal,
            eager.left.heuristic_value,
            eager.left.sample_bitset,
        );
        apply_cache_and_leaf(arena, config, search_state, left_id);
        if left_cart_completed {
            mark_data_state_cart_completed(arena, left_id);
        }

        let right_parent = ParentRef::ShapeFnState(sf_id);
        let right_cart_completed = eager.right.cart_completed;
        let right_id = arena.alloc_data_state(
            eager.right.sample_indices,
            eager.right.depth_budget,
            right_parent,
            eager.right.n_samples_local,
            eager.right.pred_class,
            eager.right.leaf_error,
            eager.right.dual,
            eager.right.primal,
            eager.right.heuristic_value,
            eager.right.sample_bitset,
        );
        apply_cache_and_leaf(arena, config, search_state, right_id);
        if right_cart_completed {
            mark_data_state_cart_completed(arena, right_id);
        }

        let and_id = arena.alloc_and_state(left_id, right_id, sf_id);

        // The AndState now exists, but its parent SF queue is not wired yet.
        // Complete exact-CART children through non-propagating backpropagation;
        // the normal terminal sweep will notify parents after queue wiring.
        if left_cart_completed {
            backprop_data_state(arena, config, search_state, left_id, false);
        }
        if right_cart_completed {
            backprop_data_state(arena, config, search_state, right_id, false);
        }

        // Compute and bounds from actual arena DataStates (cache may have
        // overridden left/right duals/heuristics via apply_cache_and_leaf)
        let actual_and_dual = arena.and_dual(and_id);
        let actual_and_primal = arena.and_primal(and_id);
        let actual_and_heuristic =
            arena.ds(left_id).heuristic_value + arena.ds(right_id).heuristic_value;

        let child_ref = ShapeFnChildRef::And(and_id);
        let sf = arena.sf_mut(sf_id);
        insort_dual_queue_sf(
            &mut sf.dual_queue,
            OrderedFloat(actual_and_dual),
            OrderedFloat(actual_and_heuristic),
            child_ref,
        );
        insort_heuristic_queue_sf(
            &mut sf.heuristic_queue,
            OrderedFloat(actual_and_heuristic),
            child_ref,
        );

        // Update SF bounds from actual (possibly cache-adjusted) data
        let sf = arena.sf_mut(sf_id);
        sf.dual = config.alpha_.min(actual_and_dual);
        sf.primal = actual_and_primal;
        sf.primal_best_child = Some(ShapeFnChildRef::And(and_id));
        sf.heuristic_value = actual_and_heuristic;

        if order_budget == 0 {
            let sf = arena.sf_mut(sf_id);
            sf.expanded = true;
            sf.sample_indices = None;
            sf.bin_sample_indices = None;
            sf.left_local_indices = None;
            sf.right_local_indices = None;
            sf.k_set = None;
        }
    }

    Some(sf_id)
}

fn set_data_state_leaf(arena: &mut Arena, id: DataStateId) {
    let ds = arena.ds_mut(id);
    ds.dual = ds.leaf_error;
    ds.primal = ds.leaf_error;
    ds.primal_best_child = None;
    ds.solved = true;
    ds.expanded = true;
    ds.dual_queue.clear();
    ds.heuristic_queue.clear();
    ds.heuristic_value = ds.init_heuristic_value;
    ds.cart_completed = false;
}

fn mark_data_state_cart_completed(arena: &mut Arena, id: DataStateId) {
    arena.ds_mut(id).cart_completed = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// ShapeFunctionState initialization helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute heuristic for a ShapeFunctionState.
fn compute_sf_heuristic(
    config: &Config,
    search_state: Option<&SearchState>,
    sample_indices: &[u32],
    left_local_indices: &[u32],
    right_local_indices: &[u32],
    depth_budget: u8,
    dual: f64,
    primal: f64,
) -> f64 {
    let nc = config.n_classes as usize;
    match config.heuristic {
        Heuristic::Dual => dual,
        Heuristic::Primal => primal,
        Heuristic::Gini | Heuristic::Entropy => {
            let criterion = if config.heuristic == Heuristic::Gini {
                Criterion::Gini
            } else {
                Criterion::Entropy
            };
            let counts = |indices: &[u32]| -> Vec<u32> {
                let mut c = vec![0; nc];
                for &i in indices {
                    c[config.y[sample_indices[i as usize] as usize] as usize] += 1;
                }
                c
            };
            left_local_indices.len() as f64
                * compute_impurity(
                    criterion,
                    &counts(left_local_indices),
                    left_local_indices.len() as u32,
                )
                + right_local_indices.len() as f64
                    * compute_impurity(
                        criterion,
                        &counts(right_local_indices),
                        right_local_indices.len() as u32,
                    )
        }
        Heuristic::Cart => {
            let child_depth = depth_budget.saturating_sub(1);
            let score = |indices: &[u32]| -> f64 {
                let globals: Vec<u32> = indices
                    .iter()
                    .map(|&i| sample_indices[i as usize])
                    .collect();
                match search_state {
                    Some(search_state) => {
                        cached_cart_heuristic(config, search_state, &globals, child_depth)
                    }
                    None => compute_cart_heuristic(config, &globals, child_depth),
                }
            };
            if left_local_indices.is_empty() || right_local_indices.is_empty() {
                f64::INFINITY
            } else {
                score(left_local_indices) + score(right_local_indices)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Expansion
// ─────────────────────────────────────────────────────────────────────────────

fn cached_multivariate_discretization(
    config: &Config,
    search_state: &SearchState,
    sample_indices: &[u32],
    max_leaf_nodes: usize,
) -> crate::discretize::multivariate::MultivarDiscretizeResult {
    if config.use_cache {
        let key = DiscretizerCacheKey {
            sample_indices: sample_indices.to_vec(),
            max_leaf_nodes,
            criterion: config.criterion,
            n_features: config.n_features,
            n_classes: config.n_classes,
        };
        if let Some(result) = {
            let mut cache = search_state.discretizer_cache.lock().unwrap();
            let result = cache.map.get(&key).cloned();
            if result.is_some() {
                cache.hits += 1;
            } else {
                cache.misses += 1;
            }
            result
        } {
            return result;
        }

        let result = compute_multivariate_discretization(
            config,
            search_state,
            sample_indices,
            max_leaf_nodes,
        );
        let mut cache = search_state.discretizer_cache.lock().unwrap();
        if let Some(existing) = cache.map.get(&key).cloned() {
            cache.hits += 1;
            existing
        } else if search_state.reserve_exact_cache_entry(
            SearchState::discretizer_cache_entry_bytes(&key, &result),
            crate::config::EXACT_CACHE_ENTRY_CAP,
            crate::config::EXACT_CACHE_BYTE_BUDGET,
        ) {
            cache.map.insert(key, result.clone());
            cache.stores += 1;
            result
        } else {
            result
        }
    } else {
        // Cache-off deliberately avoids key allocation, locking, and insertion.
        compute_multivariate_discretization(config, search_state, sample_indices, max_leaf_nodes)
    }
}

fn compute_multivariate_discretization(
    config: &Config,
    search_state: &SearchState,
    sample_indices: &[u32],
    max_leaf_nodes: usize,
) -> crate::discretize::multivariate::MultivarDiscretizeResult {
    let nf = config.n_features as usize;
    let t0 = Instant::now();
    let mut x_local = Vec::with_capacity(sample_indices.len() * nf);
    let mut y_local = Vec::with_capacity(sample_indices.len());
    for &idx in sample_indices {
        let idx = idx as usize;
        x_local.extend_from_slice(&config.x[idx * nf..(idx + 1) * nf]);
        y_local.push(config.y[idx]);
    }
    let result = fit_multivariate_cart(
        &x_local,
        &y_local,
        nf,
        config.n_classes as usize,
        max_leaf_nodes,
        config.criterion,
    );
    search_state
        .discretizer_nanos
        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    result
}

/// Expand a DataState: discretize features, create ShapeFunctionState children.
pub fn expand_data_state(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    ds_id: DataStateId,
) {
    if arena.ds(ds_id).expanded {
        return;
    }

    let sample_indices = match arena.ds_mut(ds_id).sample_indices.take() {
        Some(si) => si,
        None => return,
    };
    let depth_budget = arena.ds(ds_id).depth_budget;

    // Call discretizer
    let nf = config.n_features as usize;
    let nc = config.n_classes as usize;
    let depth = (config.max_depth - depth_budget) as usize;
    let n_cand = config.n_candidates_list[depth] as usize;
    let max_leaf_nodes = n_cand + 1;

    // Collect per-feature thresholds, bin_indices, bin_distributions
    // Structure: Vec<(feature_idx, feature_name, thresholds, bin_indices, bin_distributions)>
    #[allow(clippy::type_complexity)]
    let mut feature_splits: Vec<(u16, String, Vec<f64>, Vec<Vec<u32>>, Vec<Vec<u32>>)> = Vec::new();
    match config.discretizer {
        Discretizer::Univariate => {
            // Build local y as u8
            let y_local: Vec<u8> = sample_indices
                .iter()
                .map(|&idx| config.y[idx as usize])
                .collect();

            for (feat_name, col_indices) in &config.feature_dict {
                let feat_idx = col_indices[0];
                let x_feature: Vec<f64> = sample_indices
                    .iter()
                    .map(|&idx| config.x[idx as usize * nf + feat_idx])
                    .collect();

                let result = discretize(&x_feature, &y_local, nc, max_leaf_nodes, config.criterion);

                if !result.thresholds.is_empty() {
                    let bin_indices_u32: Vec<Vec<u32>> = result
                        .bin_indices
                        .into_iter()
                        .map(|v| v.into_iter().map(|i| i as u32).collect())
                        .collect();
                    feature_splits.push((
                        feat_idx as u16,
                        feat_name.clone(),
                        result.thresholds,
                        bin_indices_u32,
                        result.bin_distributions,
                    ));
                }
            }
        }
        Discretizer::Multivariate => {
            let result = cached_multivariate_discretization(
                config,
                search_state,
                &sample_indices,
                max_leaf_nodes,
            );

            for (&feat_idx, thresholds) in &result.feature_thresholds {
                // Find feature name
                let feat_name = config
                    .feature_dict
                    .iter()
                    .find(|(_, cols)| cols.contains(&feat_idx))
                    .map(|(name, _)| name.clone())
                    .unwrap_or_else(|| format!("feature_{}", feat_idx));

                let bin_indices_u32: Vec<Vec<u32>> = result.feature_bin_indices[&feat_idx]
                    .iter()
                    .map(|v| v.iter().map(|&i| i as u32).collect())
                    .collect();
                feature_splits.push((
                    feat_idx as u16,
                    feat_name,
                    thresholds.clone(),
                    bin_indices_u32,
                    result.feature_bin_distributions[&feat_idx].clone(),
                ));
            }
        }
    }

    // Create ShapeFunctionState children for first-order splits
    if feature_splits.is_empty() {
        // No splits found — will become a leaf via backpropagation.
        // Don't call set_data_state_leaf here: it sets solved=true, and the
        // subsequent backprop_data_state would see old_solved==true, detect no
        // change, and skip propagation to the parent. By deferring the leaf
        // transition to backprop, we guarantee changed=true and propagation.
        arena.ds_mut(ds_id).expanded = true;
        return;
    }

    let lambda_ = config.lambda_;

    // ── COMPUTE PHASE: parallel if enough samples, sequential otherwise ──
    let n_local = sample_indices.len();
    let si_ref = &sample_indices;
    let cache_state: &SearchState = search_state;
    let prepared: Vec<PreparedShapeFnChild> = if n_local >= PARALLEL_EXPANSION_THRESHOLD {
        feature_splits
            .into_par_iter()
            .flat_map(
                |(feat_idx, feat_name, thresholds, bin_indices, bin_distributions)| {
                    let n_bins = thresholds.len() + 1;
                    (0..(n_bins - 1))
                        .filter_map(move |i| {
                            let split: Vec<u8> =
                                (0..n_bins).map(|j| if j <= i { 1 } else { 0 }).collect();
                            let k_set = vec![i as u8];
                            let p = prepare_shape_fn_child(
                                config,
                                Some(cache_state),
                                si_ref,
                                depth_budget,
                                feat_idx,
                                feat_name.clone(),
                                thresholds.clone(),
                                &bin_indices,
                                Some(&bin_distributions),
                                split,
                                k_set,
                                config.max_order.saturating_sub(1),
                            )?;
                            Some(p)
                        })
                        .collect::<Vec<_>>()
                },
            )
            .collect()
    } else {
        feature_splits
            .into_iter()
            .flat_map(
                |(feat_idx, feat_name, thresholds, bin_indices, bin_distributions)| {
                    let n_bins = thresholds.len() + 1;
                    (0..(n_bins - 1))
                        .filter_map(move |i| {
                            let split: Vec<u8> =
                                (0..n_bins).map(|j| if j <= i { 1 } else { 0 }).collect();
                            let k_set = vec![i as u8];
                            let p = prepare_shape_fn_child(
                                config,
                                Some(cache_state),
                                si_ref,
                                depth_budget,
                                feat_idx,
                                feat_name.clone(),
                                thresholds.clone(),
                                &bin_indices,
                                Some(&bin_distributions),
                                split,
                                k_set,
                                config.max_order.saturating_sub(1),
                            )?;
                            Some(p)
                        })
                        .collect::<Vec<_>>()
                },
            )
            .collect()
    };

    // ── SEQUENTIAL PHASE: allocate into arena ─────────────────────────
    let parent_ref = ParentRef::DataState(ds_id);
    for prepared_child in prepared {
        if let Some(sf_id) =
            allocate_prepared_sf_child(arena, config, search_state, prepared_child, parent_ref)
        {
            let child_dual = arena.sf(sf_id).dual + lambda_;
            let child_heuristic = arena.sf(sf_id).heuristic_value;
            let child_ref = DataChildRef::ShapeFn(sf_id);
            let ds = arena.ds_mut(ds_id);
            insort_dual_queue_ds(
                &mut ds.dual_queue,
                OrderedFloat(child_dual),
                OrderedFloat(child_heuristic),
                child_ref,
            );
            insort_heuristic_queue_ds(
                &mut ds.heuristic_queue,
                OrderedFloat(child_heuristic),
                child_ref,
            );
        }
    }

    arena.ds_mut(ds_id).expanded = true;
}

/// Expand a ShapeFunctionState: create AndState + higher-order children.
pub fn expand_shape_fn_state(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    sf_id: ShapeFnStateId,
) {
    if arena.sf(sf_id).expanded {
        return;
    }

    // Create AndState if not already done (eager_expansion already did this)
    if !config.eager_expansion {
        create_and_state_for_sf(arena, config, search_state, sf_id);
    }

    let order_budget = arena.sf(sf_id).order_budget;
    if order_budget == 0 {
        let sf = arena.sf_mut(sf_id);
        sf.expanded = true;
        sf.sample_indices = None;
        sf.bin_sample_indices = None;
        sf.left_local_indices = None;
        sf.right_local_indices = None;
        sf.k_set = None;
        return;
    }

    // Generate higher-order children
    let sf = arena.sf(sf_id);
    let k_set = match &sf.k_set {
        Some(ks) => ks.clone(),
        None => {
            arena.sf_mut(sf_id).expanded = true;
            return;
        }
    };
    let split = sf.split.clone();
    let sample_indices = sf.sample_indices.clone();
    let bin_sample_indices = sf.bin_sample_indices.clone();
    let thresholds = sf.thresholds.clone();
    let feature_idx = sf.feature_idx;
    let feature_name = sf.feature_name.clone();
    let depth_budget = sf.depth_budget;

    let sample_indices = match sample_indices {
        Some(si) => si,
        None => {
            arena.sf_mut(sf_id).expanded = true;
            return;
        }
    };

    let bin_sample_indices_flat = match bin_sample_indices {
        Some(bsi) => bsi,
        None => {
            arena.sf_mut(sf_id).expanded = true;
            return;
        }
    };
    let bin_sample_indices: Vec<Vec<u32>> = (0..bin_sample_indices_flat.n_bins())
        .map(|i| bin_sample_indices_flat.bin(i).to_vec())
        .collect();
    let bin_distributions: Vec<Vec<u32>> = bin_sample_indices
        .iter()
        .map(|bin| {
            let mut counts = vec![0u32; config.n_classes as usize];
            for &local_idx in bin {
                counts[config.y[sample_indices[local_idx as usize] as usize] as usize] += 1;
            }
            counts
        })
        .collect();

    let n_bins = split.len();
    let last_idx = *k_set.last().unwrap() as usize;
    let earliest_valid_bit = last_idx + 1;

    let alpha_ = config.alpha_;

    // ── COMPUTE PHASE: parallel if enough samples, sequential otherwise ──
    let n_local = sample_indices.len();
    let cache_state: &SearchState = search_state;
    let prepared: Vec<PreparedShapeFnChild> = if n_local >= PARALLEL_EXPANSION_THRESHOLD {
        (earliest_valid_bit..n_bins)
            .into_par_iter()
            .filter_map(|new_bit| {
                let mut new_k_set = k_set.clone();
                new_k_set.push(new_bit as u8);

                let mut new_split: Vec<u8> = split[..new_bit].to_vec();
                for &s in &split[new_bit..] {
                    new_split.push(if s == 1 { 0 } else { 1 });
                }

                let p = prepare_shape_fn_child(
                    config,
                    Some(cache_state),
                    &sample_indices,
                    depth_budget,
                    feature_idx,
                    feature_name.clone(),
                    thresholds.clone(),
                    &bin_sample_indices,
                    Some(&bin_distributions),
                    new_split,
                    new_k_set,
                    order_budget.saturating_sub(1),
                )?;
                Some(p)
            })
            .collect()
    } else {
        (earliest_valid_bit..n_bins)
            .filter_map(|new_bit| {
                let mut new_k_set = k_set.clone();
                new_k_set.push(new_bit as u8);

                let mut new_split: Vec<u8> = split[..new_bit].to_vec();
                for &s in &split[new_bit..] {
                    new_split.push(if s == 1 { 0 } else { 1 });
                }

                let p = prepare_shape_fn_child(
                    config,
                    Some(cache_state),
                    &sample_indices,
                    depth_budget,
                    feature_idx,
                    feature_name.clone(),
                    thresholds.clone(),
                    &bin_sample_indices,
                    Some(&bin_distributions),
                    new_split,
                    new_k_set,
                    order_budget.saturating_sub(1),
                )?;
                Some(p)
            })
            .collect()
    };

    // ── SEQUENTIAL PHASE ──────────────────────────────────────────────
    let parent_ref = ParentRef::ShapeFnState(sf_id);
    for prepared_child in prepared {
        if let Some(child_sf_id) =
            allocate_prepared_sf_child(arena, config, search_state, prepared_child, parent_ref)
        {
            let child_dual = arena.sf(child_sf_id).dual + alpha_;
            let child_heuristic = arena.sf(child_sf_id).heuristic_value;
            let child_ref = ShapeFnChildRef::ShapeFn(child_sf_id);
            let sf = arena.sf_mut(sf_id);
            insort_dual_queue_sf(
                &mut sf.dual_queue,
                OrderedFloat(child_dual),
                OrderedFloat(child_heuristic),
                child_ref,
            );
            insort_heuristic_queue_sf(
                &mut sf.heuristic_queue,
                OrderedFloat(child_heuristic),
                child_ref,
            );
        }
    }

    let sf = arena.sf_mut(sf_id);
    sf.expanded = true;
    sf.sample_indices = None;
    sf.bin_sample_indices = None;
    sf.left_local_indices = None;
    sf.right_local_indices = None;
    sf.k_set = None;
}

/// Create an AndState for a ShapeFunctionState (its first-order split).
fn create_and_state_for_sf(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    sf_id: ShapeFnStateId,
) {
    let sf = arena.sf(sf_id);
    let sample_indices = match &sf.sample_indices {
        Some(si) => si.clone(),
        None => return,
    };
    let left_local = match &sf.left_local_indices {
        Some(li) => li.clone(),
        None => return,
    };
    let right_local = match &sf.right_local_indices {
        Some(ri) => ri.clone(),
        None => return,
    };
    let depth_budget = sf.depth_budget;

    // Map local indices to global sample indices
    let left_sample_indices: Vec<u32> = left_local
        .iter()
        .map(|&li| sample_indices[li as usize])
        .collect();
    let right_sample_indices: Vec<u32> = right_local
        .iter()
        .map(|&ri| sample_indices[ri as usize])
        .collect();

    // Create child DataStates with parent pointing to the ShapeFunctionState.
    // In Python, DataState.parent_state = ShapeFunctionState (the SF that created it).
    // Backpropagation chain: child DataState → ShapeFnState → parent DataState.
    let parent_ref = ParentRef::ShapeFnState(sf_id);
    let left_id = create_data_state(
        arena,
        config,
        search_state,
        left_sample_indices,
        depth_budget.saturating_sub(1),
        parent_ref,
    );
    let right_id = create_data_state(
        arena,
        config,
        search_state,
        right_sample_indices,
        depth_budget.saturating_sub(1),
        ParentRef::ShapeFnState(sf_id),
    );

    let and_id = arena.alloc_and_state(left_id, right_id, sf_id);

    if arena.ds(left_id).cart_completed {
        backprop_data_state(arena, config, search_state, left_id, false);
    }
    if arena.ds(right_id).cart_completed {
        backprop_data_state(arena, config, search_state, right_id, false);
    }

    // Insert the AndState into the SF's queues
    let and_dual = arena.and_dual(and_id);
    let and_heuristic = arena.ds(left_id).heuristic_value + arena.ds(right_id).heuristic_value;

    let sf = arena.sf_mut(sf_id);
    let child_ref = ShapeFnChildRef::And(and_id);
    insort_dual_queue_sf(
        &mut sf.dual_queue,
        OrderedFloat(and_dual),
        OrderedFloat(and_heuristic),
        child_ref,
    );
    insort_heuristic_queue_sf(
        &mut sf.heuristic_queue,
        OrderedFloat(and_heuristic),
        child_ref,
    );

    // Free local indices
    let sf = arena.sf_mut(sf_id);
    sf.left_local_indices = None;
    sf.right_local_indices = None;
}

pub fn selection(
    arena: &mut Arena,
    root_id: DataStateId,
    config: &Config,
    search_state: &mut SearchState,
    anytime: bool,
) -> Option<SelectionTarget> {
    select_data_state(arena, root_id, config, search_state, anytime)
}

fn select_data_state(
    arena: &mut Arena,
    ds_id: DataStateId,
    config: &Config,
    search_state: &mut SearchState,
    anytime: bool,
) -> Option<SelectionTarget> {
    if !arena.ds(ds_id).expanded {
        return Some(SelectionTarget::DataState(ds_id));
    }

    // Python: if len(self.dual_queue) == 0 and not self.solved:
    //     raise ValueError("Selection called on DataState with no children, but not marked as solved")
    assert!(
        !arena.ds(ds_id).dual_queue.is_empty() || arena.ds(ds_id).is_solved(),
        "Selection called on expanded DataState {:?} with no children but not solved",
        ds_id
    );

    if anytime {
        // Follow heuristic queue, skip solved children
        let hq = arena.ds(ds_id).heuristic_queue.clone();
        for (_, child_ref) in &hq {
            match child_ref {
                DataChildRef::ShapeFn(sf_id) => {
                    if let Some(sf) = arena.shape_fn_states.get(sf_id.0 as usize) {
                        if !sf.is_solved() {
                            return select_shape_fn_state(
                                arena,
                                *sf_id,
                                config,
                                search_state,
                                anytime,
                            );
                        }
                    }
                }
            }
        }
        // Python: this path doesn't exist — Python would call .expansion() on None and crash.
        // If all children in heuristic_queue are solved, this DataState should be solved too.
        panic!(
            "select_data_state: DataState {:?} exhausted heuristic queue but is not solved. \
             solved={}, dual_queue_len={}, heuristic_queue_len={}, leaf_error={}",
            ds_id,
            arena.ds(ds_id).is_solved(),
            arena.ds(ds_id).dual_queue.len(),
            arena.ds(ds_id).heuristic_queue.len(),
            arena.ds(ds_id).leaf_error
        );
    } else {
        // Follow dual_queue[0]
        let dq = arena.ds(ds_id).dual_queue.clone();
        assert!(
            !dq.is_empty(),
            "select_data_state: non-anytime DataState {:?} has empty dual_queue but not solved",
            ds_id
        );
        let (_, _, child_ref) = &dq[0];
        match child_ref {
            DataChildRef::ShapeFn(sf_id) => {
                assert!(
                    arena.shape_fn_states.contains(sf_id.0 as usize),
                    "select_data_state: dual_queue[0] references pruned ShapeFn {:?}",
                    sf_id
                );
                select_shape_fn_state(arena, *sf_id, config, search_state, anytime)
            }
        }
    }
}

fn select_shape_fn_state(
    arena: &mut Arena,
    sf_id: ShapeFnStateId,
    config: &Config,
    search_state: &mut SearchState,
    anytime: bool,
) -> Option<SelectionTarget> {
    if !arena.sf(sf_id).expanded {
        return Some(SelectionTarget::ShapeFnState(sf_id));
    }

    // Python: assert len(self.dual_queue) > 0, "Selection called on ShapeFunctionState with no children."
    assert!(
        !arena.sf(sf_id).dual_queue.is_empty(),
        "Selection called on expanded ShapeFnState {:?} with no children",
        sf_id
    );

    if anytime {
        let hq = arena.sf(sf_id).heuristic_queue.clone();
        for (_, child_ref) in &hq {
            match child_ref {
                ShapeFnChildRef::And(and_id) => {
                    if arena.and_states.contains(and_id.0 as usize) && !arena.and_is_solved(*and_id)
                    {
                        return select_and_state(arena, *and_id, config, search_state, anytime);
                    }
                }
                ShapeFnChildRef::ShapeFn(child_sf_id) => {
                    if let Some(sf) = arena.shape_fn_states.get(child_sf_id.0 as usize) {
                        if !sf.is_solved() {
                            return select_shape_fn_state(
                                arena,
                                *child_sf_id,
                                config,
                                search_state,
                                anytime,
                            );
                        }
                    }
                }
            }
        }
        // Python: raise ValueError("All children are solved during anytime selection.")
        panic!(
            "select_shape_fn_state: ShapeFnState {:?} exhausted heuristic queue but is not solved. \
             solved={}, dual_queue_len={}, heuristic_queue_len={}",
            sf_id, arena.sf(sf_id).is_solved(),
            arena.sf(sf_id).dual_queue.len(), arena.sf(sf_id).heuristic_queue.len()
        );
    } else {
        let dq = arena.sf(sf_id).dual_queue.clone();
        assert!(
            !dq.is_empty(),
            "select_shape_fn_state: non-anytime ShapeFnState {:?} has empty dual_queue",
            sf_id
        );
        let (_, _, child_ref) = &dq[0];
        match child_ref {
            ShapeFnChildRef::And(and_id) => {
                assert!(
                    arena.and_states.contains(and_id.0 as usize),
                    "select_shape_fn_state: dual_queue[0] references pruned And {:?}",
                    and_id
                );
                select_and_state(arena, *and_id, config, search_state, anytime)
            }
            ShapeFnChildRef::ShapeFn(child_sf_id) => {
                assert!(
                    arena.shape_fn_states.contains(child_sf_id.0 as usize),
                    "select_shape_fn_state: dual_queue[0] references pruned ShapeFn {:?}",
                    child_sf_id
                );
                select_shape_fn_state(arena, *child_sf_id, config, search_state, anytime)
            }
        }
    }
}

fn select_and_state(
    arena: &mut Arena,
    and_id: AndStateId,
    config: &Config,
    search_state: &mut SearchState,
    anytime: bool,
) -> Option<SelectionTarget> {
    let a = arena.as_(and_id);
    let left_id = a.left;
    let right_id = a.right;
    let left_expanded = arena.ds(left_id).expanded;
    let right_expanded = arena.ds(right_id).expanded;

    if !left_expanded && !right_expanded {
        // Neither expanded — expand both (Python: AndState.expansion() expands both)
        // But selection returns one at a time; just pick left first
        return Some(SelectionTarget::DataState(left_id));
    }

    let left_solved = arena.ds(left_id).is_solved();
    let right_solved = arena.ds(right_id).is_solved();

    if left_solved && right_solved {
        // Both children solved — this AndState is solved. Backprop should have caught this.
        panic!(
            "select_and_state: And {:?} has both children solved but was not marked as solved \
             by backpropagation. left={:?} right={:?}",
            and_id, left_id, right_id
        );
    }
    if left_solved {
        return select_data_state(arena, right_id, config, search_state, anytime);
    }
    if right_solved {
        return select_data_state(arena, left_id, config, search_state, anytime);
    }

    // Both unsolved — dispatch to selection scheme
    match config.selection_scheme {
        SelectionScheme::RoundRobin => {
            let choose_left = arena.as_(and_id).choose_left;
            arena.as_mut(and_id).choose_left = !choose_left;
            if choose_left {
                select_data_state(arena, left_id, config, search_state, anytime)
            } else {
                select_data_state(arena, right_id, config, search_state, anytime)
            }
        }
        SelectionScheme::Weighted => {
            let (left_value, right_value) = if anytime {
                (
                    arena.ds(left_id).heuristic_value,
                    arena.ds(right_id).heuristic_value,
                )
            } else {
                (arena.ds(left_id).dual, arena.ds(right_id).dual)
            };
            let total = left_value + right_value;
            if total == 0.0 {
                // Fallback to sample_weighted
                return select_and_sample_weighted(arena, and_id, config, search_state, anytime);
            }
            let left_prob = left_value / total;
            let r: f64 = search_state.rng.gen();
            if r < left_prob {
                select_data_state(arena, left_id, config, search_state, anytime)
            } else {
                select_data_state(arena, right_id, config, search_state, anytime)
            }
        }
        SelectionScheme::SampleWeighted => {
            select_and_sample_weighted(arena, and_id, config, search_state, anytime)
        }
        SelectionScheme::Bfs => {
            let left_id = arena.as_(and_id).left;
            let right_id = arena.as_(and_id).right;
            let (left_value, right_value) = if anytime {
                (
                    arena.ds(left_id).heuristic_value,
                    arena.ds(right_id).heuristic_value,
                )
            } else {
                (arena.ds(left_id).dual, arena.ds(right_id).dual)
            };
            // Select the child with the *worse* (higher) bound to prune it quickly
            if left_value <= right_value {
                select_data_state(arena, right_id, config, search_state, anytime)
            } else {
                select_data_state(arena, left_id, config, search_state, anytime)
            }
        }
        SelectionScheme::Branches => {
            // Always uses primal regardless of `anytime` — deliberate ablation
            // semantic mirroring AndState.branches_selection in src/literati.py.
            // Mirrors Python's `>` comparison: ties route to the right child.
            let left_id = arena.as_(and_id).left;
            let right_id = arena.as_(and_id).right;
            let left_primal = arena.ds(left_id).primal;
            let right_primal = arena.ds(right_id).primal;
            if left_primal > right_primal {
                select_data_state(arena, left_id, config, search_state, anytime)
            } else {
                select_data_state(arena, right_id, config, search_state, anytime)
            }
        }
    }
}

fn select_and_sample_weighted(
    arena: &mut Arena,
    and_id: AndStateId,
    config: &Config,
    search_state: &mut SearchState,
    anytime: bool,
) -> Option<SelectionTarget> {
    let a = arena.as_(and_id);
    let left_n = arena.ds(a.left).n_samples_local as f64;
    let right_n = arena.ds(a.right).n_samples_local as f64;
    let total = left_n + right_n;
    // Python: assert total_samples > 0
    assert!(
        total > 0.0,
        "select_and_sample_weighted: And {:?} has 0 total samples",
        and_id
    );
    let left_prob = left_n / total;
    let r: f64 = search_state.rng.gen();
    let left_id = a.left;
    let right_id = a.right;
    if r < left_prob {
        select_data_state(arena, left_id, config, search_state, anytime)
    } else {
        select_data_state(arena, right_id, config, search_state, anytime)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backpropagation
// ─────────────────────────────────────────────────────────────────────────────

/// Trigger backpropagation from a just-expanded selection target.
/// The target may have been pruned during expansion, so liveness is checked
/// before traversing the parent chain.
pub fn backpropagate_from(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    target: SelectionTarget,
) {
    match target {
        SelectionTarget::DataState(ds_id) => {
            if arena.data_states.contains(ds_id.0 as usize) {
                backprop_data_state(arena, config, search_state, ds_id, true);
            }
        }
        SelectionTarget::ShapeFnState(sf_id) => {
            if arena.shape_fn_states.contains(sf_id.0 as usize) {
                backprop_shape_fn_state(arena, config, search_state, sf_id, true);
            }
        }
    }
}

fn backprop_data_state(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    ds_id: DataStateId,
    propagate: bool,
) {
    if arena.ds(ds_id).cart_completed && !arena.ds(ds_id).solved {
        let (parent, n_samples, bitset, depth_budget, dual, leaf_error, from_cache) = {
            let ds = arena.ds(ds_id);
            (
                ds.parent,
                ds.n_samples_local as usize,
                ds.sample_bitset.clone(),
                ds.depth_budget,
                ds.dual,
                ds.leaf_error,
                ds.from_cache,
            )
        };
        {
            let ds = arena.ds_mut(ds_id);
            ds.primal = dual;
            ds.heuristic_value = dual;
            ds.primal_best_child = None;
            ds.solved = true;
            ds.expanded = true;
            ds.dual_queue.clear();
            ds.heuristic_queue.clear();
        }
        if config.use_cache && !from_cache && leaf_error > 0.0 {
            search_state
                .cache
                .store(n_samples, &bitset, depth_budget, dual, dual, true);
        }
        if propagate {
            if let ParentRef::ShapeFnState(sf_id) = parent {
                backprop_shape_fn_state(arena, config, search_state, sf_id, true);
            }
        }
        return;
    }

    // Eager allocation may create terminal SF children whose AndStates are
    // already solved. Their parent queues are wired now, so transition them
    // through backpropagation before evaluating this DataState.
    let terminal_children = arena
        .ds(ds_id)
        .dual_queue
        .iter()
        .map(|(_, _, child_ref)| match child_ref {
            DataChildRef::ShapeFn(sf_id) => *sf_id,
        })
        .collect();
    backprop_terminal_shape_children(arena, config, search_state, terminal_children);

    let old_dual = arena.ds(ds_id).dual;
    let old_primal = arena.ds(ds_id).primal;
    let old_heuristic = arena.ds(ds_id).heuristic_value;
    let old_solved = arena.ds(ds_id).solved;
    let leaf_error = arena.ds(ds_id).leaf_error;
    let lambda_ = config.lambda_;

    // ── Step 0: Read cache ONCE for cross-path bound tightening ──
    let cached_entry = if config.use_cache {
        let ds = arena.ds(ds_id);
        search_state
            .cache
            .get(
                ds.n_samples_local as usize,
                &ds.sample_bitset,
                ds.depth_budget,
            )
            .cloned()
    } else {
        None
    };

    // First pass: find best primal child
    // Initialize with cached_optimal if available — it's a proven-achievable bound
    let dq: Vec<_> = arena.ds(ds_id).dual_queue.clone();
    let cached_opt = arena.ds(ds_id).cached_optimal.unwrap_or(f64::MAX);
    let mut best_primal = leaf_error.min(cached_opt);
    let mut best_primal_child: Option<DataChildRef> = None;

    for (_, _, child_ref) in &dq {
        let child_primal = match child_ref {
            DataChildRef::ShapeFn(sf_id) => {
                match arena.shape_fn_states.get(sf_id.0 as usize) {
                    Some(sf) => sf.primal + lambda_,
                    None => continue, // pruned
                }
            }
        };
        if child_primal < best_primal {
            best_primal = child_primal;
            best_primal_child = Some(*child_ref);
        }
    }

    if let Some(ref entry) = cached_entry {
        if !entry.solved {
            search_state.cache.partial_hits += 1;
        }
    }

    {
        let ds = arena.ds_mut(ds_id);
        ds.primal_best_child = best_primal_child;
        ds.primal = best_primal;
    }

    // Second pass: prune and rebuild queues
    // cached_optimal is already folded into best_primal (initialized above),
    // so standard pruning with best_primal is cache-aware
    let prune_threshold = best_primal;

    let mut new_dual_queue: Vec<(OrderedFloat<f64>, OrderedFloat<f64>, DataChildRef)> = Vec::new();
    let mut new_heuristic_queue: Vec<(OrderedFloat<f64>, DataChildRef)> = Vec::new();

    for (_, _, child_ref) in &dq {
        let (child_dual, child_heuristic) = match child_ref {
            DataChildRef::ShapeFn(sf_id) => {
                match arena.shape_fn_states.get(sf_id.0 as usize) {
                    Some(sf) => (sf.dual + lambda_, sf.heuristic_value),
                    None => continue, // already removed
                }
            }
        };

        if child_dual <= prune_threshold && child_dual < leaf_error {
            insort_dual_queue_ds(
                &mut new_dual_queue,
                OrderedFloat(child_dual),
                OrderedFloat(child_heuristic),
                *child_ref,
            );
            insort_heuristic_queue_ds(
                &mut new_heuristic_queue,
                OrderedFloat(child_heuristic),
                *child_ref,
            );
        } else {
            prune_child_ref(arena, child_ref);
        }
    }

    // Update state
    if new_dual_queue.is_empty() {
        let has_cached = arena.ds(ds_id).cached_optimal.is_some()
            && arena.ds(ds_id).cached_optimal.unwrap() < leaf_error - 1e-12;
        if has_cached {
            // Cache-hit node with no children.
            // Mark solved with cached bounds — do NOT call set_data_state_leaf.
            let ds = arena.ds_mut(ds_id);
            ds.solved = true;
            ds.expanded = true;
            ds.dual_queue = new_dual_queue;
            ds.heuristic_queue = new_heuristic_queue;
        } else {
            assert!(
                best_primal_child.is_none(),
                "backprop_data_state {:?}: no children left but primal_best_child is Some",
                ds_id
            );
            assert!(
                (best_primal - leaf_error).abs() < 1e-12,
                "backprop_data_state {:?}: no children left but primal ({}) != leaf_error ({})",
                ds_id,
                best_primal,
                leaf_error
            );
            let ds = arena.ds_mut(ds_id);
            ds.dual_queue = new_dual_queue;
            ds.heuristic_queue = new_heuristic_queue;
            set_data_state_leaf(arena, ds_id);
        }
    } else {
        let best_child_solved = match &new_dual_queue[0].2 {
            DataChildRef::ShapeFn(sf_id) => arena
                .shape_fn_states
                .get(sf_id.0 as usize)
                .is_some_and(|sf| sf.is_solved()),
        };

        if best_child_solved {
            // Python: assert self.dual == self.primal, "If best child is solved, dual should equal primal"
            let best_dual = new_dual_queue[0].0.into_inner();
            assert!(
                (best_dual - best_primal).abs() < 1e-12,
                "backprop_data_state {:?}: best child solved but dual ({}) != primal ({})",
                ds_id,
                best_dual,
                best_primal
            );

            // Prune all non-best children
            for (_, _, child_ref) in new_dual_queue.iter().skip(1) {
                prune_child_ref(arena, child_ref);
            }
            let best_entry = new_dual_queue[0];
            let ds = arena.ds_mut(ds_id);
            ds.dual = best_entry.0.into_inner();
            ds.primal_best_child = Some(best_entry.2);
            ds.solved = true;
            ds.expanded = true;
            ds.heuristic_value = best_entry.1.into_inner();
            ds.dual_queue = vec![best_entry];
            ds.heuristic_queue = vec![(best_entry.1, best_entry.2)];

            // Cache store: store solved subproblem if not from cache
            // and not a pure node (leaf_error == 0 means savings are negligible)
            {
                let ds = arena.ds(ds_id);
                if config.use_cache && !ds.from_cache && ds.leaf_error > 0.0 {
                    search_state.cache.store(
                        ds.n_samples_local as usize,
                        &ds.sample_bitset,
                        ds.depth_budget,
                        ds.dual,
                        ds.heuristic_value,
                        true,
                    );
                }
            }
        } else {
            let children_dual = new_dual_queue[0].0.into_inner();
            let cached_opt_for_dual = arena.ds(ds_id).cached_optimal.unwrap_or(0.0);
            let mut tightened_dual = children_dual.max(cached_opt_for_dual.min(leaf_error));

            // ── Step 2: Merge cached dual ──
            if let Some(ref entry) = cached_entry {
                tightened_dual = tightened_dual.max(entry.dual);
            }
            tightened_dual = tightened_dual.min(best_primal); // clamp: dual <= primal

            let ds = arena.ds_mut(ds_id);
            ds.dual = tightened_dual;
            ds.heuristic_value = new_heuristic_queue[0].0.into_inner();
            ds.dual_queue = new_dual_queue;
            ds.heuristic_queue = new_heuristic_queue;
            ds.solved = false;

            // ── Step 4: Write cache ONCE (if improved) ──
            // Re-read cache to avoid TOCTOU (recursive backprop could have
            // written a solved entry between step 0 and here).
            if config.use_cache && !ds.from_cache {
                let final_dual = ds.dual;
                let final_heuristic = ds.heuristic_value;
                search_state.cache.store(
                    ds.n_samples_local as usize,
                    &ds.sample_bitset,
                    ds.depth_budget,
                    final_dual,
                    final_heuristic,
                    false,
                );
            }
        }
    }

    // Python: consistency checks
    // assert old_primal >= self.primal, "Primal should not increase"
    {
        let ds = arena.ds(ds_id);
        assert!(
            old_primal >= ds.primal - 1e-12,
            "backprop_data_state {:?}: primal increased from {} to {}",
            ds_id,
            old_primal,
            ds.primal
        );
        assert!(
            ds.dual <= ds.primal + 1e-12,
            "backprop_data_state {:?}: dual ({}) > primal ({})",
            ds_id,
            ds.dual,
            ds.primal
        );
    }

    // Propagate upward
    if propagate {
        let ds = arena.ds(ds_id);
        let changed = ds.solved != old_solved
            || ds.dual != old_dual
            || ds.primal != old_primal
            || ds.heuristic_value != old_heuristic;

        if changed {
            let parent = ds.parent;
            match parent {
                ParentRef::ShapeFnState(sf_id) => {
                    // Find which AndState this DS belongs to, then backprop the SF
                    backprop_shape_fn_state(arena, config, search_state, sf_id, true);
                }
                ParentRef::None => {} // root — no propagation
                _ => {}
            }
        }
    }
}

/// Backpropagate terminal eager SF children after their parent has inserted
/// them into its queues. This applies to children of both DataStates and
/// higher-order ShapeFnStates; setting solved during allocation would skip the
/// parent notification chain.
fn backprop_terminal_shape_children(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    child_ids: Vec<ShapeFnStateId>,
) {
    for sf_id in child_ids {
        let is_terminal_eager_child =
            arena
                .shape_fn_states
                .get(sf_id.0 as usize)
                .is_some_and(|sf| {
                    sf.expanded
                        && !sf.solved
                        && sf.order_budget == 0
                        && sf
                            .dual_queue
                            .iter()
                            .any(|(_, _, child_ref)| match child_ref {
                                ShapeFnChildRef::And(and_id) => arena.and_is_solved(*and_id),
                                ShapeFnChildRef::ShapeFn(_) => false,
                            })
                });
        if is_terminal_eager_child {
            backprop_shape_fn_state(arena, config, search_state, sf_id, false);
        }
    }
}

fn backprop_shape_fn_state(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    sf_id: ShapeFnStateId,
    propagate: bool,
) {
    // Guard against backprop on pruned SF
    if !arena.shape_fn_states.contains(sf_id.0 as usize) {
        return;
    }

    // A higher-order expansion can eagerly allocate terminal SF children. As
    // above, they must become solved through backpropagation only after this
    // parent SF has wired them into its queues.
    let terminal_children = arena
        .sf(sf_id)
        .dual_queue
        .iter()
        .filter_map(|(_, _, child_ref)| match child_ref {
            ShapeFnChildRef::And(_) => None,
            ShapeFnChildRef::ShapeFn(child_sf_id) => Some(*child_sf_id),
        })
        .collect();
    backprop_terminal_shape_children(arena, config, search_state, terminal_children);

    let old_dual = arena.sf(sf_id).dual;
    let old_primal = arena.sf(sf_id).primal;
    let old_heuristic = arena.sf(sf_id).heuristic_value;
    let old_solved = arena.sf(sf_id).solved;
    let alpha_ = config.alpha_;

    // Python: assert len(self.dual_queue) > 0, "Backpropagation called on ShapeFunctionState with no children."
    let dq: Vec<_> = arena.sf(sf_id).dual_queue.clone();
    assert!(
        !dq.is_empty(),
        "backprop_shape_fn_state: ShapeFnState {:?} has empty dual_queue",
        sf_id
    );

    // First pass: find best primal child
    let mut best_primal = f64::INFINITY;
    let mut best_primal_child: Option<ShapeFnChildRef> = None;

    for (_, _, child_ref) in &dq {
        let (child_primal, penalty) = match child_ref {
            ShapeFnChildRef::And(and_id) => {
                if !arena.and_states.contains(and_id.0 as usize) {
                    continue;
                }
                (arena.and_primal(*and_id), 0.0)
            }
            ShapeFnChildRef::ShapeFn(child_sf_id) => {
                match arena.shape_fn_states.get(child_sf_id.0 as usize) {
                    Some(sf) => (sf.primal, alpha_),
                    None => continue,
                }
            }
        };
        let total = child_primal + penalty;
        if total < best_primal {
            best_primal = total;
            best_primal_child = Some(*child_ref);
        }
    }

    {
        let sf = arena.sf_mut(sf_id);
        sf.primal_best_child = best_primal_child;
        sf.primal = best_primal;
    }

    // Second pass: prune and rebuild queues
    let mut new_dual_queue: Vec<(OrderedFloat<f64>, OrderedFloat<f64>, ShapeFnChildRef)> =
        Vec::new();
    let mut new_heuristic_queue: Vec<(OrderedFloat<f64>, ShapeFnChildRef)> = Vec::new();

    for (_, _, child_ref) in &dq {
        let (child_dual, child_heuristic, penalty) = match child_ref {
            ShapeFnChildRef::And(and_id) => {
                if !arena.and_states.contains(and_id.0 as usize) {
                    continue;
                }
                let d = arena.and_dual(*and_id);
                let a = arena.as_(*and_id);
                let h = arena.ds(a.left).heuristic_value + arena.ds(a.right).heuristic_value;
                (d, h, 0.0)
            }
            ShapeFnChildRef::ShapeFn(child_sf_id) => {
                match arena.shape_fn_states.get(child_sf_id.0 as usize) {
                    Some(sf) => (sf.dual, sf.heuristic_value, alpha_),
                    None => continue,
                }
            }
        };

        let penalized_dual = child_dual + penalty;
        if penalized_dual <= best_primal {
            insort_dual_queue_sf(
                &mut new_dual_queue,
                OrderedFloat(penalized_dual),
                OrderedFloat(child_heuristic),
                *child_ref,
            );
            insort_heuristic_queue_sf(
                &mut new_heuristic_queue,
                OrderedFloat(child_heuristic),
                *child_ref,
            );
        } else {
            prune_sf_child_ref(arena, child_ref);
        }
    }

    // Python: assert len(new_dual_queue) > 0, "All children pruned during backpropagation"
    assert!(
        !new_dual_queue.is_empty(),
        "backprop_shape_fn_state {:?}: all children pruned",
        sf_id
    );
    // Python: assert old_primal >= self.primal, "Primal should not increase"
    assert!(
        old_primal >= best_primal - 1e-12,
        "backprop_shape_fn_state {:?}: primal increased from {} to {}",
        sf_id,
        old_primal,
        best_primal
    );
    // Python: assert new_dual_queue[0][0] <= self.primal
    assert!(
        new_dual_queue[0].0.into_inner() <= best_primal + 1e-12,
        "backprop_shape_fn_state {:?}: dual ({}) > primal ({})",
        sf_id,
        new_dual_queue[0].0,
        best_primal
    );
    // Python: assert self.parent_state is not None
    assert!(
        !matches!(arena.sf(sf_id).parent, ParentRef::None),
        "backprop_shape_fn_state {:?}: parent is None",
        sf_id
    );
    // Python: assert best_primal_child is not None
    assert!(
        best_primal_child.is_some(),
        "backprop_shape_fn_state {:?}: no best primal child found",
        sf_id
    );

    // Update state
    let best_child_solved = match &new_dual_queue[0].2 {
        ShapeFnChildRef::And(and_id) => {
            arena.and_states.contains(and_id.0 as usize) && arena.and_is_solved(*and_id)
        }
        ShapeFnChildRef::ShapeFn(child_sf_id) => arena
            .shape_fn_states
            .get(child_sf_id.0 as usize)
            .is_some_and(|sf| sf.is_solved()),
    };

    if best_child_solved {
        // Python: assert self.dual == self.primal, "If best child is solved, dual should equal primal"
        let best_dual = new_dual_queue[0].0.into_inner();
        assert!(
            (best_dual - best_primal).abs() < 1e-12,
            "backprop_shape_fn_state {:?}: best child solved but dual ({}) != primal ({})",
            sf_id,
            best_dual,
            best_primal
        );

        for (_, _, child_ref) in new_dual_queue.iter().skip(1) {
            prune_sf_child_ref(arena, child_ref);
        }
        let best_entry = new_dual_queue[0];
        let sf = arena.sf_mut(sf_id);
        sf.dual = best_entry.0.into_inner();
        sf.primal_best_child = Some(best_entry.2);
        sf.solved = true;
        sf.expanded = true;
        sf.heuristic_value = best_entry.1.into_inner();
        sf.dual_queue = vec![best_entry];
        sf.heuristic_queue = vec![(best_entry.1, best_entry.2)];
    } else {
        let sf = arena.sf_mut(sf_id);
        sf.dual = new_dual_queue[0].0.into_inner();
        sf.heuristic_value = new_heuristic_queue[0].0.into_inner();
        sf.dual_queue = new_dual_queue;
        sf.heuristic_queue = new_heuristic_queue;
        sf.solved = false;
    }

    // Propagate upward
    let sf = arena.sf(sf_id);
    let changed = sf.solved != old_solved
        || sf.dual != old_dual
        || sf.primal != old_primal
        || sf.heuristic_value != old_heuristic;

    if propagate && changed {
        let parent = sf.parent;
        match parent {
            ParentRef::DataState(ds_id) => {
                backprop_data_state(arena, config, search_state, ds_id, true);
            }
            ParentRef::ShapeFnState(parent_sf_id) => {
                backprop_shape_fn_state(arena, config, search_state, parent_sf_id, true);
            }
            ParentRef::None => {}
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pruning
// ─────────────────────────────────────────────────────────────────────────────

fn prune_child_ref(arena: &mut Arena, child_ref: &DataChildRef) {
    match child_ref {
        DataChildRef::ShapeFn(sf_id) => prune_shape_fn(arena, *sf_id),
    }
}

fn prune_sf_child_ref(arena: &mut Arena, child_ref: &ShapeFnChildRef) {
    match child_ref {
        ShapeFnChildRef::And(and_id) => prune_and_state(arena, *and_id),
        ShapeFnChildRef::ShapeFn(sf_id) => prune_shape_fn(arena, *sf_id),
    }
}

fn prune_shape_fn(arena: &mut Arena, sf_id: ShapeFnStateId) {
    let dq = match arena.shape_fn_states.get_mut(sf_id.0 as usize) {
        Some(sf) => std::mem::take(&mut sf.dual_queue),
        None => return, // already removed
    };
    for (_, _, child_ref) in &dq {
        prune_sf_child_ref(arena, child_ref);
    }
    arena.shape_fn_states.try_remove(sf_id.0 as usize);
}

fn prune_and_state(arena: &mut Arena, and_id: AndStateId) {
    let (left_id, right_id) = match arena.and_states.get(and_id.0 as usize) {
        Some(a) => (a.left, a.right),
        None => return, // already removed
    };
    prune_data_state(arena, left_id);
    prune_data_state(arena, right_id);
    arena.and_states.try_remove(and_id.0 as usize);
}

fn prune_data_state(arena: &mut Arena, ds_id: DataStateId) {
    let dq = match arena.data_states.get_mut(ds_id.0 as usize) {
        Some(ds) => std::mem::take(&mut ds.dual_queue),
        None => return, // already removed
    };
    for (_, _, child_ref) in &dq {
        prune_child_ref(arena, child_ref);
    }
    arena.data_states.try_remove(ds_id.0 as usize);
}

// ─────────────────────────────────────────────────────────────────────────────
// Queue insertion (bisect.insort equivalent)
// ─────────────────────────────────────────────────────────────────────────────

fn insort_dual_queue_ds(
    queue: &mut Vec<(OrderedFloat<f64>, OrderedFloat<f64>, DataChildRef)>,
    dual: OrderedFloat<f64>,
    heuristic: OrderedFloat<f64>,
    child: DataChildRef,
) {
    let pos = queue.partition_point(|entry| (entry.0, entry.1) < (dual, heuristic));
    queue.insert(pos, (dual, heuristic, child));
}

fn insort_heuristic_queue_ds(
    queue: &mut Vec<(OrderedFloat<f64>, DataChildRef)>,
    heuristic: OrderedFloat<f64>,
    child: DataChildRef,
) {
    let pos = queue.partition_point(|entry| entry.0 < heuristic);
    queue.insert(pos, (heuristic, child));
}

fn insort_dual_queue_sf(
    queue: &mut Vec<(OrderedFloat<f64>, OrderedFloat<f64>, ShapeFnChildRef)>,
    dual: OrderedFloat<f64>,
    heuristic: OrderedFloat<f64>,
    child: ShapeFnChildRef,
) {
    let pos = queue.partition_point(|entry| (entry.0, entry.1) < (dual, heuristic));
    queue.insert(pos, (dual, heuristic, child));
}

fn insort_heuristic_queue_sf(
    queue: &mut Vec<(OrderedFloat<f64>, ShapeFnChildRef)>,
    heuristic: OrderedFloat<f64>,
    child: ShapeFnChildRef,
) {
    let pos = queue.partition_point(|entry| entry.0 < heuristic);
    queue.insert(pos, (heuristic, child));
}

fn cart_witness_indices(witness: &CartWitness) -> &[usize] {
    match witness {
        CartWitness::Leaf { local_indices, .. } | CartWitness::Split { local_indices, .. } => {
            local_indices
        }
    }
}

fn cart_witness_cost(witness: &CartWitness) -> f64 {
    match witness {
        CartWitness::Leaf { cost, .. } | CartWitness::Split { cost, .. } => *cost,
    }
}

fn allocate_cart_witness_data_state(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    root_samples: &[u32],
    witness: &CartWitness,
    parent: ParentRef,
) -> DataStateId {
    let samples: Vec<u32> = cart_witness_indices(witness)
        .iter()
        .map(|&idx| root_samples[idx])
        .collect();
    let mut counts = vec![0u32; config.n_classes as usize];
    for &idx in &samples {
        counts[config.y[idx as usize] as usize] += 1;
    }
    let mut pred_class = 0;
    let mut max_count = 0;
    for (class, &count) in counts.iter().enumerate() {
        if count > max_count {
            pred_class = class as i32;
            max_count = count;
        }
    }
    let leaf_error = samples.len() as f64 - max_count as f64;
    let depth_budget = match witness {
        CartWitness::Leaf {
            remaining_depth, ..
        }
        | CartWitness::Split {
            remaining_depth, ..
        } => *remaining_depth as u8,
    };
    let cost = cart_witness_cost(witness);
    let n_samples = samples.len();
    let bitset = sample_bitset(config, &samples);
    let id = arena.alloc_data_state(
        samples,
        depth_budget,
        parent,
        n_samples as u32,
        pred_class,
        leaf_error,
        cost,
        cost,
        cost,
        bitset,
    );
    materialize_cart_witness_into(arena, config, search_state, id, root_samples, witness);
    id
}

fn materialize_cart_witness_into(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    ds_id: DataStateId,
    root_samples: &[u32],
    witness: &CartWitness,
) {
    match witness {
        CartWitness::Leaf {
            pred_class, cost, ..
        } => {
            let ds = arena.ds_mut(ds_id);
            ds.pred_class = *pred_class as i32;
            ds.leaf_error = *cost;
            ds.dual = *cost;
            ds.primal = *cost;
            ds.heuristic_value = *cost;
            ds.init_heuristic_value = *cost;
            ds.cart_completed = false;
            set_data_state_leaf(arena, ds_id);
            arena.ds_mut(ds_id).sample_indices = None;
        }
        CartWitness::Split {
            local_indices,
            remaining_depth,
            feature_idx,
            threshold,
            cost,
            left,
            right,
        } => {
            let node_samples: Vec<u32> =
                local_indices.iter().map(|&idx| root_samples[idx]).collect();
            let positions: HashMap<usize, u32> = local_indices
                .iter()
                .enumerate()
                .map(|(position, &idx)| (idx, position as u32))
                .collect();
            let left_local: Vec<u32> = cart_witness_indices(left)
                .iter()
                .map(|idx| positions[idx])
                .collect();
            let right_local: Vec<u32> = cart_witness_indices(right)
                .iter()
                .map(|idx| positions[idx])
                .collect();
            let feature_name = config
                .feature_dict
                .iter()
                .find(|(_, columns)| columns.contains(feature_idx))
                .map(|(name, _)| name.clone())
                .unwrap_or_else(|| format!("feature_{}", feature_idx));
            let child_cost = cart_witness_cost(left) + cart_witness_cost(right);
            let sf_id = arena.alloc_shape_fn_state(
                node_samples,
                *remaining_depth as u8,
                ParentRef::DataState(ds_id),
                *feature_idx as u16,
                feature_name,
                vec![*threshold],
                FlatBins::from_bins(&[left_local.clone(), right_local.clone()]),
                vec![1, 0],
                vec![0],
                0,
                left_local,
                right_local,
                child_cost,
                child_cost,
                child_cost,
            );
            let left_id = allocate_cart_witness_data_state(
                arena,
                config,
                search_state,
                root_samples,
                left,
                ParentRef::ShapeFnState(sf_id),
            );
            let right_id = allocate_cart_witness_data_state(
                arena,
                config,
                search_state,
                root_samples,
                right,
                ParentRef::ShapeFnState(sf_id),
            );
            let and_id = arena.alloc_and_state(left_id, right_id, sf_id);
            let sf_child = ShapeFnChildRef::And(and_id);
            let sf = arena.sf_mut(sf_id);
            sf.dual_queue = vec![(OrderedFloat(child_cost), OrderedFloat(child_cost), sf_child)];
            sf.heuristic_queue = vec![(OrderedFloat(child_cost), sf_child)];
            backprop_shape_fn_state(arena, config, search_state, sf_id, false);
            {
                let sf = arena.sf_mut(sf_id);
                sf.sample_indices = None;
                sf.bin_sample_indices = None;
                sf.left_local_indices = None;
                sf.right_local_indices = None;
                sf.k_set = None;
            }

            let reconstructed_cost = child_cost + config.lambda_;
            assert!(
                (*cost - reconstructed_cost).abs() <= 1e-12,
                "CART witness cost {cost} differs from child costs plus lambda {reconstructed_cost}"
            );
            let ds_child = DataChildRef::ShapeFn(sf_id);
            let ds = arena.ds_mut(ds_id);
            ds.cart_completed = false;
            ds.dual_queue = vec![(OrderedFloat(*cost), OrderedFloat(*cost), ds_child)];
            ds.heuristic_queue = vec![(OrderedFloat(*cost), ds_child)];
            ds.solved = false;
            backprop_data_state(arena, config, search_state, ds_id, false);
            arena.ds_mut(ds_id).sample_indices = None;
        }
    }
}

fn primal_and_state(arena: &Arena, sf_id: ShapeFnStateId) -> Option<AndStateId> {
    match arena.sf(sf_id).primal_best_child? {
        ShapeFnChildRef::And(and_id) => Some(and_id),
        ShapeFnChildRef::ShapeFn(next) => primal_and_state(arena, next),
    }
}

fn materialize_reachable_cart_completions(
    arena: &mut Arena,
    config: &Config,
    search_state: &mut SearchState,
    ds_id: DataStateId,
) {
    if arena.ds(ds_id).cart_completed {
        let samples = arena.ds(ds_id).sample_indices.clone().unwrap();
        let depth = arena.ds(ds_id).depth_budget;
        let expected = arena.ds(ds_id).dual;
        let (cost, _, witness) = compute_cart_heuristic_with_subtrees(config, &samples, depth);
        assert_eq!(cost, expected);
        materialize_cart_witness_into(
            arena,
            config,
            search_state,
            ds_id,
            &samples,
            &witness.unwrap(),
        );
        return;
    }
    let Some(DataChildRef::ShapeFn(sf_id)) = arena.ds(ds_id).primal_best_child else {
        return;
    };
    let Some(and_id) = primal_and_state(arena, sf_id) else {
        return;
    };
    let left = arena.as_(and_id).left;
    let right = arena.as_(and_id).right;
    materialize_reachable_cart_completions(arena, config, search_state, left);
    materialize_reachable_cart_completions(arena, config, search_state, right);
}

pub fn run_ao_star(config: &Config, search_state: &mut SearchState) -> SearchResult {
    let mut arena = Arena::new();

    // Start the clock before building the root so total_time brackets the
    // root DataState's CART-heuristic computation (which happens inside
    // create_data_state). Otherwise the root heuristic would be counted into
    // cart_heuristic_time but not total_time.
    let start = Instant::now();
    search_state
        .cache
        .reset(config.n_samples as usize, config.use_cache);

    let sample_indices: Vec<u32> = (0..config.n_samples).collect();
    let root_id = create_data_state(
        &mut arena,
        config,
        search_state,
        sample_indices,
        config.max_depth,
        ParentRef::None,
    );
    if arena.ds(root_id).cart_completed {
        backprop_data_state(&mut arena, config, search_state, root_id, false);
    }

    let mut iteration: usize = 0;
    let n = config.n_samples as f64;

    // Tracking: mirror Python's track list
    let mut track: Vec<TrackEntry> = Vec::new();
    let mut best_dual = f64::NEG_INFINITY;
    let mut best_primal = f64::INFINITY;

    // Memory guard: probe /proc/self/status once to check platform support
    let memory_guard_active = if config.memory_byte_limit > 0 {
        read_rss_bytes().is_some()
    } else {
        false
    };
    let mut terminated_by_memory = false;

    while !arena.ds(root_id).is_solved()
        && start.elapsed().as_secs_f64() < config.time_limit
        && (config.max_iter < 0 || (iteration as i64) < config.max_iter)
    {
        let anytime = config.anytime;
        let target = selection(&mut arena, root_id, config, search_state, anytime).expect(
            "selection returned None but root is not solved — backpropagation invariant violated",
        );

        match target {
            SelectionTarget::DataState(ds_id) => {
                expand_data_state(&mut arena, config, search_state, ds_id);
            }
            SelectionTarget::ShapeFnState(sf_id) => {
                expand_shape_fn_state(&mut arena, config, search_state, sf_id);
            }
        }

        backpropagate_from(&mut arena, config, search_state, target);
        iteration += 1;

        // Memory guard check
        if memory_guard_active && iteration.is_multiple_of(MEMORY_CHECK_INTERVAL) {
            if let Some(rss) = read_rss_bytes() {
                if rss >= config.memory_byte_limit {
                    if config.verbose {
                        eprintln!(
                            "Memory guard triggered at iteration {}: RSS={:.1} GB, limit={:.1} GB",
                            iteration,
                            rss as f64 / (1024.0 * 1024.0 * 1024.0),
                            config.memory_byte_limit as f64 / (1024.0 * 1024.0 * 1024.0),
                        );
                    }
                    terminated_by_memory = true;
                    break;
                }
            }
        }

        // Track when dual or primal improves
        let dual = arena.ds(root_id).dual;
        let primal = arena.ds(root_id).primal;
        let mut update = false;
        if dual > best_dual {
            best_dual = dual;
            update = true;
        }
        if primal < best_primal {
            best_primal = primal;
            update = true;
        }
        if update {
            let gap = if primal > 0.0 {
                (primal - dual) / primal
            } else {
                0.0
            };
            track.push(TrackEntry {
                iteration,
                dual: dual / n,
                primal: primal / n,
                gap,
                time: start.elapsed().as_secs_f64(),
            });
        }

        if config.verbose && iteration.is_multiple_of(10000) {
            let dual_frac = dual / n;
            let primal_frac = primal / n;
            let gap = if primal > 0.0 {
                (primal - dual) / primal
            } else {
                0.0
            };
            eprintln!(
                "Iteration {}: dual={:.4}, primal={:.4}, gap={:.4}, time={:.2}s",
                iteration,
                dual_frac,
                primal_frac,
                gap,
                start.elapsed().as_secs_f64()
            );
        }
    }

    // CART equality may close states using only an exact scalar during the
    // timed search. Re-run CART without a deadline and materialize every such
    // state on the reachable primal path before export or prediction.
    if config.memory_byte_limit <= 0 {
        materialize_reachable_cart_completions(&mut arena, config, search_state, root_id);
    }

    // Final entry (matches Python's post-loop append)
    let final_dual = arena.ds(root_id).dual / n;
    let final_primal = arena.ds(root_id).primal / n;
    let final_gap = if final_primal > 0.0 {
        (final_primal - final_dual) / final_primal
    } else {
        0.0
    };
    track.push(TrackEntry {
        iteration: iteration + 1,
        dual: final_dual,
        primal: final_primal,
        gap: final_gap,
        time: start.elapsed().as_secs_f64(),
    });

    let total_time = start.elapsed().as_secs_f64();
    let cart_heuristic_time = config.cart_heuristic_nanos.load(Ordering::Relaxed) as f64 / 1e9;
    let discretizer_time = search_state.discretizer_nanos.load(Ordering::Relaxed) as f64 / 1e9;
    let (cart_cache_hits, cart_cache_misses, cart_cache_stores) = if config.use_cache {
        let cache = search_state.cart_cache.lock().unwrap();
        (cache.hits, cache.misses, cache.stores)
    } else {
        (0, 0, 0)
    };
    let (discretizer_cache_hits, discretizer_cache_misses, discretizer_cache_stores) =
        if config.use_cache {
            let cache = search_state.discretizer_cache.lock().unwrap();
            (cache.hits, cache.misses, cache.stores)
        } else {
            (0, 0, 0)
        };
    let (exact_cache_cap_rejections, exact_cache_bytes) = if config.use_cache {
        let accounting = search_state.exact_cache_accounting.lock().unwrap();
        (
            search_state
                .exact_cache_cap_rejections
                .load(Ordering::Relaxed),
            accounting.bytes,
        )
    } else {
        (0, 0)
    };

    SearchResult {
        root_id,
        arena,
        iterations_run: iteration,
        track,
        terminated_by_memory,
        memory_guard_active,
        total_time,
        cart_heuristic_time,
        cache_hits: search_state.cache.hits,
        cache_stores: search_state.cache.stores,
        cache_partial_hits: search_state.cache.partial_hits,
        cart_cache_hits,
        cart_cache_misses,
        cart_cache_stores,
        discretizer_cache_hits,
        discretizer_cache_misses,
        discretizer_cache_stores,
        exact_cache_cap_rejections,
        exact_cache_bytes,
        discretizer_time,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn predict_cart_witness(witness: &CartWitness, row: &[f64]) -> i32 {
        match witness {
            CartWitness::Leaf { pred_class, .. } => *pred_class as i32,
            CartWitness::Split {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                if row[*feature_idx] <= *threshold {
                    predict_cart_witness(left, row)
                } else {
                    predict_cart_witness(right, row)
                }
            }
        }
    }

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
            max_depth: 4,
            lambda_: 0.0,
            alpha_: 0.0,
            max_order: 1,
            n_candidates_list: vec![3; 4],
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

    fn separable_cart_config(lambda_: f64) -> Config {
        let mut config = make_config(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0, 0, 0, 1, 1, 1],
            1,
            2,
        );
        config.max_depth = 1;
        config.lambda_ = lambda_;
        config.heuristic = Heuristic::Cart;
        config
    }

    #[test]
    fn exact_cart_equality_closes_root_without_ao_iterations_and_exports_witness() {
        let config = separable_cart_config(0.0);
        let samples: Vec<u32> = (0..config.n_samples).collect();
        let (cart_cost, _, witness) =
            compute_cart_heuristic_with_subtrees(&config, &samples, config.max_depth);
        let witness = witness.unwrap();
        let direct_predictions: Vec<i32> = config
            .x
            .chunks_exact(config.n_features as usize)
            .map(|row| predict_cart_witness(&witness, row))
            .collect();

        let mut search_state = SearchState::new(42);
        let result = run_ao_star(&config, &mut search_state);
        let root = result.arena.ds(result.root_id);
        let predictions = crate::predict::batch_predict(
            &result.arena,
            &config,
            &config.x,
            config.n_samples as usize,
            config.n_features as usize,
            result.root_id,
        );
        let tree = crate::tree_export::export_tree(&result.arena, &config, result.root_id);

        assert_eq!(result.iterations_run, 0);
        assert!(root.is_solved());
        assert!(
            !root.cart_completed,
            "the deferred witness must be materialized"
        );
        assert_eq!(root.primal, cart_cost);
        assert_eq!(root.dual, cart_cost);
        assert_eq!(predictions, direct_predictions);
        assert_eq!(predictions, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.iter().filter(|node| node.is_leaf).count(), 2);
    }

    #[test]
    fn cart_equality_is_pending_until_data_state_backpropagation() {
        let config = separable_cart_config(0.0);
        let mut arena = Arena::new();
        let mut search_state = SearchState::new(42);
        search_state
            .cache
            .reset(config.n_samples as usize, config.use_cache);
        let samples: Vec<u32> = (0..config.n_samples).collect();
        let root_id = create_data_state(
            &mut arena,
            &config,
            &mut search_state,
            samples,
            config.max_depth,
            ParentRef::None,
        );

        assert!(arena.ds(root_id).cart_completed);
        assert!(!arena.ds(root_id).is_solved());
        backprop_data_state(&mut arena, &config, &mut search_state, root_id, false);
        assert!(arena.ds(root_id).is_solved());
        assert_eq!(arena.ds(root_id).primal, arena.ds(root_id).dual);
    }

    #[test]
    fn cart_non_equality_does_not_close_root() {
        let mut config = make_config(vec![0.0, 1.0, 2.0, 3.0], vec![0, 1, 0, 1], 1, 2);
        config.max_depth = 1;
        config.heuristic = Heuristic::Cart;
        config.max_iter = 0;

        let mut search_state = SearchState::new(42);
        let result = run_ao_star(&config, &mut search_state);
        let root = result.arena.ds(result.root_id);

        assert_eq!(result.iterations_run, 0);
        assert_eq!(root.dual, 0.0);
        assert!(root.heuristic_value > root.dual);
        assert!(!root.is_solved());
        assert!(!root.cart_completed);
    }

    #[test]
    fn exact_cart_equality_closes_with_positive_lambda() {
        let config = separable_cart_config(1.0);
        let mut search_state = SearchState::new(42);
        let result = run_ao_star(&config, &mut search_state);
        let root = result.arena.ds(result.root_id);

        assert_eq!(result.iterations_run, 0);
        assert!(root.is_solved());
        assert_eq!(root.primal, 1.0);
        assert_eq!(root.dual, 1.0);
        assert_eq!(
            crate::tree_export::export_tree(&result.arena, &config, result.root_id).len(),
            3
        );
    }

    #[test]
    fn cart_completion_is_independent_of_adaptive_candidate_counts() {
        let mut config = separable_cart_config(0.0);
        config.n_candidates_list = vec![0];

        let mut search_state = SearchState::new(42);
        let result = run_ao_star(&config, &mut search_state);

        assert_eq!(result.iterations_run, 0);
        assert!(result.arena.ds(result.root_id).is_solved());
        assert_eq!(
            crate::tree_export::export_tree(&result.arena, &config, result.root_id).len(),
            3
        );
    }

    #[test]
    fn cart_witness_reconstruction_runs_after_the_deadline() {
        let mut config = separable_cart_config(0.0);
        config.time_limit = 0.0;

        let mut search_state = SearchState::new(42);
        let result = run_ao_star(&config, &mut search_state);

        assert_eq!(result.iterations_run, 0);
        assert!(result.total_time > config.time_limit);
        assert_eq!(
            crate::tree_export::export_tree(&result.arena, &config, result.root_id).len(),
            3
        );
    }

    #[test]
    fn cart_completion_is_disabled_when_memory_guard_is_requested() {
        let mut config = separable_cart_config(0.0);
        config.memory_byte_limit = 1;
        config.max_iter = 0;

        let mut search_state = SearchState::new(42);
        let result = run_ao_star(&config, &mut search_state);
        let root = result.arena.ds(result.root_id);

        assert_eq!(result.iterations_run, 0);
        assert!(!root.cart_completed);
        assert!(!root.is_solved());
        assert_eq!(root.primal, root.leaf_error);
        assert_eq!(
            crate::tree_export::export_tree(&result.arena, &config, result.root_id).len(),
            1
        );
    }

    #[test]
    fn partial_cache_does_not_supply_primal_bound() {
        let config = make_config(vec![0.0, 1.0], vec![0, 1], 1, 2);
        let samples = vec![0, 1];
        let mut state = SearchState::new(42);
        state.cache.reset(config.n_samples as usize, true);
        state.cache.store(
            samples.len(),
            &sample_bitset(&config, &samples),
            2,
            0.5,
            0.25,
            false,
        );

        let mut arena = Arena::new();
        let id = create_data_state(
            &mut arena,
            &config,
            &mut state,
            samples.clone(),
            2,
            ParentRef::None,
        );
        let ds = arena.ds(id);

        assert_eq!(ds.dual, 0.5);
        assert_eq!(ds.primal, ds.leaf_error);
        assert_eq!(ds.heuristic_value, 0.25);
        assert!(!ds.solved);
    }

    #[test]
    fn exact_adaptive_cache_preserves_order_nan_and_result_isolation() {
        let mut config = make_config(
            vec![f64::NAN, 0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 0, 1, 1, 1],
            1,
            2,
        );
        config.heuristic = Heuristic::Cart;
        config.discretizer = Discretizer::Multivariate;
        let state = SearchState::new(42);
        let ordered = vec![0, 1, 2, 3, 4, 5];

        let mut caller_copy = cached_multivariate_discretization(&config, &state, &ordered, 3);
        assert!(!caller_copy.feature_thresholds.is_empty());
        caller_copy.feature_thresholds.clear();
        let cached = cached_multivariate_discretization(&config, &state, &ordered, 3);
        assert!(
            !cached.feature_thresholds.is_empty(),
            "cached discretizer results must be cloned before callers mutate local indices"
        );
        cached_multivariate_discretization(&config, &state, &[5, 4, 3, 2, 1, 0], 3);
        cached_cart_heuristic(&config, &state, &ordered, 2);
        cached_cart_heuristic(&config, &state, &[5, 4, 3, 2, 1, 0], 2);

        let discretizer = state.discretizer_cache.lock().unwrap();
        assert_eq!((discretizer.stores, discretizer.hits), (2, 1));
        let discretizer_entries = discretizer.map.len();
        drop(discretizer);
        let cart = state.cart_cache.lock().unwrap();
        assert!(cart.stores >= 2);
        let cart_entries = cart.map.len();
        drop(cart);
        let accounting = state.exact_cache_accounting.lock().unwrap();
        assert_eq!(accounting.entries, discretizer_entries + cart_entries);
        assert!(accounting.bytes > 0);
    }

    #[test]
    fn cart_cache_concurrent_callers_store_once() {
        let n = 512;
        let mut config = make_config(
            (0..n).map(|i| i as f64).collect(),
            (0..n).map(|i| u8::from(i >= n / 2)).collect(),
            1,
            2,
        );
        config.heuristic = Heuristic::Cart;
        let state = SearchState::new(42);
        let indices: Vec<u32> = (0..n as u32).collect();
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                use rayon::prelude::*;
                (0..8usize).into_par_iter().for_each(|_| {
                    cached_cart_heuristic(&config, &state, &indices, 2);
                });
            });
        let cache = state.cart_cache.lock().unwrap();
        assert_eq!(cache.stores, 3);
        assert_eq!(cache.hits, 7);
        assert_eq!(cache.map.len(), 3);
    }

    #[test]
    fn cart_rollout_warms_exact_depth_descendants() {
        let mut config = make_config(
            (0..8).map(|i| i as f64).collect(),
            vec![0, 0, 1, 1, 0, 0, 1, 1],
            1,
            2,
        );
        config.heuristic = Heuristic::Cart;
        let state = SearchState::new(42);
        let root_indices: Vec<u32> = (0..8).collect();
        cached_cart_heuristic(&config, &state, &root_indices, 2);

        let descendants: Vec<_> = state
            .cart_cache
            .lock()
            .unwrap()
            .map
            .iter()
            .filter(|(key, _)| key.sample_indices != root_indices)
            .map(|(key, &value)| (key.clone(), value))
            .collect();
        assert!(!descendants.is_empty());
        for (key, value) in &descendants {
            assert_eq!(
                *value,
                compute_cart_heuristic(&config, &key.sample_indices, key.depth)
            );
        }

        let (child_key, child_cost) = descendants.iter().find(|(key, _)| key.depth == 1).unwrap();
        let depth_zero_key = CartCacheKey {
            sample_indices: child_key.sample_indices.clone(),
            depth: 0,
            criterion: child_key.criterion,
            lambda_bits: child_key.lambda_bits,
            n_features: child_key.n_features,
        };
        assert!(!state
            .cart_cache
            .lock()
            .unwrap()
            .map
            .contains_key(&depth_zero_key));
        assert_eq!(
            cached_cart_heuristic(&config, &state, &child_key.sample_indices, 1),
            *child_cost
        );
        let depth_zero_cost = cached_cart_heuristic(&config, &state, &child_key.sample_indices, 0);
        assert_eq!(
            depth_zero_cost,
            compute_cart_heuristic(&config, &child_key.sample_indices, 0)
        );
        assert!(state
            .cart_cache
            .lock()
            .unwrap()
            .map
            .contains_key(&depth_zero_key));
    }

    #[test]
    fn pure_data_root_is_leaf() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<u8> = vec![0, 0, 0, 0];
        let config = make_config(x, y, 1, 1);
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        assert!(result.arena.ds(result.root_id).is_solved());
        assert_eq!(result.iterations_run, 0);
        assert!((result.arena.ds(result.root_id).leaf_error - 0.0).abs() < 1e-12);
    }

    #[test]
    fn separable_data_finds_optimal() {
        // Perfectly separable in 1D
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let mut config = make_config(x, y, 1, 2);
        config.max_depth = 1;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        assert!(result.arena.ds(result.root_id).is_solved());
        // primal should be 0 (perfect split) + lambda*1 = 0
        assert!((result.arena.ds(result.root_id).primal - 0.0).abs() < 1e-12);
    }

    #[test]
    fn branches_scheme_solves_depth2_problem() {
        // Same dataset as search_descends_past_depth1: 3 classes, requires
        // depth-2 to perfectly separate. This test exercises the
        // SelectionScheme::Branches AND-node dispatch arm — depth-1
        // DataStates are unsolved when first reached, so select_and_state
        // routes through the new arm before backpropagation.
        let mut x = Vec::new();
        let mut y: Vec<u8> = Vec::new();
        for v in &[1.0, 2.0, 3.0, 1.5, 2.5] {
            x.push(*v);
            x.push(5.0);
            y.push(0u8);
        }
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(1.0);
            y.push(1u8);
        }
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(9.0);
            y.push(2u8);
        }
        let mut config = make_config(x, y, 2, 3);
        config.max_depth = 2;
        config.lambda_ = 0.0;
        config.max_iter = 5000;
        config.selection_scheme = SelectionScheme::Branches;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);

        let root = result.arena.ds(result.root_id);
        assert!(
            root.primal < 1.0,
            "Branches scheme should achieve near-0 errors at depth 2, got primal={}",
            root.primal
        );
    }

    #[test]
    fn max_iter_terminates() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y: Vec<u8> = vec![0, 1, 0, 1, 1, 0, 1, 0];
        let mut config = make_config(x, y, 1, 2);
        config.max_iter = 3;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        assert!(result.iterations_run <= 3);
    }

    #[test]
    fn data_state_init_leaf_error() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1];
        let config = make_config(x, y, 1, 2);
        let mut arena = Arena::new();
        let mut ss = SearchState::new(42);
        let indices: Vec<u32> = (0..5).collect();
        let id = create_data_state(&mut arena, &config, &mut ss, indices, 3, ParentRef::None);
        // leaf_error = 5 - 3 = 2
        assert!((arena.ds(id).leaf_error - 2.0).abs() < 1e-12);
        assert_eq!(arena.ds(id).pred_class, 0);
    }

    #[test]
    fn data_state_depth_zero_is_leaf() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<u8> = vec![0, 0, 1, 1];
        let config = make_config(x, y, 1, 2);
        let mut arena = Arena::new();
        let mut ss = SearchState::new(42);
        let indices: Vec<u32> = (0..4).collect();
        let id = create_data_state(&mut arena, &config, &mut ss, indices, 0, ParentRef::None);
        assert!(arena.ds(id).is_solved());
        assert!(arena.ds(id).is_leaf());
    }

    #[test]
    fn separable_with_lambda_penalty() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let mut config = make_config(x, y, 1, 2);
        config.lambda_ = 2.0; // penalty of 2 per split
        config.max_depth = 1;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        // Perfect split gives 0 misclassifications + 2.0 penalty = 2.0
        // Leaf gives 3 misclassifications
        // Split is better (2 < 3)
        assert!(result.arena.ds(result.root_id).is_solved());
        assert!((result.arena.ds(result.root_id).primal - 2.0).abs() < 1e-12);
    }

    #[test]
    fn multivariate_discretizer_works() {
        let x = vec![
            1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0,
        ];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let mut config = make_config(x, y, 2, 2);
        config.discretizer = Discretizer::Multivariate;
        config.max_depth = 1;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        assert!(result.arena.ds(result.root_id).is_solved());
    }

    #[test]
    fn anytime_mode_works() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let mut config = make_config(x, y, 1, 2);
        config.anytime = true;
        config.max_depth = 1;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        assert!(result.arena.ds(result.root_id).is_solved());
    }

    /// Regression test: data that needs depth >= 2 to classify well.
    /// 3-class data where feature 0 separates class 0 from {1,2},
    /// and feature 1 separates class 1 from class 2.
    /// Depth 1 can only use one feature → misclassifies one class.
    /// Depth 2 can split on both → perfect classification.
    /// This tests that backpropagation correctly propagates bounds from
    /// depth-2 children back to the root (the ParentRef bug).
    #[test]
    fn depth2_improves_over_depth1() {
        let mut x = Vec::new();
        let mut y: Vec<u8> = Vec::new();
        // Class 0: low feature 0, any feature 1
        for v in &[1.0, 2.0, 3.0, 1.5, 2.5] {
            x.push(*v);
            x.push(5.0);
            y.push(0u8);
        }
        // Class 1: high feature 0, low feature 1
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(1.0);
            y.push(1u8);
        }
        // Class 2: high feature 0, high feature 1
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(9.0);
            y.push(2u8);
        }
        let mut config = make_config(x, y, 2, 3);
        config.max_depth = 2;
        config.lambda_ = 0.0;
        config.max_iter = 5000;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);

        let root = result.arena.ds(result.root_id);
        assert!(
            root.primal < root.leaf_error,
            "Depth-2 tree should improve over leaf: primal={} vs leaf_error={}",
            root.primal,
            root.leaf_error
        );
        // Should achieve 0 errors (perfectly separable at depth 2)
        assert!(
            root.primal <= 1.0,
            "Should achieve near-0 errors at depth 2, got primal={}",
            root.primal
        );
    }

    /// Regression test: verify the search actually reaches depth 2+ nodes.
    /// Uses data where depth 1 gives ~50% accuracy but depth 2 gives ~100%.
    /// Checks that iterations_run > number of root-level SFs, proving the
    /// search descended past the first level.
    #[test]
    fn search_descends_past_depth1() {
        // Two features, each separable into 2 bins.
        // Feature 0 separates classes A vs B+C, Feature 1 separates B vs C.
        // Need depth 2 for full separation.
        let mut x = Vec::new();
        let mut y: Vec<u8> = Vec::new();
        // Class 0: low on feature 0
        for v in &[1.0, 2.0, 3.0, 1.5, 2.5] {
            x.push(*v);
            x.push(5.0); // feature 1 doesn't matter
            y.push(0u8);
        }
        // Class 1: high on feature 0, low on feature 1
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(1.0);
            y.push(1u8);
        }
        // Class 2: high on feature 0, high on feature 1
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(9.0);
            y.push(2u8);
        }
        let mut config = make_config(x, y, 2, 3);
        config.max_depth = 2;
        config.lambda_ = 0.0;
        config.max_iter = 5000;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);

        // The tree should be deeper than 1
        let nodes = crate::tree_export::export_tree(&result.arena, &config, result.root_id);
        let max_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);
        assert!(
            max_depth >= 2,
            "Should find a depth-2 tree, got depth {}",
            max_depth
        );

        // Should achieve 0 errors (perfectly separable at depth 2)
        let root = result.arena.ds(result.root_id);
        assert!(
            root.primal < 1.0,
            "Should achieve near-0 errors at depth 2, got primal={}",
            root.primal
        );
    }

    /// Regression test: backpropagation from depth-2 child DataStates must
    /// update the root's primal bound. Without parent refs, the root primal
    /// stays at the depth-1 value even when depth-2 finds a better tree.
    #[test]
    fn backprop_updates_root_primal_from_depth2() {
        // Same data as search_descends_past_depth1
        let mut x = Vec::new();
        let mut y: Vec<u8> = Vec::new();
        for v in &[1.0, 2.0, 3.0, 1.5, 2.5] {
            x.push(*v);
            x.push(5.0);
            y.push(0u8);
        }
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(1.0);
            y.push(1u8);
        }
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(9.0);
            y.push(2u8);
        }
        let _n = y.len();

        // First: run with max_depth=1 to get baseline primal
        let mut config1 = make_config(x.clone(), y.clone(), 2, 3);
        config1.max_depth = 1;
        config1.lambda_ = 0.0;
        config1.max_iter = 5000;
        let mut ss1 = SearchState::new(42);
        let result1 = run_ao_star(&config1, &mut ss1);
        let primal_depth1 = result1.arena.ds(result1.root_id).primal;

        // Then: run with max_depth=2
        let mut config2 = make_config(x, y, 2, 3);
        config2.max_depth = 2;
        config2.lambda_ = 0.0;
        config2.max_iter = 5000;
        let mut ss2 = SearchState::new(42);
        let result2 = run_ao_star(&config2, &mut ss2);
        let primal_depth2 = result2.arena.ds(result2.root_id).primal;

        // Depth-1 can only split on one feature → 5 errors (one class mixed)
        // Depth-2 can split on both → 0 errors
        assert!(
            primal_depth2 < primal_depth1,
            "Depth-2 primal ({}) should be strictly less than depth-1 primal ({}). \
             If equal, backpropagation from depth-2 children is broken.",
            primal_depth2,
            primal_depth1
        );
    }

    /// Verify that child DataStates created by AndState have valid parent refs
    /// (not ParentRef::None), so backpropagation can propagate upward.
    #[test]
    fn and_state_children_have_parent_refs() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let mut config = make_config(x, y, 1, 2);
        config.max_depth = 2;
        config.lambda_ = 0.0;
        config.max_iter = 100;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);

        // Walk the tree and check that every non-root DataState has a parent
        for (i, ds) in &result.arena.data_states {
            let ds_id = DataStateId(i as u32);
            if ds_id == result.root_id {
                // Root should have ParentRef::None
                assert!(
                    matches!(ds.parent, ParentRef::None),
                    "Root should have ParentRef::None"
                );
            } else {
                // Non-root DataStates (still in slab) must have a parent
                assert!(
                    !matches!(ds.parent, ParentRef::None),
                    "Non-root DataState {} has ParentRef::None — \
                     backpropagation chain is broken",
                    i
                );
            }
        }
    }

    #[test]
    fn test_prepare_data_state_matches_create() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let config = make_config(x, y, 1, 2);
        let sample_indices: Vec<u32> = vec![0, 1, 2];
        let depth_budget: u8 = 2;

        let prepared = prepare_data_state(&config, None, &sample_indices, None, depth_budget);
        assert_eq!(prepared.n_samples_local, 3);
        assert_eq!(prepared.pred_class, 0);
        assert!((prepared.leaf_error - 0.0).abs() < 1e-12);
        assert!(prepared.dual == prepared.primal || prepared.depth_budget == 0);
        assert_eq!(prepared.depth_budget, depth_budget);

        let sample_indices_mixed: Vec<u32> = vec![0, 1, 3];
        let prepared2 =
            prepare_data_state(&config, None, &sample_indices_mixed, None, depth_budget);
        assert_eq!(prepared2.n_samples_local, 3);
        assert_eq!(prepared2.pred_class, 0);
        assert!((prepared2.leaf_error - 1.0).abs() < 1e-12);
        assert!(
            prepared2.dual != prepared2.primal
                || prepared2.depth_budget == 0
                || prepared2.leaf_error < 1e-12
        );
    }

    #[test]
    fn test_prepare_eager_data_computes_children() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let config = make_config(x, y, 1, 2);
        let sample_indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        let left_local: Vec<u32> = vec![0, 1, 2];
        let right_local: Vec<u32> = vec![3, 4, 5];

        let eager = prepare_eager_data(
            &config,
            None,
            &sample_indices,
            &left_local,
            &right_local,
            None,
            None,
            2,
        );

        assert_eq!(eager.left.n_samples_local, 3);
        assert_eq!(eager.left.pred_class, 0);
        assert!((eager.left.leaf_error).abs() < 1e-12);
        assert!(eager.left.dual == eager.left.primal || eager.left.depth_budget == 0);
        assert_eq!(eager.right.n_samples_local, 3);
        assert_eq!(eager.right.pred_class, 1);
        assert!((eager.right.leaf_error).abs() < 1e-12);
        assert!(eager.right.dual == eager.right.primal || eager.right.depth_budget == 0);
        assert!((eager.left.dual + eager.right.dual).abs() < 1e-12);
        assert!((eager.left.primal + eager.right.primal).abs() < 1e-12);
        assert_eq!(eager.left.depth_budget, 1);
        assert_eq!(eager.right.depth_budget, 1);
    }

    #[test]
    fn test_prepare_shape_fn_child_valid_split() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let config = make_config(x, y, 1, 2);
        let sample_indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        let bin_sample_indices: Vec<Vec<u32>> = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let result = prepare_shape_fn_child(
            &config,
            None,
            &sample_indices,
            2,
            0,
            "f0".to_string(),
            vec![3.5],
            &bin_sample_indices,
            None,
            vec![1, 0],
            vec![0],
            0,
        );
        assert!(result.is_some());
        let p = result.unwrap();
        assert_eq!(p.left_local_indices, vec![0, 1, 2]);
        assert_eq!(p.right_local_indices, vec![3, 4, 5]);
        assert!((p.sf_dual).abs() < 1e-12);
        assert!((p.sf_primal - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_prepare_shape_fn_child_invalid_split() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let config = make_config(x, y, 1, 2);
        let sample_indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        let bin_sample_indices: Vec<Vec<u32>> = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let result = prepare_shape_fn_child(
            &config,
            None,
            &sample_indices,
            2,
            0,
            "f0".to_string(),
            vec![3.5],
            &bin_sample_indices,
            None,
            vec![1, 1],
            vec![0],
            0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_prepare_shape_fn_child_with_eager() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let mut config = make_config(x, y, 1, 2);
        config.eager_expansion = true;
        let sample_indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        let bin_sample_indices: Vec<Vec<u32>> = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let result = prepare_shape_fn_child(
            &config,
            None,
            &sample_indices,
            2,
            0,
            "f0".to_string(),
            vec![3.5],
            &bin_sample_indices,
            None,
            vec![1, 0],
            vec![0],
            0,
        );
        let p = result.unwrap();
        assert!(p.eager_data.is_some());
        let eager = p.eager_data.unwrap();
        assert!((eager.left.dual + eager.right.dual).abs() < 1e-12);
        assert!((eager.left.primal + eager.right.primal).abs() < 1e-12);
        assert_eq!(eager.left.n_samples_local, 3);
        assert_eq!(eager.right.n_samples_local, 3);
    }

    #[test]
    fn eager_bin_summaries_preserve_child_state_and_order() {
        let mut config = make_config(vec![0.0; 6], vec![0, 1, 2, 0, 1, 2], 1, 3);
        config.eager_expansion = true;
        let samples = vec![5, 0, 4, 1, 3, 2];
        let bins = vec![vec![0, 3], vec![1, 4], vec![2, 5]];
        let distributions = vec![vec![0, 1, 1], vec![2, 0, 0], vec![0, 1, 1]];

        let summarized = prepare_shape_fn_child(
            &config,
            None,
            &samples,
            2,
            0,
            "f0".to_string(),
            vec![1.0, 2.0],
            &bins,
            Some(&distributions),
            vec![1, 0, 1],
            vec![0, 2],
            0,
        )
        .unwrap();
        let rescanned = prepare_shape_fn_child(
            &config,
            None,
            &samples,
            2,
            0,
            "f0".to_string(),
            vec![1.0, 2.0],
            &bins,
            None,
            vec![1, 0, 1],
            vec![0, 2],
            0,
        )
        .unwrap();

        assert_eq!(summarized.left_local_indices, vec![0, 3, 2, 5]);
        assert_eq!(summarized.right_local_indices, vec![1, 4]);
        let summarized = summarized.eager_data.unwrap();
        let rescanned = rescanned.eager_data.unwrap();
        assert_eq!(
            summarized.left.sample_indices,
            rescanned.left.sample_indices
        );
        assert_eq!(
            summarized.right.sample_indices,
            rescanned.right.sample_indices
        );
        assert_eq!(summarized.left.pred_class, rescanned.left.pred_class);
        assert_eq!(summarized.right.pred_class, rescanned.right.pred_class);
        assert_eq!(summarized.left.leaf_error, rescanned.left.leaf_error);
        assert_eq!(summarized.right.leaf_error, rescanned.right.leaf_error);
        assert_eq!(summarized.left.dual, rescanned.left.dual);
        assert_eq!(summarized.right.dual, rescanned.right.dual);
        assert_eq!(summarized.left.primal, rescanned.left.primal);
        assert_eq!(summarized.right.primal, rescanned.right.primal);
    }

    #[test]
    fn test_allocate_prepared_sf_child_eager() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];
        let mut config = make_config(x, y, 1, 2);
        config.eager_expansion = true;
        let sample_indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        let bin_sample_indices: Vec<Vec<u32>> = vec![vec![0, 1, 2], vec![3, 4, 5]];

        let prepared = prepare_shape_fn_child(
            &config,
            None,
            &sample_indices,
            2,
            0,
            "f0".to_string(),
            vec![3.5],
            &bin_sample_indices,
            None,
            vec![1, 0],
            vec![0],
            0,
        )
        .unwrap();

        let mut arena = Arena::new();
        let parent_ds_id = arena.alloc_data_state(
            vec![0, 1, 2, 3, 4, 5],
            3,
            ParentRef::None,
            6,
            0,
            3.0,
            0.0,
            3.0,
            3.0,
            Vec::new(),
        );
        let parent_ref = ParentRef::DataState(parent_ds_id);

        let mut ss = SearchState::new(42);
        let sf_id = allocate_prepared_sf_child(&mut arena, &config, &mut ss, prepared, parent_ref);
        assert!(sf_id.is_some());
        let sf_id = sf_id.unwrap();

        let sf = arena.sf(sf_id);
        assert!(!sf.dual_queue.is_empty());
        assert!(sf.primal_best_child.is_some());
        assert!((sf.primal - 0.0).abs() < 1e-12);
        assert!(sf.expanded);
        assert!(
            !sf.is_solved(),
            "eager allocation must defer solved-state changes to backpropagation"
        );
        assert_eq!(arena.data_states.len(), 3);
        assert_eq!(arena.shape_fn_states.len(), 1);
        assert_eq!(arena.and_states.len(), 1);
    }

    #[test]
    fn nested_eager_terminal_sf_backpropagates_after_parent_wiring() {
        // A higher-order SF can eagerly allocate a terminal SF child. Both of
        // that child's DataStates are leaves, but the child must not be marked
        // solved until its parent SF has inserted it in the queues.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<u8> = vec![0, 0, 1, 1, 0, 0];
        let mut config = make_config(x, y, 1, 2);
        config.eager_expansion = true;
        let mut arena = Arena::new();
        let root_id = arena.alloc_data_state(
            vec![0, 1, 2, 3, 4, 5],
            3,
            ParentRef::None,
            6,
            0,
            4.0,
            0.0,
            4.0,
            4.0,
            Vec::new(),
        );
        let parent_sf_id = arena.alloc_shape_fn_state(
            vec![0, 1, 2, 3, 4, 5],
            2,
            ParentRef::DataState(root_id),
            0,
            "f0".to_string(),
            vec![2.5, 4.5],
            FlatBins::from_bins(&[vec![0, 1], vec![2, 3], vec![4, 5]]),
            vec![1, 0, 0],
            vec![0],
            1,
            vec![0, 1],
            vec![2, 3, 4, 5],
            0.0,
            6.0,
            6.0,
        );
        arena.sf_mut(parent_sf_id).expanded = true;

        let prepared = prepare_shape_fn_child(
            &config,
            None,
            &[0, 1, 2, 3, 4, 5],
            2,
            0,
            "f0".to_string(),
            vec![2.5, 4.5],
            &[vec![0, 1], vec![2, 3], vec![4, 5]],
            None,
            vec![1, 0, 1],
            vec![0, 2],
            0,
        )
        .unwrap();
        let mut search_state = SearchState::new(42);
        let terminal_sf_id = allocate_prepared_sf_child(
            &mut arena,
            &config,
            &mut search_state,
            prepared,
            ParentRef::ShapeFnState(parent_sf_id),
        )
        .unwrap();
        let (terminal_dual, terminal_heuristic) = {
            let terminal_sf = arena.sf(terminal_sf_id);
            (terminal_sf.dual, terminal_sf.heuristic_value)
        };
        insort_dual_queue_sf(
            &mut arena.sf_mut(parent_sf_id).dual_queue,
            OrderedFloat(terminal_dual),
            OrderedFloat(terminal_heuristic),
            ShapeFnChildRef::ShapeFn(terminal_sf_id),
        );
        insort_heuristic_queue_sf(
            &mut arena.sf_mut(parent_sf_id).heuristic_queue,
            OrderedFloat(terminal_heuristic),
            ShapeFnChildRef::ShapeFn(terminal_sf_id),
        );

        assert!(!arena.sf(terminal_sf_id).is_solved());
        backprop_shape_fn_state(&mut arena, &config, &mut search_state, parent_sf_id, false);
        assert!(arena.sf(terminal_sf_id).is_solved());
        assert!(arena.sf(parent_sf_id).is_solved());
    }

    #[test]
    fn test_parallel_expansion_bounds_match_expected() {
        let mut x = Vec::new();
        let mut y: Vec<u8> = Vec::new();
        for v in &[1.0, 2.0, 3.0, 1.5, 2.5] {
            x.push(*v);
            x.push(5.0);
            y.push(0u8);
        }
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(1.0);
            y.push(1u8);
        }
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(9.0);
            y.push(2u8);
        }
        let mut config = make_config(x, y, 2, 3);
        config.max_depth = 2;
        config.lambda_ = 0.0;
        config.eager_expansion = true;
        config.max_iter = 5000;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);

        let root = result.arena.ds(result.root_id);
        assert!(
            root.primal <= 1.0,
            "Parallel expansion should find good tree, got primal={}",
            root.primal
        );
        assert!(
            root.dual <= root.primal + 1e-12,
            "Dual {} should not exceed primal {}",
            root.dual,
            root.primal
        );
        assert!(
            root.is_solved(),
            "Should be solved with lambda=0 and separable data"
        );
    }

    /// Regression test: anytime selection with many features and deeper trees.
    /// This is the minimal reproduction of the bug where an SF's AndState children
    /// are both solved leaves but the SF is not marked solved, causing selection
    /// to panic with "exhausted heuristic queue".
    ///
    /// The bug requires: anytime=true, heuristic=cart, depth>=4, multivariate
    /// discretizer, enough features to create many SF candidates where some
    /// SFs' splits lead to pure leaf children (immediately solved).
    #[test]
    fn anytime_cart_deep_multivariate_no_panic() {
        // 10 features, 5 classes, ~100 samples
        // This mimics the 'page' dataset structure that triggers the bug
        let n_features = 10;
        let n_samples_per_class = 20;
        let n_classes = 5u8;
        let mut x: Vec<f64> = Vec::new();
        let mut y: Vec<u8> = Vec::new();

        // Create data where each class is separable by a different feature
        for class in 0..n_classes {
            for i in 0..n_samples_per_class {
                for f in 0..n_features {
                    if f == class as usize {
                        // Discriminative feature for this class
                        x.push(10.0 * class as f64 + i as f64 * 0.1);
                    } else {
                        // Noise
                        x.push((i as f64 * 0.7 + f as f64 * 1.3) % 10.0);
                    }
                }
                y.push(class);
            }
        }

        let mut config = make_config(x, y, n_features as u16, n_classes);
        config.max_depth = 5;
        config.lambda_ = 0.0;
        config.alpha_ = 0.0;
        config.max_order = 1;
        config.anytime = true;
        config.heuristic = Heuristic::Cart;
        config.discretizer = Discretizer::Multivariate;
        config.n_candidates_list = vec![16, 8, 5, 5, 5];
        config.max_iter = 5000;
        config.time_limit = 30.0;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);

        // Should not panic (the bug causes a panic in selection).
        // Should either solve or hit max_iter without panicking.
        let root = result.arena.ds(result.root_id);
        assert!(
            root.primal <= root.leaf_error,
            "Primal should be <= leaf_error"
        );
    }

    /// Regression test: anytime selection must not panic when the search needs
    /// to descend past depth 1. With heuristic=Cart and anytime=true, the
    /// algorithm should find and prove optimal trees at depth >= 2 without
    /// hitting "exhausted heuristic queue" panics.
    #[test]
    fn anytime_cart_depth2_solves() {
        let mut x = Vec::new();
        let mut y: Vec<u8> = Vec::new();
        // Class 0: low feature 0
        for v in &[1.0, 2.0, 3.0, 1.5, 2.5] {
            x.push(*v);
            x.push(5.0);
            y.push(0u8);
        }
        // Class 1: high feature 0, low feature 1
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(1.0);
            y.push(1u8);
        }
        // Class 2: high feature 0, high feature 1
        for v in &[7.0, 8.0, 9.0, 7.5, 8.5] {
            x.push(*v);
            x.push(9.0);
            y.push(2u8);
        }
        let mut config = make_config(x, y, 2, 3);
        config.max_depth = 2;
        config.lambda_ = 0.0;
        config.anytime = true;
        config.heuristic = Heuristic::Cart;
        config.max_iter = 5000;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        assert!(
            result.arena.ds(result.root_id).is_solved(),
            "Should solve with anytime=true, heuristic=cart at depth 2"
        );
        assert!(
            result.arena.ds(result.root_id).primal < 1.0,
            "Should achieve near-0 errors at depth 2"
        );
    }

    /// Regression test: anytime selection with multivariate discretizer and
    /// deeper trees must not panic. This exercises the exact configuration
    /// used in optimality benchmarks.
    #[test]
    fn anytime_cart_depth3_multivariate() {
        let mut x = Vec::new();
        let mut y: Vec<u8> = Vec::new();
        // Create 4-class data that needs depth 3 to separate
        // Class 0: f0 low, f1 low
        for _ in 0..5 {
            x.push(1.0);
            x.push(1.0);
            y.push(0);
        }
        // Class 1: f0 low, f1 high
        for _ in 0..5 {
            x.push(2.0);
            x.push(9.0);
            y.push(1);
        }
        // Class 2: f0 high, f1 low
        for _ in 0..5 {
            x.push(9.0);
            x.push(2.0);
            y.push(2);
        }
        // Class 3: f0 high, f1 high
        for _ in 0..5 {
            x.push(8.0);
            x.push(8.0);
            y.push(3);
        }
        let mut config = make_config(x, y, 2, 4);
        config.max_depth = 3;
        config.lambda_ = 0.0;
        config.anytime = true;
        config.heuristic = Heuristic::Cart;
        config.discretizer = Discretizer::Multivariate;
        config.max_iter = 10000;
        let mut ss = SearchState::new(42);
        let result = run_ao_star(&config, &mut ss);
        assert!(
            result.arena.ds(result.root_id).is_solved(),
            "Should solve with anytime+cart+multivariate at depth 3"
        );
    }
}
