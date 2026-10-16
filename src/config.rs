use crate::impurity::Criterion;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::discretize::multivariate::MultivarDiscretizeResult;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Discretizer {
    Univariate,
    Multivariate,
}

impl std::str::FromStr for Discretizer {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "univariate" => Ok(Discretizer::Univariate),
            "multivariate" => Ok(Discretizer::Multivariate),
            _ => Err(format!(
                "Unknown discretizer: '{}'. Expected 'univariate' or 'multivariate'.",
                s
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Heuristic {
    Primal,
    Dual,
    Gini,
    Entropy,
    Cart,
}

/// Parse from string. Python resolves 'None'→'primal' and 'impurity'→'gini'/'entropy'
/// before passing to Rust. Rust only sees the 5 resolved variants.
impl std::str::FromStr for Heuristic {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "primal" => Ok(Heuristic::Primal),
            "dual" => Ok(Heuristic::Dual),
            "gini" => Ok(Heuristic::Gini),
            "entropy" => Ok(Heuristic::Entropy),
            "cart" => Ok(Heuristic::Cart),
            _ => Err(format!(
                "Unknown heuristic: '{}'. Expected 'primal', 'dual', 'gini', 'entropy', or 'cart'.",
                s
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectionScheme {
    Weighted,
    SampleWeighted,
    RoundRobin,
    Bfs,
    Branches,
}

impl std::str::FromStr for SelectionScheme {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "weighted" => Ok(SelectionScheme::Weighted),
            "sample_weighted" => Ok(SelectionScheme::SampleWeighted),
            "round_robin" => Ok(SelectionScheme::RoundRobin),
            "bfs" => Ok(SelectionScheme::Bfs),
            "branches" => Ok(SelectionScheme::Branches),
            _ => Err(format!(
                "Unknown selection_scheme: '{}'. Expected 'weighted', 'sample_weighted', 'round_robin', 'bfs', or 'branches'.", s
            )),
        }
    }
}

/// Immutable configuration shared across all states during search.
/// Penalties (lambda_, alpha_) are already scaled by n_samples by the Python wrapper.
pub struct Config {
    pub x: Vec<f64>, // flattened row-major feature matrix
    pub y: Vec<u8>,  // encoded labels (0..n_classes-1)
    pub n_samples: u32,
    pub bitset_words: usize, // ceil(n_samples / 64)
    pub n_features: u16,
    pub n_classes: u8,
    pub max_depth: u8,
    pub lambda_: f64, // scaled: self.lambda_ * n_samples
    pub alpha_: f64,  // scaled: self.alpha_ * n_samples
    pub max_order: u8,
    pub n_candidates_list: Vec<u8>,
    pub criterion: Criterion,
    pub discretizer: Discretizer,
    pub eager_expansion: bool,
    /// Enable subproblem caching (solved + unsolved lower-bound cache).
    pub use_cache: bool,
    pub heuristic: Heuristic, // resolved by Python before passing
    pub selection_scheme: SelectionScheme,
    pub anytime: bool,
    pub time_limit: f64,        // seconds
    pub max_iter: i64,          // -1 = unlimited
    pub memory_byte_limit: i64, // -1 = disabled; positive = byte limit
    pub verbose: bool,
    pub feature_names: Vec<String>,              // one per column
    pub feature_dict: Vec<(String, Vec<usize>)>, // (feature_name, [col_indices])
    /// Cumulative nanoseconds spent computing the CART heuristic, across all
    /// threads. Interior-mutable so the shared `&Config` can be updated from
    /// the parallel (Rayon) expansion phase. NOTE: with parallel expansion this
    /// is aggregate thread-time, so it can exceed wall-clock runtime.
    pub cart_heuristic_nanos: AtomicU64,
}

/// Cached dual bound and heuristic for one subproblem depth. Partial entries
/// intentionally carry no primal bound because they have no local tree witness.
/// When `solved` is true, `dual` is the proven optimal cost.
#[derive(Clone, Debug)]
pub struct DepthEntry {
    pub depth_budget: u8,
    pub dual: f64,
    pub heuristic: f64,
    pub solved: bool,
}

#[derive(Default)]
pub struct DatasetRecord {
    pub depths: Vec<DepthEntry>,
}

/// Membership records bucketed by sample count, with a tiny exact-depth list.
pub struct SubproblemCache {
    pub buckets: Vec<HashMap<Vec<u64>, DatasetRecord>>,
    pub hits: u64,
    pub stores: u64,
    pub partial_hits: u64,
}

impl Default for SubproblemCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SubproblemCache {
    pub fn new() -> Self {
        SubproblemCache {
            buckets: Vec::new(),
            hits: 0,
            stores: 0,
            partial_hits: 0,
        }
    }

    pub fn reset(&mut self, n_samples: usize, enabled: bool) {
        self.buckets = if enabled {
            (0..=n_samples).map(|_| HashMap::new()).collect()
        } else {
            Vec::new()
        };
        self.hits = 0;
        self.stores = 0;
        self.partial_hits = 0;
    }

    pub fn get(
        &self,
        sample_count: usize,
        sample_bitset: &[u64],
        depth_budget: u8,
    ) -> Option<&DepthEntry> {
        self.buckets
            .get(sample_count)?
            .get(sample_bitset)?
            .depths
            .iter()
            .find(|entry| entry.depth_budget == depth_budget)
    }

    pub fn store(
        &mut self,
        sample_count: usize,
        sample_bitset: &[u64],
        depth_budget: u8,
        dual: f64,
        heuristic: f64,
        solved: bool,
    ) -> bool {
        let Some(bucket) = self.buckets.get_mut(sample_count) else {
            return false;
        };
        let record = match bucket.get_mut(sample_bitset) {
            Some(record) => record,
            None => {
                bucket.insert(sample_bitset.to_vec(), DatasetRecord::default());
                bucket.get_mut(sample_bitset).unwrap()
            }
        };
        match record
            .depths
            .iter_mut()
            .find(|entry| entry.depth_budget == depth_budget)
        {
            None => record.depths.push(DepthEntry {
                depth_budget,
                dual,
                heuristic,
                solved,
            }),
            Some(entry) if solved => {
                entry.dual = dual;
                entry.heuristic = heuristic;
                entry.solved = true;
            }
            Some(entry) if !entry.solved => {
                let merged_dual = entry.dual.max(dual);
                let merged_heuristic = entry.heuristic.min(heuristic);
                if merged_dual == entry.dual && merged_heuristic == entry.heuristic {
                    return false;
                }
                entry.dual = merged_dual;
                entry.heuristic = merged_heuristic;
            }
            Some(_) => return false,
        }
        self.stores += 1;
        true
    }
}

/// Mutable search state, separated from Config so Config can be shared immutably.
/// Exact CART input identity. The sample order is deliberately retained: CART's
/// stable tie/NaN behavior is defined by the local input order, not membership.
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct CartCacheKey {
    pub sample_indices: Vec<u32>,
    pub depth: u8,
    pub criterion: Criterion,
    pub lambda_bits: u64,
    pub n_features: u16,
}

#[derive(Default)]
pub struct ExactCartCache {
    pub map: HashMap<CartCacheKey, f64>,
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
}

/// Exact multivariate discretizer identity. Ordered indices make cached local
/// bin indices valid without a membership-only remap.
#[derive(Hash, Eq, PartialEq)]
pub struct DiscretizerCacheKey {
    pub sample_indices: Vec<u32>,
    pub max_leaf_nodes: usize,
    pub criterion: Criterion,
    pub n_features: u16,
    pub n_classes: u8,
}

#[derive(Default)]
pub struct ExactDiscretizerCache {
    pub map: HashMap<DiscretizerCacheKey, MultivarDiscretizeResult>,
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
}

/// Shared bounds across both exact-result caches. A rejected insert computes
/// normally; exactness and cache-off semantics are unchanged.
pub const EXACT_CACHE_ENTRY_CAP: usize = 200_000;
pub const EXACT_CACHE_BYTE_BUDGET: usize = 2 * 1024 * 1024 * 1024;
const EXACT_CACHE_MAP_ENTRY_OVERHEAD: usize = 256;

#[derive(Default)]
pub struct ExactCacheAccounting {
    pub entries: usize,
    pub bytes: usize,
}

pub struct SearchState {
    pub rng: ChaCha8Rng,
    pub cache: SubproblemCache,
    /// Per-fit caches. Mutexes are reached only when `use_cache` is true.
    pub cart_cache: Mutex<ExactCartCache>,
    pub discretizer_cache: Mutex<ExactDiscretizerCache>,
    pub exact_cache_accounting: Mutex<ExactCacheAccounting>,
    pub exact_cache_cap_rejections: AtomicU64,
    pub discretizer_nanos: AtomicU64,
}

impl SearchState {
    pub fn new(seed: u64) -> Self {
        SearchState {
            rng: ChaCha8Rng::seed_from_u64(seed),
            cache: SubproblemCache::new(),
            cart_cache: Mutex::new(ExactCartCache::default()),
            discretizer_cache: Mutex::new(ExactDiscretizerCache::default()),
            exact_cache_accounting: Mutex::new(ExactCacheAccounting::default()),
            exact_cache_cap_rejections: AtomicU64::new(0),
            discretizer_nanos: AtomicU64::new(0),
        }
    }

    /// Reserve one combined exact-cache entry without eviction.
    pub fn reserve_exact_cache_entry(
        &self,
        bytes: usize,
        entry_cap: usize,
        byte_budget: usize,
    ) -> bool {
        let mut accounting = self.exact_cache_accounting.lock().unwrap();
        if accounting.entries >= entry_cap || bytes > byte_budget.saturating_sub(accounting.bytes) {
            self.exact_cache_cap_rejections
                .fetch_add(1, Ordering::Relaxed);
            return false;
        }
        accounting.entries += 1;
        accounting.bytes += bytes;
        true
    }

    pub fn cart_cache_entry_bytes(key: &CartCacheKey) -> usize {
        EXACT_CACHE_MAP_ENTRY_OVERHEAD
            + std::mem::size_of::<CartCacheKey>()
            + std::mem::size_of::<f64>()
            + key.sample_indices.capacity() * std::mem::size_of::<u32>()
    }

    pub fn discretizer_cache_entry_bytes(
        key: &DiscretizerCacheKey,
        result: &MultivarDiscretizeResult,
    ) -> usize {
        EXACT_CACHE_MAP_ENTRY_OVERHEAD
            + std::mem::size_of::<DiscretizerCacheKey>()
            + key.sample_indices.capacity() * std::mem::size_of::<u32>()
            + result.allocated_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    // ── Discretizer ──────────────────────────────────────────────────────

    #[test]
    fn discretizer_univariate() {
        assert_eq!(
            Discretizer::from_str("univariate").unwrap(),
            Discretizer::Univariate
        );
    }

    #[test]
    fn discretizer_multivariate() {
        assert_eq!(
            Discretizer::from_str("multivariate").unwrap(),
            Discretizer::Multivariate
        );
    }

    #[test]
    fn discretizer_invalid() {
        assert!(Discretizer::from_str("bivariate").is_err());
    }

    // ── Heuristic ────────────────────────────────────────────────────────

    #[test]
    fn heuristic_all_variants() {
        assert_eq!(Heuristic::from_str("primal").unwrap(), Heuristic::Primal);
        assert_eq!(Heuristic::from_str("dual").unwrap(), Heuristic::Dual);
        assert_eq!(Heuristic::from_str("gini").unwrap(), Heuristic::Gini);
        assert_eq!(Heuristic::from_str("entropy").unwrap(), Heuristic::Entropy);
        assert_eq!(Heuristic::from_str("cart").unwrap(), Heuristic::Cart);
    }

    #[test]
    fn heuristic_invalid() {
        assert!(Heuristic::from_str("impurity").is_err());
        assert!(Heuristic::from_str("None").is_err());
    }

    // ── SelectionScheme ──────────────────────────────────────────────────

    #[test]
    fn selection_scheme_all_variants() {
        assert_eq!(
            SelectionScheme::from_str("weighted").unwrap(),
            SelectionScheme::Weighted
        );
        assert_eq!(
            SelectionScheme::from_str("sample_weighted").unwrap(),
            SelectionScheme::SampleWeighted
        );
        assert_eq!(
            SelectionScheme::from_str("round_robin").unwrap(),
            SelectionScheme::RoundRobin
        );
        assert_eq!(
            SelectionScheme::from_str("bfs").unwrap(),
            SelectionScheme::Bfs
        );
        assert_eq!(
            SelectionScheme::from_str("branches").unwrap(),
            SelectionScheme::Branches
        );
    }

    #[test]
    fn selection_scheme_invalid() {
        assert!(SelectionScheme::from_str("random").is_err());
    }

    // ── SearchState ──────────────────────────────────────────────────────

    #[test]
    fn search_state_deterministic() {
        use rand::Rng;
        let mut s1 = SearchState::new(42);
        let mut s2 = SearchState::new(42);
        let v1: f64 = s1.rng.gen();
        let v2: f64 = s2.rng.gen();
        assert_eq!(v1, v2);
    }

    #[test]
    fn search_state_different_seeds() {
        use rand::Rng;
        let mut s1 = SearchState::new(42);
        let mut s2 = SearchState::new(99);
        let v1: f64 = s1.rng.gen();
        let v2: f64 = s2.rng.gen();
        assert_ne!(v1, v2);
    }

    #[test]
    fn exact_cache_caps_entries_and_bytes_across_cache_types() {
        let state = SearchState::new(42);
        // Model a CART entry then a discretizer entry sharing one reservation.
        assert!(state.reserve_exact_cache_entry(60, 2, 100));
        assert!(!state.reserve_exact_cache_entry(50, 2, 100));
        assert!(state.reserve_exact_cache_entry(40, 2, 100));
        assert!(!state.reserve_exact_cache_entry(1, 2, 100));
        let accounting = state.exact_cache_accounting.lock().unwrap();
        assert_eq!((accounting.entries, accounting.bytes), (2, 100));
        assert_eq!(state.exact_cache_cap_rejections.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn subproblem_cache_buckets_depths_and_merges_bounds() {
        let bits = [0b1011];
        let mut cache = SubproblemCache::new();
        cache.reset(8, true);

        assert!(cache.store(3, &bits, 2, 1.0, 5.0, false));
        assert!(cache.store(3, &bits, 4, 2.0, 4.0, false));
        assert_eq!(cache.buckets[3].len(), 1);
        assert_eq!(
            cache.buckets[3].get(bits.as_slice()).unwrap().depths.len(),
            2
        );
        assert!(cache.get(4, &bits, 2).is_none());

        assert!(cache.store(3, &bits, 2, 3.0, 6.0, false));
        assert!(cache.store(3, &bits, 2, 2.0, 2.0, false));
        let partial = cache.get(3, &bits, 2).unwrap();
        assert_eq!((partial.dual, partial.heuristic), (3.0, 2.0));

        assert!(cache.store(3, &bits, 2, 4.0, 1.0, true));
        assert!(!cache.store(3, &bits, 2, 9.0, 0.0, false));
        let solved = cache.get(3, &bits, 2).unwrap();
        assert!(solved.solved);
        assert_eq!((solved.dual, solved.heuristic), (4.0, 1.0));
    }

    #[test]
    fn subproblem_cache_supports_concurrent_borrowed_reads() {
        let bits = [0b1011];
        let mut cache = SubproblemCache::new();
        cache.reset(8, true);
        cache.store(3, &bits, 2, 1.0, 5.0, false);

        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| assert_eq!(cache.get(3, &bits, 2).unwrap().dual, 1.0));
            }
        });
    }

    #[test]
    fn disabled_subproblem_cache_allocates_no_buckets() {
        let mut cache = SubproblemCache::new();
        cache.hits = 1;
        cache.stores = 2;
        cache.partial_hits = 3;
        cache.reset(10_000, false);
        assert!(cache.buckets.is_empty());
        assert_eq!((cache.hits, cache.stores, cache.partial_hits), (0, 0, 0));
    }
}
