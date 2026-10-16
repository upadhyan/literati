// Note: jemalloc cannot be used as #[global_allocator] in a cdylib (PyO3 extension)
// due to static TLS allocation failures on Linux. The slab-based arena with compact
// types provides the primary memory savings instead.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub mod config;
pub mod discretize;
pub mod engine;
pub mod heuristic;
pub mod impurity;
pub mod predict;
mod presorted;
pub mod states;
pub mod tree_export;

use config::{Config, Discretizer, Heuristic, SearchState, SelectionScheme};
use engine::{run_ao_star, SearchResult};
use impurity::Criterion;
use predict::batch_predict;
use tree_export::{export_tree_as_dicts, ExportValue};

type TrackTuple = (usize, f64, f64, f64, f64);

#[pyclass]
struct LiteratiEngine {
    result: Option<SearchResult>,
    config: Option<Config>,
    search_state: Option<SearchState>,
}

#[pymethods]
impl LiteratiEngine {
    #[new]
    fn new() -> Self {
        LiteratiEngine {
            result: None,
            config: None,
            search_state: None,
        }
    }

    /// Fit the AO* search on the provided data.
    ///
    /// Args:
    ///     x: 2D numpy array (n_samples, n_features), float64
    ///     y: 1D numpy array (n_samples,), int32 (encoded labels 0..n_classes-1)
    ///     params: dict with all hyperparameters
    #[pyo3(signature = (x, y, params))]
    fn fit(
        &mut self,
        _py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<i32>,
        params: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let x_arr = x.as_array();
        let y_arr = y.as_array();
        let n_samples_raw = y_arr.len();
        let n_features_raw = x_arr.shape()[1];

        // Boundary validation
        if n_samples_raw > u32::MAX as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_samples ({}) exceeds u32::MAX ({})",
                n_samples_raw,
                u32::MAX
            )));
        }
        if n_features_raw > u16::MAX as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_features ({}) exceeds u16::MAX ({})",
                n_features_raw,
                u16::MAX
            )));
        }

        // Flatten X to row-major Vec
        let x_flat: Vec<f64> = x_arr.iter().cloned().collect();
        // Convert y from i32 to u8
        let y_vec: Vec<u8> = y_arr.iter().map(|&v| v as u8).collect();

        // Determine n_classes
        let n_classes_raw = y_vec.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;
        if n_classes_raw > 255 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_classes ({}) exceeds 255",
                n_classes_raw
            )));
        }

        let n_samples = n_samples_raw as u32;
        let n_features = n_features_raw as u16;
        let n_classes = n_classes_raw as u8;

        // Parse params
        let max_depth: u8 = params.get_item("max_depth")?.unwrap().extract()?;
        let lambda_: f64 = params.get_item("lambda_")?.unwrap().extract()?;
        let alpha_: f64 = params.get_item("alpha_")?.unwrap().extract()?;
        let max_order: u8 = params.get_item("max_order")?.unwrap().extract()?;
        let n_candidates_list_raw: Vec<u8> =
            params.get_item("n_candidates_list")?.unwrap().extract()?;
        let criterion_str: String = params.get_item("criterion")?.unwrap().extract()?;
        let discretizer_str: String = params.get_item("discretizer")?.unwrap().extract()?;
        let eager_expansion: bool = params.get_item("eager_expansion")?.unwrap().extract()?;
        let use_cache: bool = params
            .get_item("use_cache")?
            .map(|v| v.extract::<bool>())
            .transpose()?
            .unwrap_or(true);
        let heuristic_str: String = params.get_item("heuristic")?.unwrap().extract()?;
        let selection_scheme_str: String =
            params.get_item("selection_scheme")?.unwrap().extract()?;
        let anytime: bool = params.get_item("anytime")?.unwrap().extract()?;
        let time_limit: f64 = params.get_item("time_limit")?.unwrap().extract()?;
        let max_iter: i64 = params.get_item("max_iter")?.unwrap().extract()?;
        let verbose: bool = params.get_item("verbose")?.unwrap().extract()?;
        let random_state: u64 = params.get_item("random_state")?.unwrap().extract()?;
        let memory_byte_limit: i64 = params.get_item("memory_byte_limit")?.unwrap().extract()?;
        // Parse feature_dict: list of (name, [col_indices])
        let feature_dict_py = params.get_item("feature_dict")?.unwrap();
        let feature_dict_list: &Bound<'_, PyList> = feature_dict_py.downcast()?;
        let mut feature_dict: Vec<(String, Vec<usize>)> = Vec::new();
        let mut feature_names: Vec<String> = Vec::new();
        for item in feature_dict_list.iter() {
            let tuple = item;
            let name: String = tuple.get_item(0)?.extract()?;
            let indices: Vec<usize> = tuple.get_item(1)?.extract()?;
            feature_names.push(name.clone());
            feature_dict.push((name, indices));
        }

        let criterion: Criterion = criterion_str
            .parse()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let discretizer: Discretizer = discretizer_str
            .parse()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let heuristic: Heuristic = heuristic_str
            .parse()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let selection_scheme: SelectionScheme = selection_scheme_str
            .parse()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Scale penalties
        let lambda_scaled = lambda_ * n_samples_raw as f64;
        let alpha_scaled = alpha_ * n_samples_raw as f64;

        let config = Config {
            x: x_flat,
            y: y_vec,
            n_samples,
            bitset_words: (n_samples as usize).div_ceil(64),
            n_features,
            n_classes,
            max_depth,
            lambda_: lambda_scaled,
            alpha_: alpha_scaled,
            max_order,
            n_candidates_list: n_candidates_list_raw,
            criterion,
            discretizer,
            eager_expansion,
            use_cache,
            heuristic,
            selection_scheme,
            anytime,
            time_limit,
            max_iter,
            memory_byte_limit,
            verbose,
            feature_names,
            feature_dict,
            cart_heuristic_nanos: std::sync::atomic::AtomicU64::new(0),
        };

        let mut search_state = SearchState::new(random_state);
        let result = run_ao_star(&config, &mut search_state);

        self.config = Some(config);
        self.result = Some(result);
        self.search_state = Some(search_state);

        Ok(())
    }

    /// Batch predict on new data.
    ///
    /// Args:
    ///     x: 2D numpy array (n_samples, n_features), float64
    ///
    /// Returns:
    ///     1D numpy array of predicted classes (encoded labels)
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        let config = self.config.as_ref().unwrap();

        let x_arr = x.as_array();
        let n_samples = x_arr.shape()[0];
        let n_features = x_arr.shape()[1];
        let x_flat: Vec<f64> = x_arr.iter().cloned().collect();

        let predictions = batch_predict(
            &result.arena,
            config,
            &x_flat,
            n_samples,
            n_features,
            result.root_id,
        );

        Ok(PyArray1::from_vec(py, predictions))
    }

    /// Check if the root state is solved (provably optimal).
    fn is_solved(&self) -> PyResult<bool> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result.arena.ds(result.root_id).is_solved())
    }

    /// Get the number of AO* iterations run.
    fn iterations_run(&self) -> PyResult<usize> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result.iterations_run)
    }

    /// Get root dual bound (scaled back to fraction).
    fn root_dual(&self) -> PyResult<f64> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        let config = self.config.as_ref().unwrap();
        Ok(result.arena.ds(result.root_id).dual / config.n_samples as f64)
    }

    /// Get root primal bound (scaled back to fraction).
    fn root_primal(&self) -> PyResult<f64> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        let config = self.config.as_ref().unwrap();
        Ok(result.arena.ds(result.root_id).primal / config.n_samples as f64)
    }

    /// Export the primal best tree as a list of node dicts for visualization.
    fn export_tree<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        let config = self.config.as_ref().unwrap();

        let dicts = export_tree_as_dicts(&result.arena, config, result.root_id);
        let py_list = PyList::empty(py);

        for d in &dicts {
            let py_dict = PyDict::new(py);
            for (key, val) in d {
                match val {
                    ExportValue::Int(v) => py_dict.set_item(key, v)?,
                    ExportValue::Float(v) => py_dict.set_item(key, v)?,
                    ExportValue::Str(v) => py_dict.set_item(key, v)?,
                    ExportValue::Bool(v) => py_dict.set_item(key, v)?,
                    ExportValue::None => py_dict.set_item(key, py.None())?,
                    ExportValue::IntList(v) => py_dict.set_item(key, v)?,
                    ExportValue::FloatList(v) => py_dict.set_item(key, v)?,
                }
            }
            py_list.append(py_dict)?;
        }

        Ok(py_list)
    }

    /// Get the actual depth of the primal best tree.
    fn get_depth(&self) -> PyResult<usize> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        let config = self.config.as_ref().unwrap();
        let nodes = tree_export::export_tree(&result.arena, config, result.root_id);
        Ok(nodes.iter().map(|n| n.depth).max().unwrap_or(0))
    }

    /// Get the number of leaves in the primal best tree.
    fn get_n_leaves(&self) -> PyResult<usize> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        let config = self.config.as_ref().unwrap();
        let nodes = tree_export::export_tree(&result.arena, config, result.root_id);
        Ok(nodes.iter().filter(|n| n.is_leaf).count())
    }

    /// Total wall-clock runtime of the AO* search loop, in seconds.
    fn total_time(&self) -> PyResult<f64> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result.total_time)
    }

    /// Cumulative time spent computing the CART heuristic, in seconds.
    /// Aggregate across threads: with parallel expansion this can exceed
    /// `total_time`.
    fn cart_heuristic_time(&self) -> PyResult<f64> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result.cart_heuristic_time)
    }

    /// Number of subproblem cache hits during the search.
    fn cache_hits(&self) -> PyResult<u64> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result.cache_hits)
    }

    /// Number of subproblem cache stores during the search.
    fn cache_stores(&self) -> PyResult<u64> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result.cache_stores)
    }

    /// Number of partial (unsolved) cache hits that tightened bounds.
    fn cache_partial_hits(&self) -> PyResult<u64> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result.cache_partial_hits)
    }

    fn cart_cache_hits(&self) -> PyResult<u64> {
        Ok(self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?
            .cart_cache_hits)
    }

    fn cart_cache_misses(&self) -> PyResult<u64> {
        Ok(self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?
            .cart_cache_misses)
    }

    fn cart_cache_stores(&self) -> PyResult<u64> {
        Ok(self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?
            .cart_cache_stores)
    }

    fn discretizer_cache_hits(&self) -> PyResult<u64> {
        Ok(self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?
            .discretizer_cache_hits)
    }

    fn discretizer_cache_misses(&self) -> PyResult<u64> {
        Ok(self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?
            .discretizer_cache_misses)
    }

    fn discretizer_cache_stores(&self) -> PyResult<u64> {
        Ok(self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?
            .discretizer_cache_stores)
    }

    fn discretizer_time(&self) -> PyResult<f64> {
        Ok(self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?
            .discretizer_time)
    }

    fn exact_cache_cap_rejections(&self) -> PyResult<u64> {
        Ok(self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?
            .exact_cache_cap_rejections)
    }

    fn exact_cache_bytes(&self) -> PyResult<usize> {
        Ok(self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?
            .exact_cache_bytes)
    }

    /// Get the optimization history as a list of (iteration, dual, primal, gap, time) tuples.
    fn get_track(&self) -> PyResult<Vec<TrackTuple>> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result
            .track
            .iter()
            .map(|e| (e.iteration, e.dual, e.primal, e.gap, e.time))
            .collect())
    }

    /// Check if the search was terminated by the memory guard.
    fn terminated_by_memory(&self) -> PyResult<bool> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result.terminated_by_memory)
    }

    /// Check if the memory guard was active (platform supported /proc/self/status).
    fn memory_guard_active(&self) -> PyResult<bool> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(result.memory_guard_active)
    }
}

/// Expose the Rust multivariate CART discretizer directly to Python.
///
/// Args:
///     x: 2D numpy array (n_samples, n_features), float64
///     y: 1D numpy array (n_samples,), int32 (encoded labels 0..n_classes-1)
///     max_leaf_nodes: maximum number of leaves in the CART tree
///     criterion: "gini" or "entropy"
///
/// Returns:
///     dict mapping feature_index (int) -> list of thresholds (list of float)
#[pyfunction]
#[pyo3(signature = (x, y, max_leaf_nodes, criterion))]
fn multivariate_cart<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<i32>,
    max_leaf_nodes: usize,
    criterion: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    let n_features = x_arr.shape()[1];

    // Flatten X to row-major Vec
    let x_flat: Vec<f64> = x_arr.iter().cloned().collect();
    // Convert y from i32 to u8
    let y_vec: Vec<u8> = y_arr.iter().map(|&v| v as u8).collect();

    // Determine n_classes
    let n_classes = y_vec.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

    let crit: Criterion = criterion
        .parse()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let result = discretize::multivariate::fit_multivariate_cart(
        &x_flat,
        &y_vec,
        n_features,
        n_classes,
        max_leaf_nodes,
        crit,
    );

    // Convert BTreeMap<usize, Vec<f64>> -> Python dict {int: list[float]}
    let py_dict = PyDict::new(py);
    for (feat_idx, thresholds) in &result.feature_thresholds {
        let py_list = PyList::new(py, thresholds)?;
        py_dict.set_item(feat_idx, py_list)?;
    }

    Ok(py_dict)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LiteratiEngine>()?;
    m.add_function(wrap_pyfunction!(multivariate_cart, m)?)?;
    Ok(())
}
