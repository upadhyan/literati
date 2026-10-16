"""
LiteratiClassifier — sklearn-compatible wrapper around the native AO* engine.

Usage:
    from literati import LiteratiClassifier
    clf = LiteratiClassifier(max_depth=3, lambda_=0.005)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from literati._engine import LiteratiEngine


class LiteratiClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible Literati classifier backed by a native AO* engine."""

    _estimator_type = "classifier"

    def __init__(
        self,
        max_depth=4,
        lambda_=0.000,
        alpha_=0.000,
        max_order=3,
        time_limit=300,
        max_iter=-1,
        n_candidates_list=[16, 8],
        impurity="gini",
        verbose=False,
        anytime=True,
        eager_expansion=True,
        use_cache=True,
        discretizer="multivariate",
        heuristic="cart",
        random_state=42,
        selection_scheme="round_robin",
        memory_limit=0.999,
    ):
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.alpha_ = alpha_
        self.max_order = max_order
        self.time_limit = time_limit
        self.max_iter = max_iter
        self.n_candidates_list = n_candidates_list
        self.impurity = impurity
        self.verbose = verbose
        self.anytime = anytime
        self.eager_expansion = eager_expansion
        self.use_cache = use_cache
        self.discretizer = discretizer
        self.heuristic = heuristic
        self.random_state = random_state
        self.selection_scheme = selection_scheme
        self.memory_limit = memory_limit

        self._engine = None
        self.le = LabelEncoder()
        self.n_samples_ = 0
        self.n_classes_ = 0
        self.classes_ = None
        self._feature_dict = None
        self._X_train = None
        self.terminated_by_memory = False

    def fit(self, X, y, feature_dict=None):
        """Fit the Literati classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_dict : dict or None
            Maps feature_name -> list of column indices. If None, auto-generated.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            if feature_dict is None:
                feature_dict = {col: [i] for i, col in enumerate(X.columns)}
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.ascontiguousarray(X, dtype=np.float64)
        y_encoded = self.le.fit_transform(y).astype(np.int32)

        self.n_samples_ = X.shape[0]
        self.n_classes_ = len(self.le.classes_)
        self.classes_ = self.le.classes_

        if feature_dict is None:
            feature_dict = {f"f{i}": [i] for i in range(X.shape[1])}

        self._feature_dict = feature_dict

        # Resolve heuristic: 'None' → 'primal', 'impurity' → impurity measure
        resolved_heuristic = self.heuristic
        if resolved_heuristic == "None" or resolved_heuristic is None:
            resolved_heuristic = "primal"
        elif resolved_heuristic == "impurity":
            resolved_heuristic = self.impurity  # 'gini' or 'entropy'

        # Build feature_dict as list of (name, indices) for the engine — stringify keys
        fd_list = [(str(name), indices) for name, indices in feature_dict.items()]

        # Normalize n_candidates_list: int -> list, pad/truncate, clamp to max_order
        import warnings

        ncl = self.n_candidates_list
        if isinstance(ncl, int):
            ncl = [ncl] * self.max_depth
        else:
            ncl = list(ncl)
        if len(ncl) < self.max_depth:
            ncl.extend([self.max_order] * (self.max_depth - len(ncl)))
        elif len(ncl) > self.max_depth:
            ncl = ncl[: self.max_depth]
        clamped = False
        for i in range(len(ncl)):
            if ncl[i] < self.max_order:
                clamped = True
                ncl[i] = self.max_order
        if clamped and self.verbose:
            warnings.warn(
                f"Some n_candidates_list values were below max_order={self.max_order} and have been clamped."
            )

        # Memory guard: convert memory_limit to absolute byte limit
        import psutil

        if self.memory_limit <= 0:
            memory_byte_limit = -1
        elif self.memory_limit <= 1.0:
            memory_byte_limit = int(self.memory_limit * psutil.virtual_memory().total)
        else:
            memory_byte_limit = int(self.memory_limit * 1024**2)

        params = {
            "max_depth": self.max_depth,
            "lambda_": float(self.lambda_),
            "alpha_": float(self.alpha_),
            "max_order": self.max_order,
            "n_candidates_list": ncl,
            "criterion": self.impurity,
            "discretizer": self.discretizer,
            "eager_expansion": self.eager_expansion,
            "use_cache": bool(self.use_cache),
            "heuristic": resolved_heuristic,
            "selection_scheme": self.selection_scheme,
            "anytime": self.anytime,
            "time_limit": float(self.time_limit),
            "max_iter": int(self.max_iter),
            "verbose": self.verbose,
            "random_state": int(self.random_state),
            "feature_dict": fd_list,
            "memory_byte_limit": memory_byte_limit,
        }

        self._engine = LiteratiEngine()
        self._engine.fit(X, y_encoded, params)

        # Memory guard: extract termination flags
        import warnings

        self.terminated_by_memory = self._engine.terminated_by_memory()
        memory_guard_was_active = self._engine.memory_guard_active()

        if self.memory_limit > 0 and not memory_guard_was_active:
            warnings.warn(
                "Memory guard unavailable on this platform -- RAM checks disabled",
                RuntimeWarning,
            )

        if self.terminated_by_memory:
            limit_mb = memory_byte_limit / (1024**2)
            warnings.warn(
                f"Memory guard triggered (limit: {limit_mb:.1f} MB). "
                "Returning best solution found.",
                ResourceWarning,
            )

        # Store training data for visualization (plot_literati_tree needs feature values)
        self._X_train = X

        # Build optimization history DataFrame (matches Python LiteratiClassifier.track)
        self.track = pd.DataFrame(
            self._engine.get_track(),
            columns=["iteration", "dual", "primal", "gap", "time"],
        )

        # Timing: total wall-clock runtime and the portion spent in the CART
        # heuristic. cart_heuristic_time_ is aggregate across threads, so with
        # parallel expansion it can exceed total_time_.
        self.total_time_ = self._engine.total_time()
        self.cart_heuristic_time_ = self._engine.cart_heuristic_time()

        self.cache_hits_ = self._engine.cache_hits()
        self.cache_stores_ = self._engine.cache_stores()
        self.cache_partial_hits_ = self._engine.cache_partial_hits()
        self.cart_cache_hits_ = self._engine.cart_cache_hits()
        self.cart_cache_misses_ = self._engine.cart_cache_misses()
        self.cart_cache_stores_ = self._engine.cart_cache_stores()
        self.discretizer_cache_hits_ = self._engine.discretizer_cache_hits()
        self.discretizer_cache_misses_ = self._engine.discretizer_cache_misses()
        self.discretizer_cache_stores_ = self._engine.discretizer_cache_stores()
        self.exact_cache_cap_rejections_ = self._engine.exact_cache_cap_rejections()
        self.exact_cache_bytes_ = self._engine.exact_cache_bytes()
        self.discretizer_time_ = self._engine.discretizer_time()

        return self

    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (original label space).
        """
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.ascontiguousarray(X, dtype=np.float64)

        predictions = self._engine.predict(X)
        return self.le.inverse_transform(predictions)

    def is_solved(self):
        """Check if the search found a provably optimal solution."""
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._engine.is_solved()

    def get_depth(self):
        """Return the actual depth of the primal best tree."""
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._engine.get_depth()

    def get_n_leaves(self):
        """Return the number of leaves in the primal best tree."""
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._engine.get_n_leaves()

    def root_dual(self):
        """Return the root dual bound (error fraction)."""
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._engine.root_dual()

    def root_primal(self):
        """Return the root primal bound (error fraction)."""
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._engine.root_primal()

    def iterations_run(self):
        """Return the number of AO* iterations that were run."""
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._engine.iterations_run()

    def total_time(self):
        """Total wall-clock runtime of the AO* search loop, in seconds."""
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._engine.total_time()

    def cart_heuristic_time(self):
        """Cumulative time spent computing the CART heuristic, in seconds.

        Aggregate across threads: with parallel expansion (nodes with
        >= 2500 samples) this can exceed :meth:`total_time`.
        """
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._engine.cart_heuristic_time()

    def export_tree(self):
        """Export the primal best tree as a list of node dicts."""
        if self._engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._engine.export_tree()
