# Literati (NeurIPS 2026)

This repo is the code for the paper "Literati: Towards Anytime Optimal Shape Generalized Trees via AO*": https://arxiv.org/abs/2609.09299. 

- `src/` — the AO* search engine (Rust, exposed to Python via PyO3)
- `python/literati/` — `LiteratiClassifier`, a scikit-learn compatible classifier
- `data_utils.py` — `DataFactory_clf` dataset loader (OpenML / UCI, cached in `data/`)
- `plot_tree.py` — tree visualization (`plot_literati_tree`)
- `run.py` — fit on a dataset, print accuracy, save a tree plot
- `data/` — cached `rice` and `raisin` so they run offline; other datasets download on first use

## Install

Requires a Rust toolchain (`curl https://sh.rustup.rs -sSf | sh`) and Python >= 3.9.

```bash
python -m venv .venv && source .venv/bin/activate
pip install .
```

`pip install .` installs the Python dependencies and compiles the engine with maturin
(release mode). 

> On **Python 3.14+**, pyo3 0.23 needs the forward-compatibility flag: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install .`


For development (rebuild in place after editing `src/`): `pip install maturin && maturin develop --release`.

## Usage

`run.py` provides a simple example script demonstratinghow to use Literati. Edit the settings at the top of `run.py` (`DATASET`, `DEPTH`, `MAX_ORDER`, `TIME_LIMIT`, `LAMBDA_`, `ALPHA_`) to control the tree behavior. 

```python
from literati import LiteratiClassifier

clf = LiteratiClassifier(max_depth=3, max_order=1, time_limit=60)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.is_solved()  # True if the tree is provably optimal
```