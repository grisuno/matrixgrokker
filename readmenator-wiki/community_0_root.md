# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `Config`, `LocalComplexity`, `MLPModel`, `MatrixGrokker`, `MatrixMultiplicationDataset`, `MetricsTracker`, `Superposition`, `ThermalEngine`. Core file: `app.py` (46 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: 09/01/2026 Licencia: AGPL v3  Descripción:  MatMul 2x2 Matrix Grok.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 46 | yes |
| `install.sh` | sh | utility | 0 | no |

## Key Symbols

- `Config` (class, `app.py:26`) `class Config`
- `__init__` (method, `app.py:27`) `def __init__(self)`
- `MatrixMultiplicationDataset` (class, `app.py:58`) `class MatrixMultiplicationDataset(Dataset)`
- `__init__` (method, `app.py:59`) `def __init__(self, matrix_size, num_samples, random_range, device)`
- `_generate_matrices` (method, `app.py:73`) `def _generate_matrices(self, num_samples)`
- `_compute_products` (method, `app.py:78`) `def _compute_products(self, a, b)`
- `__len__` (method, `app.py:81`) `def __len__(self)`
- `__getitem__` (method, `app.py:84`) `def __getitem__(self, idx)`
- `MLPModel` (class, `app.py:88`) `class MLPModel(Module)`
- `__init__` (method, `app.py:89`) `def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation)`
- `forward` (method, `app.py:115`) `def forward(self, x)`
- `get_weight_matrix` (method, `app.py:121`) `def get_weight_matrix(self)`
- `expand_weights` (method, `app.py:128`) `def expand_weights(self, new_hidden_dim)`
- `expand_for_new_task` (method, `app.py:155`) `def expand_for_new_task(self, new_input_dim, new_output_dim, new_hidden_dim)`
- `LocalComplexity` (class, `app.py:183`) `class LocalComplexity`
- `compute` (method, `app.py:185`) `def compute(activations, epsilon)`
- `from_model` (method, `app.py:205`) `def from_model(model, x)`
- `hook` (method, `app.py:208`) `def hook(module, input, output)`
- `Superposition` (class, `app.py:230`) `class Superposition`
- `compute` (method, `app.py:232`) `def compute(weights, rank, epsilon)`
- `from_model` (method, `app.py:252`) `def from_model(model, rank)`
- `MetricsTracker` (class, `app.py:257`) `class MetricsTracker`
- `__init__` (method, `app.py:258`) `def __init__(self)`
- `start_epoch` (method, `app.py:273`) `def start_epoch(self)`
- `log_iteration` (method, `app.py:277`) `def log_iteration(self, iteration_time)`
- `end_epoch` (method, `app.py:280`) `def end_epoch(self)`
- `compute_ips` (method, `app.py:287`) `def compute_ips(self)`
- `log_metrics` (method, `app.py:293`) `def log_metrics(self, train_loss, val_loss, train_acc, val_acc, lc, sp, lr, wd)`
- `get_summary` (method, `app.py:305`) `def get_summary(self)`
- `ThermalEngine` (class, `app.py:325`) `class ThermalEngine`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `install.sh`
