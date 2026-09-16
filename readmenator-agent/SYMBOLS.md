# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `Config` | class | `app.py:26` | `class Config` |
| `LocalComplexity` | class | `app.py:183` | `class LocalComplexity` |
| `MLPModel` | class | `app.py:88` | `class MLPModel(Module)` |
| `MatrixGrokker` | class | `app.py:355` | `class MatrixGrokker` |
| `MatrixMultiplicationDataset` | class | `app.py:58` | `class MatrixMultiplicationDataset(Dataset)` |
| `MetricsTracker` | class | `app.py:257` | `class MetricsTracker` |
| `Superposition` | class | `app.py:230` | `class Superposition` |
| `ThermalEngine` | class | `app.py:325` | `class ThermalEngine` |
| `__getitem__` | method | `app.py:84` | `def __getitem__(self, idx)` |
| `__init__` | method | `app.py:27` | `def __init__(self)` |
| `__init__` | method | `app.py:59` | `def __init__(self, matrix_size, num_samples, random_range, device)` |
| `__init__` | method | `app.py:89` | `def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation)` |
| `__init__` | method | `app.py:258` | `def __init__(self)` |
| `__init__` | method | `app.py:326` | `def __init__(self, config)` |
| `__init__` | method | `app.py:356` | `def __init__(self, config)` |
| `__len__` | method | `app.py:81` | `def __len__(self)` |
| `_compute_accuracy` | method | `app.py:474` | `def _compute_accuracy(self, predictions, targets, threshold)` |
| `_compute_products` | method | `app.py:78` | `def _compute_products(self, a, b)` |
| `_create_datasets` | method | `app.py:455` | `def _create_datasets(self)` |
| `_create_model` | method | `app.py:446` | `def _create_model(self)` |
| `_generate_matrices` | method | `app.py:73` | `def _generate_matrices(self, num_samples)` |
| `_save_checkpoint` | method | `app.py:608` | `def _save_checkpoint(self, model, optimizer, epoch, ips)` |
| `compute` | method | `app.py:185` | `def compute(activations, epsilon)` |
| `compute` | method | `app.py:232` | `def compute(weights, rank, epsilon)` |
| `compute_ips` | method | `app.py:287` | `def compute_ips(self)` |
| `compute_weight_decay` | method | `app.py:334` | `def compute_weight_decay(self, lc, sp, epoch)` |
| `end_epoch` | method | `app.py:280` | `def end_epoch(self)` |
| `expand_for_new_task` | method | `app.py:155` | `def expand_for_new_task(self, new_input_dim, new_output_dim, new_hidden_dim)` |
| `expand_weights` | method | `app.py:128` | `def expand_weights(self, new_hidden_dim)` |
| `find_latest_checkpoint` | method | `app.py:396` | `def find_latest_checkpoint(self)` |
| `forward` | method | `app.py:115` | `def forward(self, x)` |
| `from_model` | method | `app.py:205` | `def from_model(model, x)` |
| `from_model` | method | `app.py:252` | `def from_model(model, rank)` |
| `get_status` | method | `app.py:348` | `def get_status(self, lc, sp)` |
| `get_summary` | method | `app.py:305` | `def get_summary(self)` |
| `get_weight_matrix` | method | `app.py:121` | `def get_weight_matrix(self)` |
| `hook` | method | `app.py:208` | `def hook(module, input, output)` |
| `load_checkpoint` | method | `app.py:407` | `def load_checkpoint(self, checkpoint_path)` |
| `load_specific_checkpoint` | method | `app.py:770` | `def load_specific_checkpoint(checkpoint_path, config)` |
| `log_iteration` | method | `app.py:277` | `def log_iteration(self, iteration_time)` |
| `log_metrics` | method | `app.py:293` | `def log_metrics(self, train_loss, val_loss, train_acc, val_acc, lc, sp, lr, wd)` |
| `resume_from_latest_checkpoint` | method | `app.py:753` | `def resume_from_latest_checkpoint(config)` |
| `run_full_experiment` | method | `app.py:708` | `def run_full_experiment()` |
| `start_epoch` | method | `app.py:273` | `def start_epoch(self)` |
| `train` | method | `app.py:479` | `def train(self, resume_from_checkpoint)` |
| `zero_shot_transfer` | method | `app.py:640` | `def zero_shot_transfer(self, target_matrix_size)` |
