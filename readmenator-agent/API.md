# API

## app.py

### run_full_experiment `def run_full_experiment()`
- Defined: `app.py:708`

### resume_from_latest_checkpoint `def resume_from_latest_checkpoint(config)`
- Defined: `app.py:753`

### load_specific_checkpoint `def load_specific_checkpoint(checkpoint_path, config)`
- Defined: `app.py:770`

### __init__ `def __init__(self)`
- Defined: `app.py:27`

### __init__ `def __init__(self, matrix_size, num_samples, random_range, device)`
- Defined: `app.py:59`

### _generate_matrices `def _generate_matrices(self, num_samples)`
- Defined: `app.py:73`

### _compute_products `def _compute_products(self, a, b)`
- Defined: `app.py:78`

### __len__ `def __len__(self)`
- Defined: `app.py:81`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `app.py:84`

### __init__ `def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation)`
- Defined: `app.py:89`

### forward `def forward(self, x)`
- Defined: `app.py:115`

### get_weight_matrix `def get_weight_matrix(self)`
- Defined: `app.py:121`

### expand_weights `def expand_weights(self, new_hidden_dim)`
- Defined: `app.py:128`

### expand_for_new_task `def expand_for_new_task(self, new_input_dim, new_output_dim, new_hidden_dim)`
- Defined: `app.py:155`

### compute `def compute(activations, epsilon)`
- Defined: `app.py:185`

### from_model `def from_model(model, x)`
- Defined: `app.py:205`

### compute `def compute(weights, rank, epsilon)`
- Defined: `app.py:232`

### from_model `def from_model(model, rank)`
- Defined: `app.py:252`

### __init__ `def __init__(self)`
- Defined: `app.py:258`

### start_epoch `def start_epoch(self)`
- Defined: `app.py:273`

### log_iteration `def log_iteration(self, iteration_time)`
- Defined: `app.py:277`

### end_epoch `def end_epoch(self)`
- Defined: `app.py:280`

### compute_ips `def compute_ips(self)`
- Defined: `app.py:287`

### log_metrics `def log_metrics(self, train_loss, val_loss, train_acc, val_acc, lc, sp, lr, wd)`
- Defined: `app.py:293`

### get_summary `def get_summary(self)`
- Defined: `app.py:305`

### __init__ `def __init__(self, config)`
- Defined: `app.py:326`

### compute_weight_decay `def compute_weight_decay(self, lc, sp, epoch)`
- Defined: `app.py:334`

### get_status `def get_status(self, lc, sp)`
- Defined: `app.py:348`

### __init__ `def __init__(self, config)`
- Defined: `app.py:356`

### find_latest_checkpoint `def find_latest_checkpoint(self)`
- Defined: `app.py:396`

### load_checkpoint `def load_checkpoint(self, checkpoint_path)`
- Defined: `app.py:407`

### _create_model `def _create_model(self)`
- Defined: `app.py:446`

### _create_datasets `def _create_datasets(self)`
- Defined: `app.py:455`

### _compute_accuracy `def _compute_accuracy(self, predictions, targets, threshold)`
- Defined: `app.py:474`

### train `def train(self, resume_from_checkpoint)`
- Defined: `app.py:479`

### _save_checkpoint `def _save_checkpoint(self, model, optimizer, epoch, ips)`
- Defined: `app.py:608`

### zero_shot_transfer `def zero_shot_transfer(self, target_matrix_size)`
- Defined: `app.py:640`

### hook `def hook(module, input, output)`
- Defined: `app.py:208`
