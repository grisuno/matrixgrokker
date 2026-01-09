#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: 09/01/2026
Licencia: AGPL v3

Descripción:  MatMul 2x2 Matrix Grokker
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
import time
from datetime import datetime
from pathlib import Path


class Config:
    def __init__(self):
        self.SEED = 42
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.MATRIX_SIZE = 2
        self.HIDDEN_DIM = 256
        self.NUM_LAYERS = 3
        self.ACTIVATION = "relu"
        
        self.TRAIN_EPOCHS = 1000
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.001
        
        self.WEIGHT_DECAY = 0.01
        self.LC_THRESHOLD = 0.5
        self.SUPERPOSITION_THRESHOLD = 0.5
        
        self.CHECKPOINT_INTERVAL_MINUTES = 5
        self.METRICS_COMPUTE_INTERVAL = 200
        
        self.EXPANSION_SIZES = [4, 8]
        
        self.NUM_SAMPLES = 10000
        self.VAL_RATIO = 0.2
        
        self.RANDOM_RANGE = (-1.0, 1.0)
        
        self.RESUME_FROM_CHECKPOINT = True
        self.FORCE_START_FRESH = False


class MatrixMultiplicationDataset(Dataset):
    def __init__(self, matrix_size: int, num_samples: int, random_range: Tuple[float, float], device: torch.device):
        self.matrix_size = matrix_size
        self.num_samples = num_samples
        self.random_range = random_range
        self.device = device
        
        self.inputs_a = self._generate_matrices(num_samples)
        self.inputs_b = self._generate_matrices(num_samples)
        self.outputs = self._compute_products(self.inputs_a, self.inputs_b)
        
        flat_a = self.inputs_a.reshape(num_samples, -1)
        flat_b = self.inputs_b.reshape(num_samples, -1)
        self.inputs = torch.cat([flat_a, flat_b], dim=1)
        
    def _generate_matrices(self, num_samples: int) -> torch.Tensor:
        low, high = self.random_range
        matrices = torch.rand(num_samples, self.matrix_size, self.matrix_size) * (high - low) + low
        return matrices
    
    def _compute_products(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx].to(self.device), self.outputs[idx].reshape(-1).to(self.device)


class MLPModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, activation: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_name = activation
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
    
    def get_weight_matrix(self) -> torch.Tensor:
        weights = []
        for layer in self.layers:
            if layer.weight.requires_grad:
                weights.append(layer.weight.data.flatten())
        return torch.cat(weights)
    
    def expand_weights(self, new_hidden_dim: int) -> 'MLPModel':
        new_model = MLPModel(
            self.input_dim,
            self.output_dim,
            new_hidden_dim,
            self.num_layers,
            self.activation_name
        )
        
        with torch.no_grad():
            for old_layer, new_layer in zip(self.layers, new_model.layers):
                old_w = old_layer.weight
                old_b = old_layer.bias
                new_w = new_layer.weight
                new_b = new_layer.bias
                
                old_out_dim, old_in_dim = old_w.shape
                new_out_dim, new_in_dim = new_w.shape
                
                min_out = min(old_out_dim, new_out_dim)
                min_in = min(old_in_dim, new_in_dim)
                
                new_w[:min_out, :min_in] = old_w[:min_out, :min_in]
                new_b[:min_out] = old_b[:min_out]
        
        return new_model
    
    def expand_for_new_task(self, new_input_dim: int, new_output_dim: int, new_hidden_dim: int) -> 'MLPModel':
        new_model = MLPModel(
            new_input_dim,
            new_output_dim,
            new_hidden_dim,
            self.num_layers,
            self.activation_name
        )
        
        with torch.no_grad():
            for i, (old_layer, new_layer) in enumerate(zip(self.layers, new_model.layers)):
                old_w = old_layer.weight
                old_b = old_layer.bias
                new_w = new_layer.weight
                new_b = new_layer.bias
                
                old_out_dim, old_in_dim = old_w.shape
                new_out_dim, new_in_dim = new_w.shape
                
                min_out = min(old_out_dim, new_out_dim)
                min_in = min(old_in_dim, new_in_dim)
                
                new_w[:min_out, :min_in] = old_w[:min_out, :min_in]
                new_b[:min_out] = old_b[:min_out]
        
        return new_model


class LocalComplexity:
    @staticmethod
    def compute(activations: torch.Tensor, epsilon: float = 1e-8) -> float:
        if activations.numel() == 0:
            return 0.0
        
        activations = activations.float()
        
        normalized = activations / (activations.norm(dim=-1, keepdim=True) + epsilon)
        
        pairwise_dot = torch.mm(normalized, normalized.T)
        
        mask = torch.ones_like(pairwise_dot)
        mask.fill_diagonal_(0)
        
        off_diagonal_mean = (pairwise_dot * mask).sum() / (mask.sum() + epsilon)
        
        complexity = 1.0 - abs(off_diagonal_mean.item())
        
        return float(np.clip(complexity, 0.0, 1.0))
    
    @staticmethod
    def from_model(model: nn.Module, x: torch.Tensor) -> float:
        activations = []
        
        def hook(module, input, output):
            activations.append(input[0].detach())
        
        hooks = []
        for layer in model.layers[:-1]:
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(hook))
        
        with torch.no_grad():
            _ = model(x)
        
        for hook in hooks:
            hook.remove()
        
        if not activations:
            return 0.0
        
        combined = torch.cat(activations, dim=1) if len(activations) > 1 else activations[0]
        
        return LocalComplexity.compute(combined)


class Superposition:
    @staticmethod
    def compute(weights: torch.Tensor, rank: int, epsilon: float = 1e-8) -> float:
        if weights.numel() < rank * 2:
            return 1.0
        
        weights = weights.float()
        
        U, S, V = torch.svd(weights.reshape(weights.shape[0], -1))
        
        if len(S) < rank:
            effective_rank = len(S)
        else:
            effective_rank = float(rank)
        
        explained = (S[:rank].sum() / (S.sum() + epsilon)).item()
        
        superposition = 1.0 - explained
        
        return float(np.clip(superposition, 0.0, 1.0))
    
    @staticmethod
    def from_model(model: nn.Module, rank: int = 4) -> float:
        weights = model.get_weight_matrix()
        return Superposition.compute(weights, rank)


class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.local_complexities = []
        self.superpositions = []
        self.learning_rates = []
        self.weight_decays = []
        self.iterations_per_second = []
        self.step_times = []
        
        self.epoch_start_time = None
        self.iteration_times = []
        
    def start_epoch(self):
        self.epoch_start_time = time.time()
        self.iteration_times = []
    
    def log_iteration(self, iteration_time: float):
        self.iteration_times.append(iteration_time)
    
    def end_epoch(self) -> float:
        if self.epoch_start_time and self.iteration_times:
            epoch_time = time.time() - self.epoch_start_time
            self.step_times.append(epoch_time)
            return epoch_time
        return 0.0
    
    def compute_ips(self) -> float:
        if len(self.iteration_times) < 10:
            return 0.0
        recent_times = self.iteration_times[-100:]
        return 100.0 / (sum(recent_times) + 1e-8)
    
    def log_metrics(self, train_loss: float, val_loss: float, train_acc: float, val_acc: float,
                    lc: float, sp: float, lr: float, wd: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.local_complexities.append(lc)
        self.superpositions.append(sp)
        self.learning_rates.append(lr)
        self.weight_decays.append(wd)
        self.iterations_per_second.append(self.compute_ips())
    
    def get_summary(self) -> Dict:
        ips_history = self.iterations_per_second
        if isinstance(ips_history, list) and len(ips_history) >= 10:
            avg_ips = float(np.mean(ips_history[-100:]))
        elif isinstance(ips_history, (int, float)):
            avg_ips = float(ips_history)
        else:
            avg_ips = None
        
        return {
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "final_train_acc": self.train_accuracies[-1] if self.train_accuracies else None,
            "final_val_acc": self.val_accuracies[-1] if self.val_accuracies else None,
            "final_lc": self.local_complexities[-1] if self.local_complexities else None,
            "final_sp": self.superpositions[-1] if self.superpositions else None,
            "avg_ips": avg_ips
        }


class ThermalEngine:
    def __init__(self, config: Config):
        self.config = config
        self.lc_target = 1.0
        self.sp_target = 0.0
        self.lc_weight = 0.3
        self.sp_weight = 0.3
        self.base_weight_decay = config.WEIGHT_DECAY
    
    def compute_weight_decay(self, lc: float, sp: float, epoch: int) -> float:
        lc_deviation = abs(self.lc_target - lc)
        sp_deviation = abs(self.sp_target - sp)
        
        combined_deviation = self.lc_weight * lc_deviation + self.sp_weight * sp_deviation
        
        thermal_factor = 1.0 + combined_deviation * 2.0
        
        current_wd = self.base_weight_decay * thermal_factor
        
        current_wd = float(np.clip(current_wd, self.base_weight_decay * 0.1, self.base_weight_decay * 5.0))
        
        return current_wd
    
    def get_status(self, lc: float, sp: float) -> str:
        lc_progress = (lc / self.lc_target) * 100
        sp_progress = (1.0 - sp / self.sp_target) * 100 if self.sp_target > 0 else 0
        avg_progress = (lc_progress + sp_progress) / 2
        return f"Thermal Progress: {avg_progress:.1f}%"


class MatrixGrokker:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        
        self.input_dim = config.MATRIX_SIZE * config.MATRIX_SIZE * 2
        self.output_dim = config.MATRIX_SIZE * config.MATRIX_SIZE
        
        self.model = self._create_model()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.criterion = nn.MSELoss()
        
        self.train_dataset, self.val_dataset = self._create_datasets()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )
        
        self.metrics_tracker = MetricsTracker()
        self.thermal_engine = ThermalEngine(config)
        
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.last_checkpoint_time = time.time()
        self.global_step = 0
        self.epoch = 0
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        if not self.checkpoint_dir.exists():
            return None
        
        checkpoints = list(self.checkpoint_dir.glob(f"matrix_{self.config.MATRIX_SIZE}x{self.config.MATRIX_SIZE}_epoch_*.pt"))
        if not checkpoints:
            return None
        
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest_checkpoint
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Tuple[int, Dict]:
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is None or not checkpoint_path.exists():
            return 0, {}
        
        print(f"\nLoading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', -1) + 1
        
        saved_metrics = checkpoint.get('metrics', {})
        saved_config = checkpoint.get('config', {})
        
        if 'timestamp' in checkpoint:
            print(f"  Checkpoint from: {checkpoint['timestamp']}")
        print(f"  Resuming from epoch: {self.epoch + 1}")
        print(f"  Global step: {self.global_step}")
        
        if 'train_losses' in checkpoint:
            self.metrics_tracker.train_losses = checkpoint.get('train_losses', [])
            self.metrics_tracker.val_losses = checkpoint.get('val_losses', [])
            self.metrics_tracker.train_accuracies = checkpoint.get('train_accuracies', [])
            self.metrics_tracker.val_accuracies = checkpoint.get('val_accuracies', [])
            self.metrics_tracker.local_complexities = checkpoint.get('local_complexities', [])
            self.metrics_tracker.superpositions = checkpoint.get('superpositions', [])
            self.metrics_tracker.learning_rates = checkpoint.get('learning_rates', [])
            self.metrics_tracker.weight_decays = checkpoint.get('weight_decays', [])
            self.metrics_tracker.iterations_per_second = checkpoint.get('iterations_per_second_history', [])
            print(f"  Loaded {len(self.metrics_tracker.train_losses)} epochs of metrics history")
        
        return self.epoch, saved_metrics
    
    def _create_model(self) -> MLPModel:
        return MLPModel(
            self.input_dim,
            self.output_dim,
            self.config.HIDDEN_DIM,
            self.config.NUM_LAYERS,
            self.config.ACTIVATION
        ).to(self.device)
    
    def _create_datasets(self) -> Tuple[MatrixMultiplicationDataset, MatrixMultiplicationDataset]:
        full_dataset = MatrixMultiplicationDataset(
            self.config.MATRIX_SIZE,
            self.config.NUM_SAMPLES,
            self.config.RANDOM_RANGE,
            self.device
        )
        
        val_size = int(len(full_dataset) * self.config.VAL_RATIO)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.SEED)
        )
        
        return train_dataset, val_dataset
    
    def _compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.1) -> float:
        errors = torch.abs(predictions - targets)
        correct = (errors < threshold).all(dim=1)
        return correct.float().mean().item()
    
    def train(self, resume_from_checkpoint: bool = True):
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        
        start_epoch = 0
        
        if resume_from_checkpoint:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint is not None:
                start_epoch, _ = self.load_checkpoint(latest_checkpoint)
        
        print(f"Starting training for {self.config.MATRIX_SIZE}x{self.config.MATRIX_SIZE} matrix multiplication")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}")
        print(f"Training from epoch: {start_epoch + 1} to {self.config.TRAIN_EPOCHS}")
        print("-" * 80)
        
        self.metrics_tracker.start_epoch()
        
        for epoch in range(start_epoch, self.config.TRAIN_EPOCHS):
            self.epoch = epoch
            
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                step_start = time.time()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                step_time = time.time() - step_start
                self.metrics_tracker.log_iteration(step_time)
                
                self.global_step += 1
                
                epoch_loss += loss.item() * inputs.size(0)
                correct = (torch.abs(outputs - targets) < 0.1).all(dim=1).sum().item()
                epoch_correct += correct
                epoch_total += inputs.size(0)
            
            epoch_loss /= epoch_total
            epoch_acc = epoch_correct / epoch_total
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    correct = (torch.abs(outputs - targets) < 0.1).all(dim=1).sum().item()
                    val_correct += correct
                    val_total += inputs.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            compute_metrics = (self.global_step % self.config.METRICS_COMPUTE_INTERVAL == 0)
            
            if compute_metrics:
                sample_inputs = next(iter(self.train_loader))[0][:min(100, self.config.BATCH_SIZE)]
                
                lc = LocalComplexity.from_model(model, sample_inputs)
                sp = Superposition.from_model(model, rank=4)
            else:
                lc = self.metrics_tracker.local_complexities[-1] if self.metrics_tracker.local_complexities else 0.0
                sp = self.metrics_tracker.superpositions[-1] if self.metrics_tracker.superpositions else 1.0
            
            current_wd = self.thermal_engine.compute_weight_decay(lc, sp, epoch)
            
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = current_wd
            
            self.metrics_tracker.log_metrics(
                epoch_loss, val_loss, epoch_acc, val_acc,
                lc, sp, optimizer.param_groups[0]['lr'], current_wd
            )
            
            ips = self.metrics_tracker.compute_ips()
            
            thermal_status = self.thermal_engine.get_status(lc, sp)
            
            metrics_str = (
                f"Loss: train={epoch_loss:.6f}, val={val_loss:.6f} | "
                f"Acc: train={epoch_acc:.4f}, val={val_acc:.4f} | "
                f"LC={lc:.4f}, SP={sp:.4f} | "
                f"WD={current_wd:.6f} | "
                f"{thermal_status} | "
                f"Iter/s: {ips:.1f}"
            )
            
            print(f"Epoch {epoch+1:4d}/{self.config.TRAIN_EPOCHS} | {metrics_str}")
            
            current_time = time.time()
            if current_time - self.last_checkpoint_time >= self.config.CHECKPOINT_INTERVAL_MINUTES * 60:
                self._save_checkpoint(model, optimizer, epoch, ips)
                self.last_checkpoint_time = current_time
            
            if val_acc >= 0.9999 and epoch_acc >= 0.9999:
                print(f"\nGrokking achieved at epoch {epoch+1}!")
                print(f"Final metrics - LC: {lc:.4f}, SP: {sp:.4f}")
                break
        
        self._save_checkpoint(model, optimizer, self.config.TRAIN_EPOCHS - 1, self.metrics_tracker.compute_ips())
        
        summary = self.metrics_tracker.get_summary()
        print("\n" + "=" * 80)
        print("Training Summary:")
        print(f"  Final Train Loss: {summary['final_train_loss']:.6f}")
        print(f"  Final Val Loss: {summary['final_val_loss']:.6f}")
        print(f"  Final Train Acc: {summary['final_train_acc']:.4f}")
        print(f"  Final Val Acc: {summary['final_val_acc']:.4f}")
        print(f"  Final LC: {summary['final_lc']:.4f}")
        print(f"  Final SP: {summary['final_sp']:.4f}")
        print(f"  Average Iter/s: {summary['avg_ips']:.1f}")
        print("=" * 80)
        
        return self.model, self.metrics_tracker
    
    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, epoch: int, ips: float):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"matrix_{self.config.MATRIX_SIZE}x{self.config.MATRIX_SIZE}_epoch_{epoch+1}_{timestamp}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'matrix_size': self.config.MATRIX_SIZE,
                'hidden_dim': self.config.HIDDEN_DIM,
                'num_layers': self.config.NUM_LAYERS,
                'activation': self.config.ACTIVATION
            },
            'metrics': self.metrics_tracker.get_summary(),
            'timestamp': timestamp,
            'iterations_per_second': ips,
            'train_losses': self.metrics_tracker.train_losses,
            'val_losses': self.metrics_tracker.val_losses,
            'train_accuracies': self.metrics_tracker.train_accuracies,
            'val_accuracies': self.metrics_tracker.val_accuracies,
            'local_complexities': self.metrics_tracker.local_complexities,
            'superpositions': self.metrics_tracker.superpositions,
            'learning_rates': self.metrics_tracker.learning_rates,
            'weight_decays': self.metrics_tracker.weight_decays,
            'iterations_per_second_history': self.metrics_tracker.iterations_per_second
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")
    
    def zero_shot_transfer(self, target_matrix_size: int) -> Tuple[MLPModel, Dict]:
        print(f"\n{'='*80}")
        print(f"Performing Zero-Shot Transfer to {target_matrix_size}x{target_matrix_size} matrices")
        print(f"{'='*80}")
        
        new_input_dim = target_matrix_size * target_matrix_size * 2
        new_output_dim = target_matrix_size * target_matrix_size
        new_hidden_dim = self.config.HIDDEN_DIM
        
        expanded_model = self.model.expand_for_new_task(new_input_dim, new_output_dim, new_hidden_dim)
        expanded_model = expanded_model.to(self.device)
        
        test_dataset = MatrixMultiplicationDataset(
            target_matrix_size,
            self.config.NUM_SAMPLES,
            self.config.RANDOM_RANGE,
            self.device
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        expanded_model.eval()
        
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = expanded_model(inputs)
                predictions = outputs
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                correct += (torch.abs(predictions - targets) < 0.1).all(dim=1).sum().item()
                total += inputs.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        loss = nn.MSELoss()(all_predictions, all_targets).item()
        
        sample_inputs = next(iter(test_loader))[0][:min(100, self.config.BATCH_SIZE)]
        lc = LocalComplexity.from_model(expanded_model, sample_inputs)
        sp = Superposition.from_model(expanded_model, rank=4)
        
        results = {
            'target_size': target_matrix_size,
            'accuracy': accuracy,
            'loss': loss,
            'local_complexity': lc,
            'superposition': sp,
            'samples_tested': total
        }
        
        print(f"Zero-Shot Transfer Results for {target_matrix_size}x{target_matrix_size}:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Loss: {loss:.6f}")
        print(f"  Local Complexity (LC): {lc:.4f}")
        print(f"  Superposition (SP): {sp:.4f}")
        print(f"  Thermal Status: {self.thermal_engine.get_status(lc, sp)}")
        
        return expanded_model, results


def run_full_experiment():
    print("=" * 80)
    print("Matrix Grokking with Zero-Shot Transfer")
    print("=" * 80)
    
    config = Config()
    
    print(f"\nConfiguration:")
    print(f"  Matrix Size: {config.MATRIX_SIZE}x{config.MATRIX_SIZE}")
    print(f"  Hidden Dim: {config.HIDDEN_DIM}")
    print(f"  Num Layers: {config.NUM_LAYERS}")
    print(f"  Activation: {config.ACTIVATION}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Weight Decay: {config.WEIGHT_DECAY}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Train Epochs: {config.TRAIN_EPOCHS}")
    print(f"  Checkpoint Interval: {config.CHECKPOINT_INTERVAL_MINUTES} minutes")
    print(f"  Metrics Compute Interval: every {config.METRICS_COMPUTE_INTERVAL} steps")
    print(f"  Expansion Sizes: {config.EXPANSION_SIZES}")
    print(f"  Resume from Checkpoint: {config.RESUME_FROM_CHECKPOINT}")
    print(f"  Force Start Fresh: {config.FORCE_START_FRESH}")
    
    grokker = MatrixGrokker(config)
    
    should_resume = config.RESUME_FROM_CHECKPOINT and not config.FORCE_START_FRESH
    
    trained_model, metrics = grokker.train(resume_from_checkpoint=should_resume)
    
    transfer_results = {}
    for target_size in config.EXPANSION_SIZES:
        _, results = grokker.zero_shot_transfer(target_size)
        transfer_results[target_size] = results
    
    print("\n" + "=" * 80)
    print("Final Transfer Results Summary:")
    print("=" * 80)
    for size, results in transfer_results.items():
        print(f"\n{size}x{size} Matrix Transfer:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  LC: {results['local_complexity']:.4f}")
        print(f"  SP: {results['superposition']:.4f}")
    
    return grokker, metrics, transfer_results


def resume_from_latest_checkpoint(config: Optional[Config] = None):
    if config is None:
        config = Config()
    
    grokker = MatrixGrokker(config)
    
    latest = grokker.find_latest_checkpoint()
    if latest is not None:
        epoch, saved_metrics = grokker.load_checkpoint(latest)
        print(f"\nResuming training from epoch {epoch + 1}")
        trained_model, metrics = grokker.train(resume_from_checkpoint=True)
        return trained_model, metrics
    else:
        print("\nNo checkpoint found to resume from.")
        return None, None


def load_specific_checkpoint(checkpoint_path: str, config: Optional[Config] = None):
    if config is None:
        config = Config()
    
    grokker = MatrixGrokker(config)
    
    path = Path(checkpoint_path)
    if path.exists():
        epoch, saved_metrics = grokker.load_checkpoint(path)
        print(f"\nLoaded checkpoint from epoch {epoch + 1}")
        trained_model, metrics = grokker.train(resume_from_checkpoint=True)
        return trained_model, metrics
    else:
        print(f"\nCheckpoint not found: {checkpoint_path}")
        return None, None


if __name__ == "__main__":
    grokker, final_metrics, transfer_results = run_full_experiment()
