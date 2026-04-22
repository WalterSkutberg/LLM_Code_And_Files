"""
Träningsloop för transformermodellen
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
from typing import Tuple, List, Dict
import json
from tqdm import tqdm
import math


class TrainingConfig:
    """Konfiguration för träning."""
    
    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        log_interval: int = 50,
        save_interval: int = 500,
        device: str = None
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


class TokenSequenceDataset:
    """Dataset för tokeniserade sekvenser."""
    
    def __init__(self, token_ids: List[int], seq_len: int = 512):
        self.token_ids = token_ids
        self.seq_len = seq_len
    
    def __len__(self):
        # Antalet sekvenser vi kan skapa
        return max(0, len(self.token_ids) - self.seq_len)
    
    def __getitem__(self, idx):
        # Returnera sekvens och nästa token (för förutsägelse)
        input_ids = self.token_ids[idx:idx + self.seq_len]
        target_ids = self.token_ids[idx + 1:idx + self.seq_len + 1]
        
        # Pad om nödvändigt
        if len(input_ids) < self.seq_len:
            input_ids = input_ids + [0] * (self.seq_len - len(input_ids))
        if len(target_ids) < self.seq_len:
            target_ids = target_ids + [0] * (self.seq_len - len(target_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }


class ModelTrainer:
    """Trainer för transformermodellen."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        checkpoint_dir: str = "models"
    ):
        self.model = model
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimerare
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Förlustfunktion
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorera padding
        
        # Schemaläggare för learning rate
        self.scheduler = None
        
        # Historik
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.global_step = 0
        
        print(f"Trainer initialiserad. Device: {self.device}")
    
    def _setup_scheduler(self, total_steps: int):
        """Sätter upp learning rate scheduler."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Warm-up fase
                return float(step) / float(max(1, self.config.warmup_steps))
            else:
                # Cosine decay
                return max(0.0, 0.5 * (1.0 + math.cos(
                    math.pi * float(step - self.config.warmup_steps) / 
                    float(max(1, total_steps - self.config.warmup_steps))
                )))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Tränar en epoch.
        
        Args:
            train_loader: DataLoader för träningsdata
            
        Returns:
            Genomsnittlig förlust för epoken
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            logits, _ = self.model(input_ids)
            
            # Beräkna loss
            # Reshape för cross_entropy
            logits_flat = logits.view(-1, self.model.vocab_size)
            targets_flat = target_ids.view(-1)
            
            loss = self.criterion(logits_flat, targets_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Update
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': self.global_step
                })
            
            # Checkpoint saving
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        avg_loss = total_loss / max(1, num_batches)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validerar modellen.
        
        Args:
            val_loader: DataLoader för valideringsdata
            
        Returns:
            Genomsnittlig validerings-förlust
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                logits, _ = self.model(input_ids)
                
                logits_flat = logits.view(-1, self.model.vocab_size)
                targets_flat = target_ids.view(-1)
                
                loss = self.criterion(logits_flat, targets_flat)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{total_loss / (num_batches):.4f}'})
        
        avg_loss = total_loss / max(1, num_batches)
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        early_stopping_patience: int = 3
    ) -> Dict:
        """
        Tränar modellen för flera epoker.
        
        Args:
            train_loader: DataLoader för träningsdata
            val_loader: DataLoader för valideringsdata
            early_stopping_patience: Hur många epoker utan förbättring innan stoppning
            
        Returns:
            Dict med träningshistorik
        """
        # Sätt upp scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self._setup_scheduler(total_steps)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Träning
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validering
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                print(f"Val Loss:   {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(f"best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping efter {early_stopping_patience} epoker utan förbättring.")
                        break
            
            # Spara checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        # Spara slutligt resultat
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'total_steps': self.global_step
        }
        
        with open(self.checkpoint_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Träning klar!")
        print(f"{'='*60}")
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Sparar modellcheckpoint."""
        path = self.checkpoint_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        
        print(f"✓ Checkpoint sparad: {path}")
    
    def load_checkpoint(self, filename: str):
        """Laddar modellcheckpoint."""
        path = self.checkpoint_dir / filename
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"✓ Checkpoint laddat: {path}")
