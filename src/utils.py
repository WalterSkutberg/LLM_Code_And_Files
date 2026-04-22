"""
Hjälpfunktioner och verktyg
"""

import torch
import torch.nn as nn
from typing import List
import json
import os
from pathlib import Path


def count_parameters(model: nn.Module) -> int:
    """
    Räknar totalt antal trainearbara parametrar i modellen.
    
    Args:
        model: PyTorch modell
        
    Returns:
        int: Antal parametrar
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """
    Beräknar modellens storlek i MB.
    
    Args:
        model: PyTorch modell
        
    Returns:
        float: Storlek i MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    total_size = param_size + buffer_size
    size_mb = total_size / (1024 ** 2)
    
    return size_mb


def print_model_summary(model: nn.Module):
    """
    Skriver ut en sammanfattning av modellen.
    
    Args:
        model: PyTorch modell
    """
    print("\n" + "="*70)
    print("MODELL SAMMANFATTNING")
    print("="*70)
    
    # Arkitektur
    print("\nArkitektur:")
    print(model)
    
    # Parametrar
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParametrar:")
    print(f"  Totalt:        {total_params:,}")
    print(f"  Trainearbara:  {trainable_params:,}")
    
    # Storlek
    size_mb = get_model_size_mb(model)
    print(f"\nStorlek:")
    print(f"  {size_mb:.2f} MB")
    
    # Lager
    print(f"\nLager ({len(list(model.named_modules()))} moduler):")
    
    layer_types = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in layer_types:
            layer_types[module_type] = 0
        layer_types[module_type] += 1
    
    for layer_type, count in sorted(layer_types.items()):
        print(f"  {layer_type}: {count}")
    
    print("\n" + "="*70)


def load_config(config_path: str) -> dict:
    """
    Laddar konfiguration från JSON-fil.
    
    Args:
        config_path: Sökväg till config-fil
        
    Returns:
        dict: Konfiguration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: dict, config_path: str):
    """
    Sparar konfiguration till JSON-fil.
    
    Args:
        config: Konfiguration dict
        config_path: Sökväg för att spara
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_batches(
    token_ids: List[int],
    batch_size: int,
    seq_len: int,
    shuffle: bool = False
) -> List[torch.Tensor]:
    """
    Skapar batcher från tokeniserade ID:n.
    
    Args:
        token_ids: Lista av token-ID:n
        batch_size: Batch-storlek
        seq_len: Sekvens-längd
        shuffle: Om True, blandar data
        
    Returns:
        List[torch.Tensor]: Batcher av input
    """
    import random
    
    num_samples = len(token_ids) - seq_len
    indices = list(range(num_samples))
    
    if shuffle:
        random.shuffle(indices)
    
    batches = []
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        
        batch_input = []
        for idx in batch_indices:
            seq = token_ids[idx:idx + seq_len]
            
            # Pad om nödvändigt
            if len(seq) < seq_len:
                seq = seq + [0] * (seq_len - len(seq))
            
            batch_input.append(seq)
        
        batch_tensor = torch.tensor(batch_input, dtype=torch.long)
        batches.append(batch_tensor)
    
    return batches


def setup_device() -> torch.device:
    """
    Sätter upp enheten (GPU eller CPU).
    
    Returns:
        torch.device: Enheten att träna på
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA-enhet hittad: {torch.cuda.get_device_name(0)}")
        print(f"Minne: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Använder CPU")
    
    return device


def get_available_memory() -> float:
    """
    Får tillgängligt GPU-minne (om GPU finns).
    
    Returns:
        float: Minne i GB
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        return float('inf')  # CPU har inte en gräns


def seed_everything(seed: int = 42):
    """
    Sätter seed för reproducerbarhet.
    
    Args:
        seed: Seed-värde
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_learning_rate(optimizer) -> float:
    """
    Får den aktuella learning rate från optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        float: Learning rate
    """
    return optimizer.param_groups[0]['lr']


def visualize_training_history(
    train_losses: List[float],
    val_losses: List[float] = None,
    save_path: str = None
):
    """
    Visualiserar träningshistorik.
    
    Args:
        train_losses: Lista av träningsförluster
        val_losses: Lista av valideringsförluster
        save_path: Sökväg för att spara figur
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_losses, label='Train Loss', marker='o')
        
        if val_losses:
            plt.plot(val_losses, label='Val Loss', marker='s')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Träningshistorik')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=100)
            print(f"✓ Figur sparad: {save_path}")
        
        plt.show()
    
    except ImportError:
        print("Matplotlib inte installerat. Hoppa över visualisering.")


def analyze_text_statistics(text: str) -> dict:
    """
    Analyserar statistik för en text.
    
    Args:
        text: Text att analysera
        
    Returns:
        dict: Statistik
    """
    lines = text.split('\n')
    words = text.split()
    chars = list(text)
    
    unique_words = set(words)
    unique_chars = set(chars)
    
    stats = {
        'num_characters': len(text),
        'num_lines': len(lines),
        'num_words': len(words),
        'unique_words': len(unique_words),
        'unique_characters': len(unique_chars),
        'avg_word_length': sum(len(w) for w in words) / max(1, len(words)),
        'avg_line_length': sum(len(l) for l in lines) / max(1, len(lines)),
        'word_frequency_ratio': len(unique_words) / max(1, len(words))
    }
    
    return stats


if __name__ == "__main__":
    # Demo
    print("Utility-funktioner för LLM")
    print(f"CUDA tillgängligt: {torch.cuda.is_available()}")
    
    device = setup_device()
    
    # Demo med små modell
    from model import TransformerLM
    
    model = TransformerLM(vocab_size=1000)
    print_model_summary(model)
