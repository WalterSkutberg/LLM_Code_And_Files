"""
Huvudprogram för träning av transformerbaserad språkmodell

Denna fil körs för att genomföra den kompletta pipelinen:
1. Datainsamling och rensning
2. Tokenisering
3. Modellträning
4. Utvärdering
"""

import sys
import os
from pathlib import Path

# Lägg till src till path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader

from data_loader import DataLoader as DataLoaderClass
from tokenizer import BPETokenizer
from model import TransformerLM
from trainer import ModelTrainer, TrainingConfig, TokenSequenceDataset
from evaluator import ModelEvaluator, LanguageVariationAnalyzer
from utils import (
    print_model_summary,
    setup_device,
    seed_everything,
    count_parameters,
    visualize_training_history
)


def main():
    """Huvudfunktion som kör hela pipelinen."""
    
    print("\n" + "="*70)
    print("TRANSFORMERBASERAD SPRÅKMODELL - TRÄNING OCH UTVÄRDERING")
    print("="*70)
    
    # ========== SETUP ==========
    print("\n[1/7] SETUP")
    print("-" * 70)
    
    seed_everything(42)
    device = setup_device()
    
    # Skapa kataloger
    data_dir = "data"
    model_dir = "models"
    Path(data_dir).mkdir(exist_ok=True)
    Path(model_dir).mkdir(exist_ok=True)
    
    # ========== DATAINSAMLING OCH RENSNING ==========
    print("\n[2/7] DATAINSAMLING OCH RENSNING")
    print("-" * 70)
    
    data_loader = DataLoaderClass(data_dir=data_dir)
    
    # Kontrollera om data redan finns
    processed_file = Path(data_dir) / "processed" / "cleaned_text.txt"
    
    if not processed_file.exists():
        # Ladda ner Frankenstein
        raw_file = data_loader.download_frankenstein()
        
        if raw_file:
            # Rensa data
            cleaned_file = data_loader.prepare_data(raw_file)
            
            # Dela upp i train/val/test
            train_file, val_file, test_file = data_loader.split_data(cleaned_file)
        else:
            print("✗ Kunde inte ladda ner data. Avslutar.")
            return
    else:
        print(f"✓ Rensat data redan sparad: {processed_file}")
        train_file = Path(data_dir) / "processed" / "train.txt"
        val_file = Path(data_dir) / "processed" / "val.txt"
        test_file = Path(data_dir) / "processed" / "test.txt"
    
    # ========== TOKENISERING ==========
    print("\n[3/7] TOKENISERING (BPE)")
    print("-" * 70)
    
    tokenizer_path = Path(model_dir) / "tokenizer.pkl"
    
    if tokenizer_path.exists():
        print(f"Laddar sparad tokenizer...")
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.load(str(tokenizer_path))
    else:
        # Träna tokenizer
        print("Tränar tokenizer...")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_text = f.read()
        
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.train([train_text])
        tokenizer.save(str(tokenizer_path))
    
    vocab_size = len(tokenizer)
    print(f"✓ Tokenizer vocab-storlek: {vocab_size}")
    
    # Tokenisera data
    print("\nTokeniserar data...")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(val_file, 'r', encoding='utf-8') as f:
        val_text = f.read()
    with open(test_file, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    test_tokens = tokenizer.encode(test_text)
    
    print(f"✓ Träningstoken: {len(train_tokens)}")
    print(f"✓ Valideringstoken: {len(val_tokens)}")
    print(f"✓ Testtoken: {len(test_tokens)}")
    
    # ========== MODELLARKITEKTUR ==========
    print("\n[4/7] MODELLARKITEKTUR")
    print("-" * 70)
    
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=1024,
        max_seq_len=512,
        dropout=0.1
    ).to(device)
    
    print_model_summary(model)
    
    # ========== TRÄNING ==========
    print("\n[5/7] TRÄNING")
    print("-" * 70)
    
    # Skapa datasets och dataloaders
    seq_len = 256
    
    train_dataset = TokenSequenceDataset(train_tokens, seq_len=seq_len)
    val_dataset = TokenSequenceDataset(val_tokens, seq_len=seq_len)
    test_dataset = TokenSequenceDataset(test_tokens, seq_len=seq_len)
    
    batch_size = 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Träningskonfiguration
    config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=1e-3,
        num_epochs=5,  # Liten antal för demo
        weight_decay=0.01,
        warmup_steps=500,
        max_grad_norm=1.0,
        log_interval=50,
        save_interval=200,
        device=device.type
    )
    
    # Trainer
    trainer = ModelTrainer(model, config, checkpoint_dir=model_dir)
    
    # Träna modellen
    history = trainer.train(
        train_loader,
        val_loader,
        early_stopping_patience=2
    )
    
    # Visualisera träningshistorik
    visualize_training_history(
        history['train_losses'],
        history['val_losses'],
        save_path=str(Path(model_dir) / "training_history.png")
    )
    
    # ========== UTVÄRDERING ==========
    print("\n[6/7] UTVÄRDERING")
    print("-" * 70)
    
    evaluator = ModelEvaluator(model, tokenizer, device=device.type)
    
    # Utvärdera på test set
    print("\nTessetav med tränad modell...")
    test_stats = evaluator.evaluate_on_dataset(test_loader)
    
    print(f"\nTest resultater:")
    print(f"  Förlust: {test_stats['loss']:.4f}")
    print(f"  Perplexity: {test_stats['perplexity']:.2f}")
    print(f"  Noggrannhet: {test_stats['accuracy']*100:.2f}%")
    
    # ========== TEXTGENERERING OCH SPRÅKVARIATION ==========
    print("\n[7/7] TEXTGENERERING OCH SPRÅKVARIATION")
    print("-" * 70)
    
    variation_analyzer = LanguageVariationAnalyzer(
        model,
        tokenizer,
        device=device.type
    )
    
    # Generera text
    print("\nGenererar text...")
    seed = "The monster"
    
    for temp in [0.5, 0.7, 1.0]:
        generated = evaluator.generate_text(
            seed,
            max_length=100,
            temperature=temp,
            top_k=50
        )
        
        print(f"\nTemperature={temp}:")
        print(f"  Seed: '{seed}'")
        print(f"  Genererad: {generated[:100]}...")
    
    # Analysera språkvariation
    print("\n" + "-" * 70)
    print("Analyserar språkvariation...")
    
    variation = variation_analyzer.analyze_output_variance(
        seed_text="Victor",
        num_samples=3,
        temperature=0.8
    )
    
    variation_analyzer.print_variation_analysis(variation)
    
    # ========== SAMMANFATTNING ==========
    print("\n" + "="*70)
    print("TRÄNING AVSLUTAD!")
    print("="*70)
    
    print(f"\nSparat till mapp: {model_dir}")
    print(f"\nFiler:")
    print(f"  - best_model.pt: Bästa modell baserad på validering")
    print(f"  - tokenizer.pkl: Sparad tokenizer")
    print(f"  - training_history.json: Träningshistorik")
    print(f"  - training_history.png: Visualisering av träning")
    
    print(f"\nModellstatistik:")
    print(f"  - Parametrar: {count_parameters(model):,}")
    print(f"  - Vocab-storlek: {vocab_size}")
    print(f"  - Träningstoken: {len(train_tokens):,}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Träning avbruten av användare.")
    except Exception as e:
        print(f"\n✗ Fel: {e}")
        import traceback
        traceback.print_exc()

