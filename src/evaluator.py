"""
Modellbedömning och utvärdering
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from typing import List, Dict, Tuple
import numpy as np


class ModelEvaluator:
    """
    Utvärderar modellens prestanda på olika mätvärden.
    """
    
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
    
    def calculate_perplexity(self, data_loader: DataLoader) -> float:
        """
        Beräknar perplexity (förvirring) på testdata.
        
        Perplexity = exp(genomsnittlig cross-entropy förlust)
        Lägre perplexity = bättre modell
        
        Args:
            data_loader: DataLoader med testdata
            
        Returns:
            float: Perplexity
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                logits, _ = self.model(input_ids)
                
                # Beräkna loss
                logits_flat = logits.view(-1, self.model.vocab_size)
                targets_flat = target_ids.view(-1)
                
                loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def generate_text(
        self,
        seed_text: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = None
    ) -> str:
        """
        Genererar text utifrån ett seed.
        
        Args:
            seed_text: Initiell text
            max_length: Maximal längd på utgången
            temperature: Randomness (höger = mer random)
            top_k: Top-K sampling (None = greedy)
            
        Returns:
            str: Genererad text
        """
        self.model.eval()
        
        # Enkoda seed
        seed_ids = self.tokenizer.encode(seed_text)
        input_ids = torch.tensor([seed_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Generera
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
        
        # Dekoda
        output_ids = output_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(output_ids)
        
        return generated_text
    
    def analyze_attention_patterns(self, text: str) -> Dict:
        """
        Analyserar attention-mönster för en given text.
        
        Args:
            text: Text att analysera
            
        Returns:
            Dict med attention-statistik
        """
        self.model.eval()
        
        # Enkoda
        token_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits, hidden = self.model(input_ids, return_hidden=True)
        
        analysis = {
            'num_tokens': len(token_ids),
            'hidden_dim': hidden.shape[-1],
            'hidden_stats': {
                'mean': float(hidden.mean().item()),
                'std': float(hidden.std().item()),
                'max': float(hidden.max().item()),
                'min': float(hidden.min().item())
            },
            'logits_stats': {
                'mean': float(logits.mean().item()),
                'std': float(logits.std().item()),
            }
        }
        
        return analysis
    
    def evaluate_on_dataset(self, data_loader: DataLoader) -> Dict:
        """
        Utvärderar modellen på ett dataset.
        
        Args:
            data_loader: DataLoader
            
        Returns:
            Dict med statistik
        """
        self.model.eval()
        
        all_logits = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                logits, _ = self.model(input_ids)
                
                # Spara logits och targets
                all_logits.append(logits.cpu())
                all_targets.append(target_ids.cpu())
                
                # Beräkna loss
                logits_flat = logits.view(-1, self.model.vocab_size)
                targets_flat = target_ids.view(-1)
                
                loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)
                total_loss += loss.item()
                num_batches += 1
        
        # Beräkna accuracy
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        predictions = all_logits.argmax(dim=-1)
        
        # Ignorera padding tokens
        mask = all_targets != 0
        accuracy = (predictions[mask] == all_targets[mask]).float().mean()
        
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(avg_loss)
        
        stats = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': float(accuracy.item()),
            'num_tokens': int(mask.sum()),
            'num_batches': num_batches
        }
        
        return stats
    
    def compare_datasets(
        self,
        dataset1_name: str,
        dataset1_loader: DataLoader,
        dataset2_name: str,
        dataset2_loader: DataLoader
    ) -> Dict:
        """
        Jämför modellprestanda på två dataset.
        
        Args:
            dataset1_name: Namn på första dataset
            dataset1_loader: DataLoader för första dataset
            dataset2_name: Namn på andra dataset
            dataset2_loader: DataLoader för andra dataset
            
        Returns:
            Dict med jämförelse
        """
        stats1 = self.evaluate_on_dataset(dataset1_loader)
        stats2 = self.evaluate_on_dataset(dataset2_loader)
        
        comparison = {
            'dataset1': {
                'name': dataset1_name,
                'stats': stats1
            },
            'dataset2': {
                'name': dataset2_name,
                'stats': stats2
            },
            'difference': {
                'loss_diff': stats2['loss'] - stats1['loss'],
                'perplexity_diff': stats2['perplexity'] - stats1['perplexity'],
                'accuracy_diff': stats2['accuracy'] - stats1['accuracy']
            }
        }
        
        return comparison
    
    def print_evaluation_report(self, stats: Dict):
        """Skriver ut en utvärderingsrapport."""
        print("\n" + "="*60)
        print("UTVÄRDERINGSRAPPORT")
        print("="*60)
        
        print(f"\nFörlust:        {stats.get('loss', 'N/A'):.4f}")
        print(f"Perplexity:     {stats.get('perplexity', 'N/A'):.2f}")
        print(f"Noggrannhet:    {stats.get('accuracy', 'N/A')*100:.2f}%")
        print(f"Antal tokens:   {stats.get('num_tokens', 'N/A')}")
        
        if 'hidden_stats' in stats:
            hs = stats['hidden_stats']
            print(f"\nHidden state statistik:")
            print(f"  Medelvärde:  {hs.get('mean', 'N/A'):.4f}")
            print(f"  Std Dev:     {hs.get('std', 'N/A'):.4f}")
        
        print("\n" + "="*60)


class LanguageVariationAnalyzer:
    """
    Analyserar hur modellen varierar sitt språk baserat på kontext.
    """
    
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = ModelEvaluator(model, tokenizer, device)
        self.device = device
    
    def analyze_output_variance(
        self,
        seed_text: str,
        num_samples: int = 5,
        temperature: float = 0.7
    ) -> Dict:
        """
        Analyserar variationen i genererad text för samma seed.
        
        Args:
            seed_text: Initiell text
            num_samples: Antal sampel att generera
            temperature: Temperature för sampling
            
        Returns:
            Dict med variationsanalys
        """
        samples = []
        
        for i in range(num_samples):
            text = self.evaluator.generate_text(
                seed_text,
                max_length=50,
                temperature=temperature
            )
            samples.append(text)
        
        analysis = {
            'seed': seed_text,
            'num_samples': num_samples,
            'samples': samples,
            'temperature': temperature,
            'unique_samples': len(set(samples))
        }
        
        return analysis
    
    def print_variation_analysis(self, analysis: Dict):
        """Skriver ut variationsanalysis."""
        print("\n" + "="*60)
        print("SPRÅKVARIATION - ANALYS")
        print("="*60)
        
        print(f"\nSeed: '{analysis['seed']}'")
        print(f"Temperature: {analysis['temperature']}")
        print(f"Unika sampel: {analysis['unique_samples']}/{analysis['num_samples']}")
        
        print("\nGenererade sampel:")
        for i, sample in enumerate(analysis['samples'], 1):
            print(f"  {i}. {sample[:60]}...")
        
        print("\n" + "="*60)
