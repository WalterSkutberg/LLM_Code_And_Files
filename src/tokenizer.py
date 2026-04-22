"""
Byte Pair Encoding (BPE) Tokenisering
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import re


class BPETokenizer:
    """
    Implementerar Byte Pair Encoding för tokenisering.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_tokenizer = re.compile(r'\b\w+\b|[^\w\s]')
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.word_vocab: Dict[str, int] = {}
    
    def _get_stats(self, pairs: Dict[Tuple[str, str], int]) -> Tuple[str, str]:
        """Hittar det mest frekventa paret."""
        return max(pairs, key=pairs.get)
    
    def _merge_vocab(self, pair: Tuple[str, str], word_dict: Dict[str, int]) -> Dict[str, int]:
        """Mergear ett specifikt par i ordförrådet."""
        new_dict = {}
        for word, freq in word_dict.items():
            new_word = word.replace(' '.join(pair), ''.join(pair))
            new_dict[new_word] = freq
        return new_dict
    
    def train(self, texts: List[str]):
        """
        Tränar tokenizern på texterna.
        
        Args:
            texts (List[str]): Lista med texter att träna på
        """
        print(f"Tränar BPE tokenizer med vocab_size={self.vocab_size}...")
        
        # Skapa initial ordförteckning från alla ord
        word_dict = Counter()
        all_chars = set()
        
        for text in texts:
            # Tokenisera till ord
            words = self.word_tokenizer.findall(text.lower())
            
            for word in words:
                # Dela ord i tecken med mellanslag
                word_spaced = ' '.join(word) + ' </w>'
                word_dict[word_spaced] += 1
                all_chars.update(word)
        
        # Initiera vocab med alla unika tecken
        self.vocab = {char: i for i, char in enumerate(sorted(all_chars))}
        vocab_size = len(self.vocab)
        
        print(f"Initial alphabet: {vocab_size} tecken")
        
        # Iterera och mergea de mest frekventa paren
        num_merges = self.vocab_size - vocab_size
        
        for i in range(num_merges):
            # Räkna alla par
            pairs = Counter()
            
            for word, freq in word_dict.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j + 1])
                    pairs[pair] += freq
            
            if not pairs:
                print(f"Inga fler par att mergea. Stoppat vid {vocab_size} tokens.")
                break
            
            # Hittar det mest frekventa paret
            best_pair = self._get_stats(pairs)
            
            # Mergea detta par
            word_dict = self._merge_vocab(best_pair, word_dict)
            self.merges.append(best_pair)
            
            # Lägg till det nya paret till vocab
            new_token = ''.join(best_pair)
            self.vocab[new_token] = vocab_size
            vocab_size += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Merge {i + 1}/{num_merges}, Vocab size: {vocab_size}")
        
        print(f"✓ Träning klar. Final vocab size: {len(self.vocab)}")
    
    def _encode_word(self, word: str) -> List[str]:
        """Enkoderar ett enstaka ord med BPE."""
        # Dela ord i tecken
        symbols = list(word) + ['</w>']
        
        # Applicera merges
        for pair in self.merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(''.join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            
            if len(symbols) == 1:
                break
        
        return symbols
    
    def encode(self, text: str) -> List[int]:
        """
        Enkoderar text till token-ID:n.
        
        Args:
            text (str): Text att enkoda
            
        Returns:
            List[int]: Token-ID:n
        """
        # Tokenisera till ord
        words = self.word_tokenizer.findall(text.lower())
        
        tokens = []
        for word in words:
            # Enkoda varje ord med BPE
            subwords = self._encode_word(word)
            
            for subword in subwords:
                if subword in self.vocab:
                    tokens.append(self.vocab[subword])
                else:
                    # Unknown token - använd första tecken
                    tokens.append(self.vocab.get(subword[0], 0))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Dekoderar token-ID:n tillbaka till text.
        
        Args:
            tokens (List[int]): Token-ID:n
            
        Returns:
            str: Dekoderad text
        """
        # Skapa reverse vocab
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        text_parts = []
        for token_id in tokens:
            if token_id in reverse_vocab:
                text_parts.append(reverse_vocab[token_id])
        
        # Kombinera och ta bort word boundaries
        text = ''.join(text_parts)
        text = text.replace('</w>', ' ')
        
        return text.strip()
    
    def save(self, path: str):
        """Sparar tokenizer."""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Tokenizer sparad: {path}")
    
    def load(self, path: str):
        """Laddar tokenizer."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab = data['vocab']
        self.merges = data['merges']
        self.vocab_size = data['vocab_size']
        
        print(f"✓ Tokenizer laddad: {path}")
    
    def __len__(self) -> int:
        """Returnerar vocab-storlek."""
        return len(self.vocab)
    
    def get_vocab_size(self) -> int:
        """Returnerar vocab-storlek."""
        return len(self.vocab)


if __name__ == "__main__":
    # Demo
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "The lazy dog sleeps under the tree",
        "A quick brown fox runs through the forest"
    ]
    
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(sample_texts)
    
    # Test enkodning/dekodning
    text = "The brown fox"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {text}")
    print(f"Enkodad: {encoded}")
    print(f"Dekodad: {decoded}")
    print(f"Vocab size: {len(tokenizer)}")
