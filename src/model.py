"""
Transformer-baserad språkmodell (GPT-liknande)

Arkitektur:
- Token Embeddings
- Positional Embeddings
- Transformer Blocks (Multi-Head Attention + Feed-Forward)
- Output Layer (nästa token prediction)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mekanismen.
    
    Låter modellen fokusera på olika delar av sekvensen samtidigt.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim måste vara delbar med num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linjära transformationer för Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Utgångslinjär
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: Causal mask för att förhindra att se framtida tokens
            
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Projicera till Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Dela in i flera heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch_size, num_heads, seq_len, head_dim)
        
        # Beräkna attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch_size, num_heads, seq_len, seq_len)
        
        # Applicera causal mask (förhindra att se framtida tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax för att få attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Applicera attention på values
        output = torch.matmul(attn_weights, V)
        # (batch_size, num_heads, seq_len, head_dim)
        
        # Kombinera heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        
        # Slutgiltig linjär transformation
        output = self.W_o(output)
        
        return output


class FeedForward(nn.Module):
    """
    Feed-Forward Network med GELU aktivering.
    
    Används efter attention för att processera features.
    """
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    En Transformer-block med:
    - Multi-Head Attention
    - Layer Normalization
    - Feed-Forward Network
    - Residual connections
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: Causal mask
            
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # Multi-Head Attention med residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-Forward med residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding för att ge modellen information om token-positionen.
    
    Använder sinusformler för att skapa unika encodingar för varje position.
    """
    
    def __init__(self, embed_dim: int, max_seq_len: int = 2048):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Beräkna frequency för varje dimension
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            -(math.log(10000.0) / embed_dim)
        )
        
        # Applicera sin på jämna dimensioner
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Applicera cos på ojämna dimensioner
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Registrera som buffer (inte en parameter, men sparas vid save/load)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        return x + self.pe[:, :x.shape[1], :]


class TransformerLM(nn.Module):
    """
    Hela GPT-liknande språkmodellen.
    
    Arkitektur:
    Input → Embeddings → Positional Encoding → Transformer Blocks → Output Layer
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initiera vikter
        self._init_weights()
    
    def _init_weights(self):
        """Initiera modellvikter."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Skapar en causal mask för att förhindra att modellen ser framtida tokens.
        
        Args:
            seq_len: Längden på sekvensen
            device: Device (cpu eller gpu)
            
        Returns:
            mask: (seq_len, seq_len) där mask[i, j] = 1 om j <= i, annars 0
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass genom modellen.
        
        Args:
            input_ids: (batch_size, seq_len) - token IDs
            return_hidden: Om True, returnerar även hidden states
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            hidden_states: (batch_size, seq_len, embed_dim) om return_hidden=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        
        # Skala embedding enligt Transformer-pappret
        x = x * math.sqrt(self.embed_dim)
        
        # Positional encoding
        x = self.positional_encoding(x)
        
        x = self.dropout(x)
        
        # Causal mask
        mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Output layer
        x = self.norm(x)
        hidden_states = x if return_hidden else None
        
        logits = self.output_layer(x)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden_states
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = None
    ) -> torch.Tensor:
        """
        Genererar text autoregressivt.
        
        Args:
            input_ids: (batch_size, seq_len) - seed tokens
            max_length: Maximal längd på sekvensen
            temperature: Kontrollerar randomness (höger = mer random)
            top_k: Om satt, sample endast från top-k tokens
            
        Returns:
            generated: (batch_size, max_length)
        """
        device = input_ids.device
        
        for _ in range(max_length - input_ids.shape[1]):
            # Begränsa input till max_seq_len
            if input_ids.shape[1] > self.max_seq_len:
                input_slice = input_ids[:, -self.max_seq_len:]
            else:
                input_slice = input_ids
            
            # Forward pass
            logits, _ = self.forward(input_slice)
            
            # Nästa tokens logits
            next_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Applicera temperature
            next_logits = next_logits / temperature
            
            # Top-K sampling
            if top_k is not None:
                values, indices = torch.topk(next_logits, top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(-1, indices, values)
            
            # Sample från distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # Lägg till nästa token
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids


if __name__ == "__main__":
    # Demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Skapa modell
    model = TransformerLM(
        vocab_size=1000,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=1024,
        max_seq_len=512
    ).to(device)
    
    print(f"Modell skapad. Totalt parametrar: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        logits, hidden = model(input_ids, return_hidden=True)
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Hidden states shape: {hidden.shape}")
    
    # Test generation
    print("\nGenerering av text...")
    with torch.no_grad():
        seed = torch.tensor([[1, 2, 3]]).to(device)
        generated = model.generate(seed, max_length=50)
        print(f"Genererad sekvens shape: {generated.shape}")
