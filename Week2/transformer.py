import torch
import torch.nn as nn
import math
from dataset import AnimalSoundDataset

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformers
    
    Since transformers process all positions in parallel, they lose sequential order.
    We add positional information using sine/cosine waves of different frequencies:
    1. Each position gets a unique pattern across dimensions
    2. Similar positions have similar patterns
    3. The pattern is deterministic (no learning required)
    4. Can extrapolate to longer sequences than seen in training
    
    The use of sine/cosine functions is key because:
    - They create smooth transitions between positions
    - They can represent relative positions (through arithmetic properties of sine/cosine)
    - They provide a unique pattern for each position that doesn't decay or explode
    """
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        # Create position indices: [0, 1, 2, ...max_seq_length-1]
        # Shape: (max_seq_length, 1)
        # These indices represent each position in the sequence
        position = torch.arange(max_seq_length).unsqueeze(1)
        
        # Create division term for different frequency bands
        # Shape: (d_model/2,)
        # Each dimension pair gets a different frequency:
        # - Low indices: High frequency → captures local position dependencies
        # - High indices: Low frequency → captures long-range dependencies
        # The log-space progression ensures good coverage of different scales
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initialize positional encoding matrix
        # Shape: (max_seq_length, d_model)
        # Each row will hold the encoding for one position
        pe = torch.zeros(max_seq_length, d_model)
        
        # Fill even indices with sin, odd indices with cos
        # Using both sin and cos gives each position a unique pattern
        # The frequency progression ensures these patterns are distinct
        # Example for position p, dimension i:
        # pe[p, 2i]     = sin(p / 10000^(2i/d_model))
        # pe[p, 2i + 1] = cos(p / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        # Register as buffer (not a parameter, but part of state)
        # - Buffers are saved/loaded with model but not trained
        # - This ensures PE moves to GPU with model if needed
        # - We don't train PE because the fixed pattern works well
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        Returns:
            Positional encoding added to input
        """
        # Add positional encoding to input embeddings
        # pe[:x.size(1)] slices to actual sequence length
        # Broadcasting handles batch dimension automatically
        return x + self.pe[:x.size(1)]


class LayerNorm(nn.Module):
    """
    Layer Normalization
    
    Normalizes input to have mean=0 and variance=1 across the feature dimension.
    This helps with training stability and gradient flow by:
    1. Preventing internal covariate shift
    2. Allowing deeper networks to train
    3. Making training more robust to different initialization
    4. Helping gradients flow through attention and deep networks
    
    Unlike Batch Norm, Layer Norm operates on each sample independently,
    making it more suitable for variable length sequences and NLP tasks.
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable parameters for scaling and shifting
        # Shape: (d_model,) - will broadcast across batch and sequence length
        self.gamma = nn.Parameter(torch.ones(d_model))    # Scaling factor
        self.beta = nn.Parameter(torch.zeros(d_model))    # Shift factor
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # Normalize over the last dimension (d_model)
        # Each position in the sequence is normalized independently
        # This is key for transformers as it allows parallel processing
        
        # Calculate mean and variance along last dimension
        # keepdim=True preserves the dimension for broadcasting
        # Shape: (batch, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False matches PyTorch's native LayerNorm
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize input
        # eps prevents division by zero and helps with numerical stability
        # x_norm shape: (batch, seq_len, d_model)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift normalized values
        # gamma and beta enable the network to control normalization strength
        # This is crucial as sometimes the network needs to "undo" normalization
        # For example, the network might learn to selectively bypass normalization
        # for certain features if they're already well-behaved
        return self.gamma * x_norm + self.beta

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    """
    def __init__(self, head_size=8, n_heads=4):
        super().__init__()
        self.n_heads = n_heads          # Number of attention heads
        self.head_size = head_size      # Size of each attention head
        self.d_model = head_size * n_heads  # Total model dimension
        
        # Linear projections for Q, K, V
        # Input shape:  (batch, seq_len, d_model)
        # Output shape: (batch, seq_len, d_model)
        # The d_model dimension is later reshaped into (n_heads, head_size)
        self.q_proj = nn.Linear(self.d_model, self.d_model)  # Query projection
        self.k_proj = nn.Linear(self.d_model, self.d_model)  # Key projection
        self.v_proj = nn.Linear(self.d_model, self.d_model)  # Value projection
        self.out_proj = nn.Linear(self.d_model, self.d_model)  # Final output projection
        
    def forward(self, x):
        # batch size, Time, dimension size.
        B, T, D = x.shape
        H = self.n_heads
        
        # x comes in as (Batch, Time, Dims)
        # 1. Project and reshape to (Batch, Time, n_heads, head_size)
        # 2. Transpose to (Batch, n_heads, Time, head_size) for parallel head processing
        q = self.q_proj(x).reshape(B, T, H, self.head_size).transpose(1, 2)  # B,H,T,head_size
        k = self.k_proj(x).reshape(B, T, H, self.head_size).transpose(1, 2)  # B,H,T,head_size
        v = self.v_proj(x).reshape(B, T, H, self.head_size).transpose(1, 2)  # B,H,T,head_size
        
        # We want each query vector to compute similarity scores with all key vectors
        # For a single query and key:
        # - Query is a vector of size (head_size)
        # - Key is a vector of size (head_size)
        # - Their dot product gives us a single attention score
        #
        # To do this for ALL queries with ALL keys efficiently:
        # - q shape:          (Batch, n_heads, Time_q, head_size)
        # - k needs shape:    (Batch, n_heads, head_size, Time_k)
        #
        # By transposing k, each query vector will dot product with all key vectors,
        # giving us attention scores between every query position and every key position
        k_for_matmul = k.transpose(-2, -1)  # (Batch, n_heads, head_size, Time_k)
        
        # Now each query will compute attention scores with all keys
        # Output shape (Batch, n_heads, Time_q, Time_k) gives us:
        # - For each query position (Time_q)
        # - Its attention score with each key position (Time_k)
        attn = torch.matmul(q, k_for_matmul)
        
        # Scale dot products by sqrt(head_size) to keep gradients stable
        # Otherwise values can get too large after softmax
        attn = attn / math.sqrt(self.head_size)
        
        # Create causal mask to prevent attending to future tokens
        # Shape: (T, T) where T is sequence length
        # - Lower triangular matrix of 1's, upper triangle of 0's
        # - At position (i,j): 1 if position i can attend to position j, else 0
        # - Each position i can only attend to positions j ≤ i
        # Example for T=4:
        # [[1 0 0 0],
        #  [1 1 0 0],
        #  [1 1 1 0],
        #  [1 1 1 1]]
        causal_mask = torch.tril(torch.ones(T, T)).to(x.device)
        
        # Apply causal mask to attention scores
        # 1. masked_fill puts -inf where mask is 0 (future positions)
        # 2. Shape of mask is (T,T) and broadcasts to (Batch, n_heads, T, T)
        # 3. -inf values will become 0 after softmax
        # This ensures each position cannot attend to future positions
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        # Convert attention scores to probabilities
        # Each query will have a probability distribution over all keys
        # -inf values become 0, maintaining causality
        attn = torch.softmax(attn, dim=-1)  # (Batch, n_heads, Time, Time)
        
        # Apply attention to V
        out = torch.matmul(attn, v)  # B, H, T, head_size
        
        # out shape is currently (Batch, n_heads, Time, head_size)
        # We need to:
        # 1. Move Time back to the middle: transpose(1,2) -> (Batch, Time, n_heads, head_size)
        # 2. Combine n_heads × head_size back into d_model: reshape -> (Batch, Time, d_model)
        #
        # This converts from "parallel heads" format back to the standard sequence format
        # that the rest of the model expects
        out = out.transpose(1, 2).reshape(B, T, D)  # B, T, D
        
        return self.out_proj(out)

class TinyTransformer(nn.Module):
    """
    A minimal transformer implementation for sequence modeling.
    Converts token indices to predictions of the next token.
    """
    def __init__(self, vocab_size, head_size=8, n_heads=4):
        super().__init__()
        # Total dimension = head_size * n_heads
        # Example: 8 * 4 = 32 dimensional model
        self.d_model = head_size * n_heads
        
        # Token embeddings and positional encoding
        # Input shape to embedding:  (batch, seq_len)        - indices
        # Output shape of embedding: (batch, seq_len, d_model) - dense vectors
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        
        # Positional encoding adds position information
        # Input & Output shape: (batch, seq_len, d_model)
        # Added to embeddings to preserve position information
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # Transformer blocks
        self.attention = MultiHeadAttention(head_size, n_heads)
        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)
        
        # Feed Forward network - applied to each position independently and in parallel
        # If input is (batch_size, seq_len, d_model):
        # 1. Each position's d_model vector goes through the same FF network
        # 2. No interaction between positions (unlike attention)
        # 3. PyTorch handles the parallelization automatically
        self.ff = nn.Sequential(
            # For each position:
            nn.Linear(self.d_model, self.d_model * 2),  # Expand to larger size d_model*2
            nn.ReLU(),                                  # Non-linearity
            nn.Linear(self.d_model * 2, self.d_model)   # Project back to d_model size, from d_model*2 to d_model
            # Each position's output is independent of all other positions
        )
        
        # Final projection to vocabulary
        # Converts model dimensions back to vocabulary probabilities
        # Input shape:  (batch, seq_len, d_model)   - e.g., (4, 10, 32)
        # Output shape: (batch, seq_len, vocab_size) - e.g., (4, 10, 20)
        # 
        # Each position gets a probability distribution over the entire vocabulary
        # These logits are typically passed through softmax during training
        self.to_logits = nn.Linear(self.d_model, vocab_size)
        
    def forward(self, x):
        # 1. Token embeddings
        x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # 2. Add positional encoding using our sinusoidal implementation
        x = self.pos_encoding(x)  # (batch, seq_len, d_model)
        
        # 3. Attention block with residual
        attended = self.attention(x)
        x = self.norm1(x + attended)
        
        # 4. FF block with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return self.to_logits(x)
    

def test_transformer():
    from dataset import AnimalSoundDataset
    
    # Load dataset and create model
    dataset = AnimalSoundDataset()
    model = TinyTransformer(vocab_size=len(dataset.vocab), head_size=8, n_heads=4)
    
    # Get test prompts
    test_prompts = dataset.get_test_prompts()
    
    # Convert prompts to tensor
    def encode(text):
        return torch.tensor([dataset.word2idx[word] for word in text.split()])
    
    # Test the model
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            # encode() returns shape: (seq_len,)
            # But our model expects: (batch_size, seq_len)
            # unsqueeze(0) adds a dimension at position 0, making it: (1, seq_len)
            # This is necessary because neural networks process data in batches
            # Even when we have just one sample, we need a batch dimension!
            x = encode(prompt).unsqueeze(0)  # Shape: (1, seq_len)
            
            # Generate two tokens
            for _ in range(2):
                # model(x) returns shape: (batch_size, seq_len, vocab_size)
                logits = model(x)
                
                # We only want the predictions for the last position
                # logits[0]: Select first (and only) item in batch
                # logits[0, -1]: Get the last position in the sequence
                # Shape: (vocab_size,) - a vector of scores for each possible next word
                last_token_logits = logits[0, -1]
                
                # Get top 5 most likely next tokens
                # top_k.values: tensor of the 5 highest scores
                # top_k.indices: tensor of the 5 corresponding token indices
                top_k = torch.topk(last_token_logits, k=5)
                
                # Take the most likely token (index 0 of top_k)
                # Shape: (1,)
                next_token_idx = top_k.indices[0]
                
                # We need to add this token to our input sequence
                # 1. unsqueeze(0): Add batch dim -> (1, 1)
                # 2. unsqueeze(0): Add sequence dim -> (1, 1)
                # 3. cat(..., dim=1): Concatenate along sequence dimension
                # Before cat: x shape is (1, current_seq_len)
                # After cat: x shape is (1, current_seq_len + 1)
                # because we're adding a new token to the end of the sequence
                x = torch.cat([x, next_token_idx.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Print results
            print(f"\nPrompt: {prompt}")
            print("Generated:", end=" ")
            for idx in x[0, len(prompt.split()):]:  # Skip prompt tokens
                print(dataset.vocab[idx.item()], end=" ")
            print("\n")
            
            # Also show top 5 alternatives for final position
            print("Top 5 alternatives for final position:")
            for score, idx in zip(top_k.values, top_k.indices):
                word = dataset.vocab[idx]
                print(f"{word}: {score:.2f}")

if __name__ == "__main__":
    test_transformer()
