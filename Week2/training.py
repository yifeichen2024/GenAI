import torch
from torch.utils.data import DataLoader
from dataset import AnimalSoundDataset
from transformer import TinyTransformer
import torch.nn.functional as F
from tqdm import tqdm
import time  # Add this at the top


def train():
    # Device configuration
    # Note: For a production transformer, you'd typically want to use a GPU:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # For Mac users with M1/M2 chips, you could potentially use:
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #
    # But for our tiny dataset and model, CPU should be fine
    device = torch.device("cpu")
    
    # Initialize dataset and dataloader
    dataset = AnimalSoundDataset()
    print(f"Vocabulary size: {len(dataset.vocab)}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary: {dataset.vocab}")
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Small batch size for our tiny dataset
        shuffle=True
    )
    
    # Initialize model and move to device
    model = TinyTransformer(
        vocab_size=len(dataset.vocab),
        head_size=8,
        n_heads=4
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.1
    )
    
    # Training hyperparameters
    n_epochs = 60
    eval_every = 5  # Every 5 epochs
    best_loss = float('inf')
    
    # Training loop
    model.train()
    for epoch in tqdm(range(n_epochs), desc="Training"):
        total_loss = 0
        for batch_idx, x in enumerate(dataloader):
            # Move batch to device
            x = x.to(device)
            
            # Forward pass
            logits = model(x)  # shape: (batch, seq_len, vocab_size)
            
            # For language modeling, our targets are the input sequence shifted by 1
            # If input is ["the", "cow", "says", "moo"]:
            # - Input to model:  ["the", "cow", "says", "moo"]
            # - Target outputs: ["cow", "says", "moo", <ignored>]
            B, T = x.shape
            targets = x[:, 1:]                    # Remove first token from targets
            logits = logits[:, :-1, :]           # Remove last prediction
            
            # Compute cross entropy loss
            # 1. Reshape logits to (batch*seq_len, vocab_size)
            # 2. Reshape targets to (batch*seq_len)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # Slightly more efficient
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Move evaluation to epoch level
        if epoch % eval_every == 0:  # Check every 5 epochs instead of every batch
            avg_loss = total_loss / (batch_idx + 1)
            print(f"\nEpoch {epoch}, Loss: {avg_loss:.4f}")
            print("-" * 50)
            
            model.eval()
            with torch.no_grad():
                generate_sample(model, dataset, "the cow says")
                time.sleep(1)
            model.train()
            
            print("\n" + "=" * 50 + "\n")
            time.sleep(1.75)
            
            # Save if best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 'best_model.pt')

def generate_sample(model, dataset, prompt, max_new_tokens=2):
    """
    Generate tokens from a prompt using our trained model.
    
    Args:
        model: The transformer model
        dataset: AnimalSoundDataset for vocab conversion
        prompt: String like "the cow says"
        max_new_tokens: How many tokens to generate (default 2 for "woof woof" cases)
    """
    # Convert prompt string to tensor of indices
    tokens = prompt.split()
    x = torch.tensor([dataset.word2idx[w] for w in tokens]).unsqueeze(0)  # Add batch dim
    
    # Generate tokens
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get predictions
            logits = model(x)
            
            # Focus on last token's predictions
            next_token_logits = logits[0, -1, :]  # Remove batch dim, get last position
            
            # Get top 5 predictions for logging
            top_k = torch.topk(next_token_logits, k=5)
            
            # Add most likely token to sequence
            next_token_idx = top_k.indices[0]
            x = torch.cat([x, next_token_idx.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # Print results
        print(f"\nPrompt: {prompt}")
        print("Generated:", end=" ")
        for idx in x[0, len(tokens):]:  # Skip prompt tokens
            print(dataset.vocab[idx.item()], end=" ")
        print()
        
        # Show top 5 alternatives for final position
        print("\nTop 5 alternatives for final position:")
        for score, idx in zip(top_k.values, top_k.indices):
            word = dataset.vocab[idx]
            print(f"{word}: {score:.2f}")
        print()

def interact_with_model(model_path, dataset):
    """
    Load a trained model and let users input prompts interactively.
    
    Args:
        model_path: Path to saved model weights
        dataset: AnimalSoundDataset for vocabulary
    """
    # Load model
    model = TinyTransformer(
        vocab_size=len(dataset.vocab),
        head_size=8,
        n_heads=4
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("\nTiny Transformer loaded! Enter prompts like 'the cow says' or 'q' to quit.")
    
    while True:
        # Get input from user
        prompt = input("\nEnter prompt: ").strip()
        if prompt.lower() == 'q':
            break
            
        # Generate from prompt
        generate_sample(model, dataset, prompt)

if __name__ == "__main__":
    # If we want to train:
    train()
    
    # If we want to interact with trained model:
    #dataset = AnimalSoundDataset()
    #interact_with_model('best_model.pt', dataset)
