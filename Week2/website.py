from flask import Flask, render_template, request, jsonify
from transformer import TinyTransformer
from dataset import AnimalSoundDataset
import torch

app = Flask(__name__)

# Load model and dataset globally (will be shared between requests)
dataset = AnimalSoundDataset()
model = TinyTransformer(
    vocab_size=len(dataset.vocab),
    head_size=8,
    n_heads=4
)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Convert prompt to lowercase before processing
    prompt = request.json['prompt'].lower()
    
    # Convert prompt to tensor
    tokens = prompt.split()
    x = torch.tensor([dataset.word2idx[w] for w in tokens]).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        logits = model(x)
        next_token_logits = logits[0, -1, :]
        top_k = torch.topk(next_token_logits, k=5)
        
        # Get predictions
        predictions = []
        for score, idx in zip(top_k.values, top_k.indices):
            word = dataset.vocab[idx]
            predictions.append({
                'word': word,
                'score': f"{score:.2f}"
            })
    
    return jsonify({
        'predictions': predictions
    })

if __name__ == '__main__':
    app.run(debug=True)