import numpy as np
from numpy import random
from numpy import dot
from scipy.special import softmax

def signature():
    name = 'YifeiChen' # TODO: Put your first name and last name here
    return name
 
class Attention:
    def __init__(self, seed: int = 42, 
                 max_words: int = 32, 
                 embedding_dim: int = 16,
    ):
        '''
        Inputs:
            seed: The seed for the random number generator. Keep this as 42 for the purposes of this assignment. 
            max_words: The maximum number of words in a sequence. Keep this as 32 for the purposes of this assignment. 
            embedding_dim: The embedding dimension that words and positional embeddings will be initialized to. Keep this as 16 for the purposes of this assignment. 
        '''
        rng = np.random.RandomState(seed=seed)
        self.W_k = rng.uniform(size=(embedding_dim, embedding_dim))  # Key weight matrix
        self.W_v = rng.uniform(size=(embedding_dim, embedding_dim))  # Value weight matrix
        self.W_q = rng.uniform(size=(embedding_dim, embedding_dim))  # Query weight matrix
        
        self.embedding_dim = embedding_dim

    def get_embeddings(self, words: str = None):
        '''
        This function will return a tuple of two np.ndarrays. 
        
        It has been implemented for you.
        '''
        if words is None:
            raise ValueError("words cannot be None")
        prime = 31
        modulus = 10**9 + 7
        hash_value = 0
        for i, char in enumerate(words):
            hash_value = (hash_value + ord(char) * pow(prime, i, modulus)) % modulus
            
        words = words.split()
        rng_embeddings = np.random.RandomState(seed=hash_value)
        embeddings = rng_embeddings.rand(len(words), self.embedding_dim)
        positional_embeddings = rng_embeddings.rand(len(words), self.embedding_dim)
        
        return embeddings, positional_embeddings

    def get_attention(self, word_embeddings: np.ndarray, positional_embeddings: np.ndarray) -> np.ndarray:
        '''
        Compute the attention mechanism.
        
        Inputs:
            `word_embeddings`: np.ndarray of shape (n_words, embedding_dim)
            `positional_embeddings`: np.ndarray of shape (n_words, embedding_dim)
        
        Outputs:
            `attn_score`: np.ndarray of shape (n_words, n_words)
            `attn_weighted_embeddings`: np.ndarray of shape (n_words, embedding_dim)
        '''
        input_embeddings = word_embeddings + positional_embeddings  # Add word and positional embeddings
        
        # Compute Q, K, V matrices
        Q = input_embeddings @ self.W_q.T
        K = input_embeddings @ self.W_k.T
        V = input_embeddings @ self.W_v.T

        # Compute attention scores (QK^T / sqrt(d))
        d_k = Q.shape[1]  # Hidden dimension
        attention_scores = Q @ K.T / np.sqrt(d_k)

        # Apply softmax to normalize attention scores
        attention_weights = softmax(attention_scores, axis=1)

        # Compute attention-weighted embeddings
        attn_weighted_embeddings = attention_weights @ V

        # Cast back to original embedding dimension
        attn_weighted_embeddings = attn_weighted_embeddings @ self.W_v

        return attention_weights, attn_weighted_embeddings

# part 2

import torch.nn as nn
import torch

def seed_torch(seed=0):
    # DO NOT MODIFY THIS FUNCTION.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AttentionTransformer(nn.Module):
    def __init__(self, max_input_length: int = 50, 
                 embedding_dim: int = 16, 
                 hidden_dim: int = 8, 
                 vocab_size: int = 30522,
                 n_outputs: int = 10,
                 device: str = 'cpu'):
        super(AttentionTransformer, self).__init__()
        
        self.max_input_length = max_input_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        seed_torch(seed=42)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Attention mechanism (Q, K, V matrices)
        self.W_q = nn.Linear(embedding_dim, hidden_dim)  # Query transformation
        self.W_k = nn.Linear(embedding_dim, hidden_dim)  # Key transformation
        self.W_v = nn.Linear(embedding_dim, hidden_dim)  # Value transformation
            
        # Feedforward layers for classification
        self.fc1 = nn.Linear(self.hidden_dim * max_input_length, embedding_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim, n_outputs + 1)
        self.softmax = nn.Softmax(dim=-1)

    def word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)
    
    def attention(self, embeddings: torch.Tensor) -> torch.Tensor:
        '''
        Compute attention mechanism.
        
        Inputs: 
            embeddings (torch.Tensor): Shape (batch_size, max_input_length, embedding_dim)
            
        Outputs:  
            attn_weighted_embeddings (torch.Tensor): Shape (batch_size, max_input_length, hidden_dim)
        '''
        # Compute Q, K, V matrices
        Q = self.W_q(embeddings)  # Shape: (batch_size, max_input_length, hidden_dim)
        K = self.W_k(embeddings)  # Shape: (batch_size, max_input_length, hidden_dim)
        V = self.W_v(embeddings)  # Shape: (batch_size, max_input_length, hidden_dim)

        # Compute attention scores: Q @ K^T / sqrt(hidden_dim)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))

        # Apply softmax to get normalized attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Compute attention-weighted embeddings
        attn_weighted_embeddings = torch.matmul(attention_weights, V)

        return attn_weighted_embeddings
    
    def feedforward(self, attention_inputs: torch.Tensor) -> torch.Tensor:
        # Flatten input for fully connected layers
        attention_inputs = attention_inputs.view(attention_inputs.size(0), -1)
        return self.softmax(self.fc2(self.relu(self.fc1(attention_inputs))))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Compute embeddings
        embeddings = self.word_embeddings(input_ids)
        
        # Apply attention mechanism
        attn_embeddings = self.attention(embeddings)
        
        # Pass through feedforward network
        output = self.feedforward(attn_embeddings)
        
        return output

from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
import dataclasses

@dataclasses.dataclass
class TrainingParameters:
    # TODO: Modify the parameters below as needed to train your transformer.
    learning_rate: float = 0.4 
    device: str = 'cpu'         # if you have an Nvidia GPU, set this to 'cuda' 
    hidden_dim: int = 5
    embedding_dim: int = 10
    batch_size: int = 5
    epochs: int = 10             # You may select any value between 1 and 500
    percent_train: float = 0.2    # You may select any value between 0 and 0.3
    token_max_length: int = 90    # You may select any value between 1 and 123
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # do not modify this line

# DO NOT MODIFY ANYTHING BELOW THIS LINE. 
# There is nothing left in the file to modify beyond this point.

def load_data(data_path: str, params: TrainingParameters):
    df = load_dataset('csv', data_files=data_path)
    df = df['train'].train_test_split(test_size=1-params.percent_train)

    df['train_tokenized'] = df['train'].map(lambda x: params.tokenizer(x['Sentence'], return_tensors='pt', 
                                                                padding="max_length", 
                                                                truncation=True, 
                                                                max_length=params.token_max_length).to(params.device))
    df['test_tokenized'] = df['test'].map(lambda x: params.tokenizer(x['Sentence'], return_tensors='pt', 
                                                              padding="max_length", 
                                                              truncation=True, 
                                                              max_length=params.token_max_length).to(params.device))
    df = df.with_format("torch")

    data_train = DataLoader(list(zip(df['train_tokenized']['input_ids'], df['train']['count'])), batch_size=params.batch_size, shuffle=True)
    data_test = DataLoader(list(zip(df['test_tokenized']['input_ids'], df['test']['count'])), batch_size=params.batch_size, shuffle=True)

    return data_train, data_test


def train(params: TrainingParameters, data_path: str):
    data_train, data_test = load_data(data_path, params)
    model = AttentionTransformer(max_input_length=params.token_max_length, 
                                 embedding_dim=params.embedding_dim, 
                                 hidden_dim=params.hidden_dim, 
                                 device=params.device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=params.learning_rate)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    eval_score_history = []

    for _ in range(params.epochs):
        epoch_loss = 0
        for batch in data_train:
            input_ids, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids.squeeze())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(data_train))
        
        eval_score = 0
        for batch in data_test:
            input_ids, labels = batch
            outputs = model(input_ids.squeeze())
            choice = torch.argmax(outputs, dim=-1)
            score = (choice == labels).sum().item() / len(labels)
            eval_score += score
        eval_score_history.append(eval_score / len(data_test))
        
        if _ % 10 == 0:
            print(f"Epoch {_} train loss: {loss_history[-1]}")
            print(f"Epoch {_} eval accuracy: {eval_score_history[-1]}")
    
    print(loss_history[-1])
    print(eval_score_history[-1])
    return loss_history, eval_score_history, model