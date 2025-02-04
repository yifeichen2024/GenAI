import numpy as np
from numpy import random
from numpy import dot
from scipy.special import softmax



# TODO: Implement Part 1 and Part 2 below.
# TODO: Find the Training Parameter class on line 294 and modify the parameters as needed. You may use test.ipynb to test your code.

# Do NOT modify the parameters or return statements of any function below. Doing so will result in a 0 on the assignment.
# Do NOT modify any code that is not part of the TODOs.
# Any code that is not part of the TODOs will be ignored.

################################################################################
# Part 0: What is your name?
################################################################################

def signature():
    name = "YifeiChen" # TODO: Put your first name and last name here
    return name

################################################################################
# Part 1: Implementing the Attention Mechanism with numpy
################################################################################

# You have learned that the attention mechanism is a way to compute the importance of each "word" in a sequence. 
# In this part of the assignment, you will implement the attention mechanism in the `Attention` class. 

# This is a toy attention mechanism. The outline of our attention mechanism is as follows:

# 1. You give it a sentence.
# 2. You will compute the word embeddings and positional embeddings for the sentence. This function has already been implemented for you. But when you run your attention mechanism you will need to first add the word embeddings and positional embeddings together first.
# 3. You will cast the embeddings to the dimensionality of the hidden layers.
# 4. Using the hidden embeddings and the weight matrices, you will compute the attention scores and the attention weighted embeddings.
# 5. You will report the attention scores.
# 6. You will report the value embeddings that have been weighted by the attention scores. 
# 7. The weighted value embeddings you report should be cast back to the dimensionality of the original word embeddings.

# The `Attention` class will have the following methods:

# - `__init__`: This method will initialize the class. It should define the Q, K, V weight matrices, and the maximum number of words in a sequence. Implement this method. 
# - `get_word_embeddings`: This method will return the embeddings for words that you want to compute the attention for. There is no need to implement this method. It has been implemented for you. 
# - `get_positional_embeddings`: This method will return the positional embeddings for words that you want to compute the attention for. Implement this method.
# - `get_attention`: This method will compute the attention mechanism. Implement this method. 

# Please see the instructions within each function for more details. 

# Hints 

# - The `softmax` function is imported from `scipy.special`. You may use this in the attention mechanism. 
# - Use the `@` operator to perform matrix multiplication. 
# - Think carefully about the dimensions of the matrices you are multiplying. 


class Attention:
    def __init__(self, seed: int = 42, 
                 max_words: int = 32, 
                 embedding_dim: int = 16,
    ):
        # DO NOT MODIFY ANYTHING BELOW THIS LINE. 
        '''
        Inputs:
            seed: The seed for the random number generator. Keep this as 42 for the purposes of this assignment. 
            max_words: The maximum number of words in a sequence. Keep this as 32 for the purposes of this assignment. 
            embedding_dim: The embedding dimension that words and positional embeddings will be initialized to. Keep this as 16 for the purposes of this assignment. 
            hidden_dim: The hidden dimension that we will perform attention computations on. Keep this as 8 for the purposes of this assignment. 
        '''
        
        rng = np.random.RandomState(seed=seed) 
        
        self.W_k = rng.uniform(size=(embedding_dim, embedding_dim))
        self.W_v = rng.uniform(size=(embedding_dim, embedding_dim))
        self.W_q = rng.uniform(size=(embedding_dim, embedding_dim))
        
        self.embedding_dim = embedding_dim
        
    def get_embeddings(self, words: str = None):
        '''
        This function will return a tuple of two np.ndarrays. 
        
        It has been implemented for you. 
        
        The first np.ndarray is the embeddings for words that you want to compute the attention for.
        The second np.ndarray is the positional embeddings for words that you want to compute the attention for.
        
        You will use the outputs of these two np.ndarrays elsewhere but you do not need to touch this function. 
        
        DO NOT TOUCH THIS FUNCTION.
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
        # DO NOT MODIFY ANYTHING BELOW THIS LINE.
        '''
        Inputs:
            `word_embeddings` is a np.ndarray of shape (n_words, embedding_dim)
            `positional_embeddings` is a np.ndarray of shape (n_words, embedding_dim)
            These two inputs should come from the `get_embeddings` method which has been implemented for you.
        
        Outputs:
            `attn_score` is a np.ndarray of shape (n_words, n_words). This should be softmax(QK^T / sqrt(d))
            `attn_weighted_embeddings` is a np.ndarray of shape (n_words, embedding_dim). This is the result of softmax(QK^T / sqrt(d)) * V and 
            then cast back to the embedding dimension.
        
        Use the self.words, self.W_Q, self.W_K, self.W_V matrices
        and return an attention for `words` as a np.ndarray.

        HINT: you may use the softmax import from scipy.
        HINT: We want to bring the word embeddings and positional embeddings to the dimension of the QKV matrices. 
        HINT: We want to bring the word embeddings and positional embeddings to the dimension of the QKV matrices. 
            Do this by left multiplying the word embeddings and positional embeddings by the QKV matrices.
        HINT: The word embeddings and positional embeddings should be added together to compute the attention. This is 
        done for you.
        '''

        input_embeddings = word_embeddings + positional_embeddings
        # DO NOT MODIFY ANYTHING ABOVE THIS LINE.
        # TODO: Implement the attention mechanism below this line.
        # Compute Query, Key and Value matrices.
        Q = input_embeddings @ self.W_q
        K = input_embeddings @ self.W_k
        V = input_embeddings @ self.W_v
        # Scaled dot-product attention score 
        scores = Q @ K.T / np.sqrt(self.embedding_dim) 
        attention_scores = softmax(scores, axis=-1)
        # Weight the value matrix by attention scores 
        attn_weighted_embeddings = attention_scores @ V

        # attention_scores = None        
        # attn_weighted_embeddings = None
        
        return attention_scores, attn_weighted_embeddings

################################################################################
# Part 2: Implementing a Transformer with Torch
# We want to use PyTorch to so we can use its autograd capabilities and actually train this model to do something. 
#
# Below is a class that has parts to be implemented. It is similar to a transformer with a single head attention mechanism.
# Read the `AttentionTransformer` class below.
#
# This transformer will be used as a classifier downstream. Specifically, we will be using the transformer to detect the number of times "cat" appears in a sentence. 
#
# If you are not familiar with PyTorch, you should refer to the PyTorch documentation to understand the methods that are being used. 
################################################################################


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
        
        ############################################################################ 
        # TODO: 
        # Our tokenizer returns our string as a list of integers. We need to convert these integers to embeddings. 
        # We will use the `nn.Embedding` layer to do this. 
        # The `nn.Embedding` layer takes in the number of embeddings we want to create and the dimension of each embedding. 
        # We want to create `vocab_size` embeddings of dimension `embedding_dim`. 
        ############################################################################ 
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim) # TODO: Implement this line
        
        ############################################################################ 
        # TODO:
        # Now we define the attention mechanism. 
        # Implement the QKV matrices with nn.Linear() and pay attention to the dimensions. 
        ############################################################################ 
        
        self.W_q = nn.Linear(self.embedding_dim, self.hidden_dim) # TODO: Implement this line
        self.W_k = nn.Linear(self.embedding_dim, self.hidden_dim) # TODO: Implement this line
        self.W_v = nn.Linear(self.embedding_dim, self.hidden_dim) # TODO: Implement this line
            
        ############################################################################ 
        # Now we define the feedforward network to make the classification  
        ############################################################################ 
        
        self.fc1 = nn.Linear(self.hidden_dim * self.max_input_length, self.embedding_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.embedding_dim, n_outputs+1)
        self.softmax = nn.Softmax(dim=-1)

    def word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        return embeddings
    
    def attention(self, embeddings: torch.Tensor) -> torch.Tensor:
        '''
        Inputs: 
            `embeddings` is a torch.Tensor of shape (batch_size, max_input_length, embedding_dim) 
        Outputs:  
            `attn_weighted_embeddings` is a torch.Tensor of shape (batch_size, max_input_length, embedding_dim) 
            
        HINT: You will need to use the QKV matrices to compute the attention scores and the attention weighted embeddings.
        HINT: You will need to use the softmax function to compute the attention scores.
        HINT: Note the dimension of embeddings is (batch_size, max_input_length, embedding_dim). In attention scoring,  
                we need to compute Q K^T. What does this mean for the dimensions of Q and K, and which dimensions are being transposed?
        '''
        
        # TODO: Implement the method below this line
        Q = self.W_q(embeddings)
        K = self.W_k(embeddings)
        V = self.W_v(embeddings)
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        attn_scores = torch.nn.functional.softmax(scores, dim=-1)
        # Apply scores to V
        attention_weighted_embeddings = torch.matmul(attn_scores, V)


        # attention_weighted_embeddings = None
        
        return attention_weighted_embeddings
    
    def feedforward(self, attention_inputs: torch.Tensor) -> torch.Tensor:
        attention_inputs = attention_inputs.view(attention_inputs.size(0), -1)
        return self.softmax(self.fc2(self.relu(self.fc1(attention_inputs))))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        attn_embeddings = self.attention(embeddings)
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
    learning_rate: float = 0.002 
    device: str = 'cpu'         # if you have an Nvidia GPU, set this to 'cuda' 
    hidden_dim: int = 32 # 32
    embedding_dim: int = 64 # 64
    batch_size: int = 32 # 32
    epochs: int = 100            # You may select any value between 1 and 500
    percent_train: float = 0.20    # You may select any value between 0 and 0.3
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