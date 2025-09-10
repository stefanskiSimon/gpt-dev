import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

#Parameters
batch_size = 64 #how many independent sequences will we process in parallel?
block_size = 256 #what is the maximum context length for predictions?
eval_interval = 500 #how often to evaluate the model?
eval_iters = 200 #how many batches to use for evaluation?
learning_rate = 1e-2 #learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu' #use GPU if available
max_iters = 5000 #number of training iterations

# --- data ---
#loading the dataset, by creating/checking the input.txt file on current directory (path)

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

#If the file does not exist create it and overwrite it with the data from the URL

    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

#Get the length of the dataset in characters

with open(input_file_path, 'r') as f:
    text = f.read()
print(f"length of dataset in characters: {len(text):,}")

#Unique characters in dataset

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(' '.join(chars))
# print(f"vocab size: {vocab_size}")

# --- Mapping characters to integers and vice versa ---
stoi = { ch:i for i,ch in enumerate(chars) } # string to integer
itos = { i:ch for i,ch in enumerate(chars) } # integer to string
# def encode(s: str):
#     #Encode a string into a list of integers.
#     return [stoi[c] for c in s]
# def decode(l):
#     #Decode a list of integers into a string.
#     return ''.join([itos[i] for i in l])
encode = lambda s: [stoi[c] for c in s] #encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
train_length = int(0.9* len(data))
train_data = data[:train_length]
val_data = data[train_length:]

# --- data loading ---
def get_batch(split):
    #Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() #Everything that happens in this function won't be called backward (more efficient)
def estimate_loss():
    #Estimate the loss on the training and validation sets
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Bigram model ---

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self(idx)
            #focus only on the last time step
            logits = logits[:, -1, :] #(B,C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# --- training ---
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- generate text ---
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))