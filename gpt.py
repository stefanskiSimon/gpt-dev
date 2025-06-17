import os
import requests

#loading the dataset, by creating/checking the input.txt file on current directory (path)

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

#If the file does not exist create it and overwrite it with the data from the URL

    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

#Get the length of the dataset in characters

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

#Unique characters in dataset

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(' '.join(chars))
print(f"vocab size: {vocab_size}")

#Mapping characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) } # string to integer
itos = { i:ch for i,ch in enumerate(chars) } # integer to string
def encode(s: str):
    #Encode a string into a list of integers.
    return [stoi[c] for c in s]
def decode(l):
    #Decode a list of integers into a string.
    return ''.join([itos[i] for i in l])

train_length = int(0.9* len(data))
train_data = data[:train_length]
val_data = data[train_length:]

#Splitting dataset into training and validation sets <90% for training, 10% for validation>

train_data = encode(train_data)
val_data = encode(val_data)
print(f"train data length: {len(train_data):,} characters")
print(f"validation data length: {len(val_data):,} characters")

# batch_size = 4 #number of sequences to be processed in parallel
# block_size = 8 #maximum context length for predictions

# def get_batch(split):
#     #Generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([torch.tensor(encode(data[i:i+block_size])) for i in ix])
#     y = torch.stack([torch.tensor(encode(data[i+1:i+block_size+1])) for i in ix])
#     return x, y
# xb, yb = get_batch('train')

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"input: {decode(context.tolist())}, target: {itos[target.item()]}")