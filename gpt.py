import os
import requests
import numpy as np
import torch

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