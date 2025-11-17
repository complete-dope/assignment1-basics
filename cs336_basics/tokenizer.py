# in this file lets build out own BPE tokenizer

import torch
import torch.nn as nn 
from datasets import load_dataset

DATASET_NAME = "roneneldan/TinyStories"

dataset = load_dataset(DATASET_NAME, streaming =True, split = 'train')

print(dataset)