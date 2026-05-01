import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.io as io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

path = '/kaggle/input/datasets/subinium/emojiimage-dataset'
csv_file = 'full_emoji.csv'
Google_Path = os.path.join(path, 'image/Google')

df = pd.read_csv(os.path.join(path, csv_file))
df.head()
class Emoji_dataset(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.path = path
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        meta = self.df.iloc[idx]
        filename = f'{meta['#']}.png'
        #label_class = torch.tensor([meta['#']])
        label_class = torch.tensor(meta['#']-1, dtype=torch.long)
        #label_class = F.one_hot(torch.tensor([meta['#']]), num_classes=len(self.df)).squeeze(0).to(torch.float32)
        name = meta['name']
        image_tensor = io.read_image(os.path.join(self.path, filename), mode="RGB").to(torch.float32)
        image_tensor = image_tensor/255
        return image_tensor,label_class, name
        
Emojis = Emoji_dataset(df, Google_Path)

trainloader = torch.utils.data.DataLoader(Emojis, batch_size=8, shuffle=True)
