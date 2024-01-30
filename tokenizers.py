import torch
from pytorch_pretrained_bert import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class BertData(Dataset):
    def __init__(self, X, y, word_max_len):
        super().__init__()
        
        self.X = X
        self.y = y
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t) + ['[SEP]'], self.X))
        self.tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, tokens)), 
                                   maxlen=word_max_len, truncating="post", padding="post", dtype="int")
        self.masks = [[float(i > 0) for i in ii] for ii in self.tokens_ids]
        
        print('Token ids size:', self.tokens_ids.shape)
        print('Masks size:', np.array(self.masks).shape)
        print('y size:', np.array(self.y).shape)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        tokens_id = self.tokens_ids[ind]
        label = self.y[ind]
        mask = self.masks[ind]
        return tokens_id, label, mask
    
    def collate_fn(self, data):
        tokens_ids, labels, masks = zip(*data)
        tokens_ids = torch.tensor(tokens_ids).to(device)
        labels = torch.tensor(labels).float().to(device)
        masks = torch.tensor(masks).to(device)
        return tokens_ids, labels, masks
    
    def choose(self):
        return self[np.random.randint(len(self))]
class RoBERTaData(Dataset):
    def __init__(self,X,y,word_max_len):
        self.X = X;
        self.y = y;
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t) + ['[SEP]'], self.X))
        self.tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, tokens)), 
                                   maxlen=word_max_len, truncating="post", padding="post", dtype="int")
        self.masks = [[float(i > 0) for i in ii] for ii in self.tokens_ids]
        
        print('Token ids size:', self.tokens_ids.shape)
        print('Masks size:', np.array(self.masks).shape)
        print('y size:', np.array(self.y).shape)
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        tokens_id = self.tokens_ids[ind]
        label = self.y[ind]
        mask = self.masks[ind]
        return tokens_id, label, mask
    
    def collate_fn(self, data):
        tokens_ids, labels, masks = zip(*data)
        tokens_ids = torch.tensor(tokens_ids).to(device)
        labels = torch.tensor(labels).float().to(device)
        masks = torch.tensor(masks).to(device)
        return tokens_ids, labels, masks
    
    def choose(self):
        return self[np.random.randint(len(self))]
class DistilBertData(Dataset):
    def __init__(self,X,y,word_max_len):
        self.X = X;
        self.y = y;
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t) + ['[SEP]'], self.X))
        self.tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, tokens)), 
                                   maxlen=word_max_len, truncating="post", padding="post", dtype="int")
        self.masks = [[float(i > 0) for i in ii] for ii in self.tokens_ids]
        
        print('Token ids size:', self.tokens_ids.shape)
        print('Masks size:', np.array(self.masks).shape)
        print('y size:', np.array(self.y).shape)
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        tokens_id = self.tokens_ids[ind]
        label = self.y[ind]
        mask = self.masks[ind]
        return tokens_id, label, mask
    
    def collate_fn(self, data):
        tokens_ids, labels, masks = zip(*data)
        tokens_ids = torch.tensor(tokens_ids).to(device)
        labels = torch.tensor(labels).float().to(device)
        masks = torch.tensor(masks).to(device)
        return tokens_ids, labels, masks
    
    def choose(self):
        return self[np.random.randint(len(self))]