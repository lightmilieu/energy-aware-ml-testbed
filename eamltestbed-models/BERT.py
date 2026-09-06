import pandas as pd
import random
import time
import numpy as np
import nltk #nlp library
from collection import Counter
from nltk import bigrams, trigrams
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer.from_pretrained("bert-based-uncased"), BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from nltk.tokenize import_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt_tab')
#device for GPU storage
device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

#Bidirectional Encoder Representations from Transformers. Language model built on transformer architecture. 
#BERT can read language in both directions enabled by the transformer architecture.
#BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#function to encode the data
def encode_data(tokenizer, comments, max_len):
    input_ids = []
    attention_masks = []

    for comment in comments:
        encoded_comment = tokenizer(
            text=comment, #preprocesses comment
            add_special_tokens=True #Add '[CLS]' and '[SEP]' *******
            max_length=max_len, #max length to truncate/pad
            truncation=True, #apply truncation according to max_length
            padding="max_length", #apply padding to max_length
            return_attention_mask=True, #return attention mask
            return_tensors='pt', #return pytorch tensors
    )

    input_ids.append(encoded_comment['input_ids'])
    attention_masks.append(encoded_comment['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

#Encoding data
max_len = 256 #can be adjusted!!!

X_train_encoded, X_train_attention = encode_data(tokenize, X_train.to_numpy, max_len)
X_test_encoded, X_test_attention = encode_data(tokenizer, X_test.to_numpy, max_len)

#initialize the label encoder
label_encoder = LabelEncoder()

#fit the encoder on the prompt labels
label_encoder.fit(df['label'])

#Tranform the prompt labels to integers
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

#convert the encoded labels to torch tensors
y_train = torch.tensor(y_train_encoded)
y_test = torch.tensor(y_test_encoded)

#create DataLoader
batch_size = 32 #can be adjusted!!!

#train
train_data = TensorDataset(X_train_encoded, X_train_attention, y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

#test
test_data = TensorDataset(X_test_encoded, X_test_attention, y_test)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=train_sampler, batch_size=batch_size)

#Load BertForSequenceClassification, a BERT model with a single linaer classification layer on top
#A linear classifier makes a classification decision by calculating a linear combination of iinput features
#It separates different categories or classes by drawing a straight decision boundary (line, plane, hyperplane) through the data space.
#A linear classifier assigns a weight (w) to each input feature (x) and adds a bias (b). It calculates a total score using a linear equation

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased', 
    num_labels = len(df['label'].unique())
    output_attentions = False,
    output_hidden_states = False,
)
model.to(device)
#use the 12 layer BERT "
#number of output labels
#whether the model returns attentions weights
#whether the model returns all hidden-states

#optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 1 #ADJUSTABLE !!!
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

#Training Loop
for epoch_i in tqdm(range(0, epochs)):
    #training
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        #avg
        avg_train_loss = total_loss / len(train_dataloader)

#evaluation
def evaluate(dataloader):
    model.eval()
    total_eval_accuracy = 0
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    return avg_val_accuracy

train_accuracy = evaluate(train_dataloader)
test_accuracy = evaluate(test_dataloader)


