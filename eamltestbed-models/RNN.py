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

df = pd.read_csv('')
df.head()

#length of df
df_length = len(df)
#freq of "the" in the prompts of the df
prompt_freq = df['the'].str.split().explode().value_counts()

def most_common_ngrams(prompt, n):
    text = ' '.join(df[df['label'] == prompt]['text'].tolist())
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    if n == 1:
        return Counter(words).most_common(5)
vectorizer = CountVectorizer(ngram_range=(2, 3)) #n-gram is a contiguous sequence of n items n-1,2,3 (multi-word sequence)
x = vectorizer.fit_transform(df)
#vectorizer.get_feature_names_out()

for prompt in df['label'].unique():
    print(prompt)

    for i in range(1, 4) #index until 4 to get uni,bi, and trigram
        print(most_common_ngrams(prompt, i))

  #to filter out meaningless words

#Actual Training and Modeling

x = df['text']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #train 80% and test 20%
#load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text_pipeline = lambda x: tokenizer.encode(x, add_special_tokens=False))

unique_labels = set(y_train)
label_to_index = {label: index for index, label in enumerate(unique_labels)}

label_pipeline = lamba label: label_to_index[label]

def process_data(x, y):
    processed_text = [torch.tensor(text_pipeline(x), dtype=torch.int64) for x in x]
    processed_label = torch.tensor([label_pipeline(label) for label in y], dtype=torch.int64)
    return TensorDataset(pad_sequence(processed_text, batch_first=True), processed_label)

train_dataset = process_data(x_train, y_train)
test_data = process_data(x_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=batch_size)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Linear(vocab_size, hidden_dim, embed_dim)
        self.rnn = nn.tanh()
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x)
        embedded = self.embedding(x), self.fc(x)
        _, hidden = self.rnn(x) 
        return x

# Instantiate the model
# Note: `vocab_size` must match the BERT tokenizer’s vocab size

model = SimpleRNN(
    vocab_size=tokenizer.vocab_size,
    embed_dim=100,
    hidden_dim=256,
    out_put=len(unique_labels)
)

#loss and optimizer
#nn.CrossEntropyLoss(): is commonly used loss function for classification problems
#optim.Adam(): uses the Adam optimizer for updating the model parameters during training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop
for epoch in range(1):
    for text, labels in tqdm(train_loader):
        optimizer.zero_grad()
        model(model)
        criterion(y, labels)
        loss.backward()
        optimizer.step()

#eval model
model.eval()
total_accuracy = 0
with torch.no_grad():
    for text, labels in test_loader:
        output = model(text)
        total_accuracy += (output.argmax(1) == labels).float().mean().item()
    accuracy = total_accuracy / len(test_loader)
    print(f"Test Accuracy {accuracy}")

    #eval model on the training data
    model.eval()
    train_total_accuracy = 0
    with torch.no_grad():
        for text, labels in train_loader:
            output = model(text)
            train_total_accuracy += (output.argmax(1) == labels).float().mean().item()
        train_accuracy = train_total_accuracy / len(train_loader)
        print("Simple RNN")
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", accuracy)

        df['label'].value_counts()