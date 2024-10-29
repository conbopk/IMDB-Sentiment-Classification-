import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# load & clean data
df = pd.read_csv('Data/IMDB Dataset.csv')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def processing_text(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return ' '.join(lemmatizer_tokens)


df['processed_review'] = df['review'].apply(processing_text)


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in word_tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = word_tokenize(text)

        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenized_text]


#Dataset class
class ReviewDataset(Dataset):
    def __init__(self, x, y, vocab):
        self.x = x
        self.y = y
        self.vocab = vocab

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        text = self.x[index]
        label = self.y[index]
        numericalized_text = [self.vocab.stoi['<SOS>']] + self.vocab.numericalize(text) + [self.vocab.stoi['<EOS>']]
        return torch.tensor(numericalized_text), torch.tensor(label)


#Function to create a padded batch
def collate_fn(batch):
    batch_text = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]
    batch_text = pad_sequence(batch_text, batch_first=True, padding_value=0)
    batch_labels = torch.stack(batch_labels)
    return batch_text, batch_labels

#define the model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        lstm_out = lstm_out[:,-1,:]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sigmoid(out)
        return sig_out

x = df['processed_review'].values
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#build vocabulary
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(x_train)

#Create datasets
train_data = ReviewDataset(x_train, y_train, vocab)
test_data = ReviewDataset(x_test, y_test, vocab)

#create dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

#define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Instantiate the model
vocab_size = len(vocab.stoi)
embedding_dim = 400
hidden_dim = 256
output_dim = 1
n_layer = 2


model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layer).to(device)

#loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
num_epochs = 10
best_val_f1 = 0
for epoch in range(num_epochs):
    model.train()
    total_loss=0
    for texts, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss/len(train_loader)
    print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.4f}')

    #Evaluation
    model.eval()
    val_predictions = []
    val_targets = []
    val_loss = 0
    with torch.no_grad():
        for texts, labels in tqdm(test_loader, desc='Validating'):
            texts, labels = texts.to(device), labels.to(device)
            output = model(texts)
            loss = criterion(output.squeeze(), labels.float())
            val_loss += loss.item()
            predictions = (output.squeeze() > 0.5).int()
            val_predictions.extend(predictions.cpu().tolist())
            val_targets.extend(labels.cpu().tolist())

    avg_val_loss = val_loss/len(test_loader)
    val_accuracy = accuracy_score(val_targets, val_predictions)
    val_f1 = f1_score(val_targets, val_predictions)

    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}')

    #Save the best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'models/best_model.pth')
        print('Best model saved!')

print('Training complete!')









