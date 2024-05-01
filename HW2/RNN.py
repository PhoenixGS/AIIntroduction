import torch
import gensim
import numpy as np
import sys

word2vec_path = './Dataset/wiki_word2vec_50.bin'
train_path = './Dataset/train.txt'
val_path = './Dataset/validation.txt'
test_path = './Dataset/test.txt'

max_len = 40

class CNN(torch.nn.Module):
    def __init__(self, embed_dim=50, filter_sizes=[3, 4, 5], num_filters=[10, 10, 10], num_classes=2, dropout=0.5):
        super(CNN, self).__init__()
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(in_channels=embed_dim, out_channels=num_filters[i], kernel_size=filter_sizes[i]) for i in range(len(filter_sizes))])
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(sum(num_filters), num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = [torch.nn.functional.relu(conv(x)) for conv in self.convs]
        x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class RNN(torch.nn.Module):
    def __init__(self, embed_dim=50, hidden_dim=100, num_layers=1, num_classes=2, dropout=0.5):
        super(RNN, self).__init__()
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

# load text to tensor
def load_text(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        inputs = []
        labels = []
        for line in lines:
            line = line.strip().split()
            label = int(line[0])
            sentence = line[1:]
            sentence = [word2vec[word] for word in sentence if word in word2vec]
            if len(sentence) < max_len:
                sentence += [np.zeros(50) for _ in range(max_len - len(sentence))]
            else:
                sentence = sentence[:max_len]
            inputs.append(sentence)
            labels.append(label)
    inputs = np.array(inputs)
    labels = np.array(labels)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def train(model, train_inputs, train_labels, val_inputs, val_labels, epochs=1000, batch_size=32, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(train_inputs), batch_size):
            optimizer.zero_grad()
            inputs = train_inputs[i:i+batch_size]
            labels = train_labels[i:i+batch_size]
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # print("Loss: %.4f" % loss)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            outputs = model(val_inputs)
            loss = loss_fn(outputs, val_labels)
            accuracy = torch.sum(torch.argmax(outputs, dim=1) == val_labels).item() / len(val_labels)
            print('Epoch: %d, Loss: %.4f, Accuracy: %.4f' % (epoch, loss.item(), accuracy))

if __name__ == '__main__':
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    train_inputs, train_labels = load_text(train_path)
    val_inputs, val_labels = load_text(val_path)
    test_inputs, test_labels = load_text(test_path)
    model = RNN()
    train(model, train_inputs, train_labels, val_inputs, val_labels)
    model.eval()
    with torch.no_grad():
        outputs = model(test_inputs)
        accuracy = torch.sum(torch.argmax(outputs, dim=1) == test_labels).item() / len(test_labels)
        print('Test Accuracy: %.4f' % accuracy)

