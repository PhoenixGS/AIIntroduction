import torch
import gensim
import numpy as np
import sys

class MLP(torch.nn.Module):
    def __init__(self, embed_dim=50, max_len=40, hidden_dim=100, num_classes=2, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(embed_dim * max_len, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLP2(torch.nn.Module):
    def __init__(self, embed_dim=50, hidden_dim=100, num_classes=2, dropout=0.5):
        super(MLP2, self).__init__()
        self.fc1 = torch.nn.Linear(embed_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.fc2(x)
        return x


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
    def __init__(self, embed_dim=50, hidden_dim=100, num_layers=2, num_classes=2, dropout=0.5):
        super(RNN, self).__init__()
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

class RNN2(torch.nn.Module):
    def __init__(self, embed_dim=50, hidden_dim=50, num_layers=2, num_classes=2, dropout=0.5):
        super(RNN2, self).__init__()
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = torch.nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)
        x = self.fc(x)
        return x