import torch
import gensim
import numpy as np
import sys
import argparse

from models import *

word2vec_path = './Dataset/wiki_word2vec_50.bin'
train_path = './Dataset/train.txt'
val_path = './Dataset/validation.txt'
test_path = './Dataset/test.txt'

max_len = 40

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

def train(model, train_inputs, train_labels, val_inputs, val_labels, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        accuracy_sum = 0
        for i in range(0, len(train_inputs), batch_size):
            optimizer.zero_grad()
            inputs = train_inputs[i:i+batch_size]
            labels = train_labels[i:i+batch_size]
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            accuracy = torch.sum(torch.argmax(outputs, dim=1) == labels).item() / len(labels)
            loss_sum += loss.item() * len(labels)
            accuracy_sum += accuracy * len(labels)
            # print("Loss: %.4f" % loss)
            loss.backward()
            optimizer.step()
        loss_avg = loss_sum / len(train_inputs)
        accuracy_avg = accuracy_sum / len(train_inputs)
        print('Epoch: %d, Loss: %.4f, Accuracy: %.4f' % (epoch, loss_avg, accuracy_avg))
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            accuracy_sum = 0
            for i in range(0, len(val_inputs), batch_size):
                val_inputs_batch = val_inputs[i:i+batch_size]
                val_labels_batch = val_labels[i:i+batch_size]
                val_inputs_batch = val_inputs_batch.to(device)
                val_labels_batch = val_labels_batch.to(device)
                
                outputs = model(val_inputs_batch)
                loss = loss_fn(outputs, val_labels_batch)
                accuracy = torch.sum(torch.argmax(outputs, dim=1) == val_labels_batch).item() / len(val_labels_batch)
                loss_sum += loss.item() * len(val_labels_batch)
                accuracy_sum += accuracy * len(val_labels_batch)

            loss_avg = loss_sum / len(val_inputs)
            accuracy_avg = accuracy_sum / len(val_inputs)
            print('Epoch: %d, Loss: %.4f, Accuracy: %.4f' % (epoch, loss_avg, accuracy_avg))

def evaluate(model, test_inputs, test_labels, batch_size):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(0, len(test_inputs), batch_size):
            test_inputs_batch = test_inputs[i:i+batch_size]
            test_labels_batch = test_labels[i:i+batch_size]
            test_inputs_batch = test_inputs_batch.to(device)
            test_labels_batch = test_labels_batch.to(device)
            
            outputs = model(test_inputs_batch)
            loss = loss_fn(outputs, test_labels_batch)

            labels = torch.argmax(outputs, dim=1)
            TP += torch.sum((labels == 1) & (test_labels_batch == 1)).item()
            TN += torch.sum((labels == 0) & (test_labels_batch == 0)).item()
            FP += torch.sum((labels == 1) & (test_labels_batch == 0)).item()
            FN += torch.sum((labels == 0) & (test_labels_batch == 1)).item()
        
        print('TP: %d, TN: %d, FP: %d, FN: %d' % (TP, TN, FP, FN))
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print('Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f' % (accuracy, precision, recall, f1))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RNN')
    parser.add_argument('--epochs', type=int, default=50,)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    train_inputs, train_labels = load_text(train_path)
    val_inputs, val_labels = load_text(val_path)
    test_inputs, test_labels = load_text(test_path)

    args = get_args()

    model = None
    if args.model == 'RNN':
        model = RNN().to(device)
    elif args.model == 'RNN2':
        model = RNN2().to(device)
    elif args.model == 'CNN':
        model = CNN().to(device)
    elif args.model == 'MLP':
        model = MLP().to(device)
    elif args.model == 'MLP2':
        model = MLP2().to(device)

    train(model, train_inputs, train_labels, val_inputs, val_labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    evaluate(model, test_inputs, test_labels, batch_size=args.batch_size)
