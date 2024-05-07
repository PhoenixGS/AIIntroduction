import torch
import gensim
import numpy as np
import sys

from CNN import CNN
from RNN import RNN
from MLP import MLP

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

def train(model, train_inputs, train_labels, val_inputs, val_labels, epochs=100, batch_size=64, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(train_inputs), batch_size):
            optimizer.zero_grad()
            inputs = train_inputs[i:i+batch_size]
            labels = train_labels[i:i+batch_size]
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # print("Loss: %.4f" % loss)
            loss.backward()
            optimizer.step()
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
                loss_sum += loss.item()
                accuracy_sum += accuracy

            loss_avg = loss_sum / (len(val_inputs) / batch_size)
            accuracy_avg = accuracy_sum / (len(val_inputs) / batch_size)
            print('Epoch: %d, Loss: %.4f, Accuracy: %.4f' % (epoch, loss_avg, accuracy_avg))

            # outputs = model(val_inputs)
            # loss = loss_fn(outputs, val_labels)
            # accuracy = torch.sum(torch.argmax(outputs, dim=1) == val_labels).item() / len(val_labels)
            # print('Epoch: %d, Loss: %.4f, Accuracy: %.4f' % (epoch, loss.item(), accuracy))

def evaluate(model, test_inputs, test_labels, batch_size=64):
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        accuracy_sum = 0
        for i in range(0, len(test_inputs), batch_size):
            test_inputs_batch = test_inputs[i:i+batch_size]
            test_labels_batch = test_labels[i:i+batch_size]
            test_inputs_batch = test_inputs_batch.to(device)
            test_labels_batch = test_labels_batch.to(device)
            
            outputs = model(test_inputs_batch)
            loss = loss_fn(outputs, test_labels_batch)
            accuracy = torch.sum(torch.argmax(outputs, dim=1) == test_labels_batch).item() / len(test_labels_batch)
            loss_sum += loss.item()
            accuracy_sum += accuracy

        loss_avg = loss_sum / (len(test_inputs) / batch_size)
        accuracy_avg = accuracy_sum / (len(test_inputs) / batch_size)
        print('Test Loss: %.4f, Test Accuracy: %.4f' % (loss_avg, accuracy_avg))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    train_inputs, train_labels = load_text(train_path)
    val_inputs, val_labels = load_text(val_path)
    test_inputs, test_labels = load_text(test_path)
    model = CNN().to(device)
    train(model, train_inputs, train_labels, val_inputs, val_labels)
    model.eval()
    evaluate(model, test_inputs, test_labels)
    # with torch.no_grad():
    #     outputs = model(test_inputs)
    #     accuracy = torch.sum(torch.argmax(outputs, dim=1) == test_labels).item() / len(test_labels)
    #     precision = torch.sum(torch.argmax(outputs, dim=1) == test_labels).item() / (torch.sum(torch.argmax(outputs, dim=1) == test_labels).item() + torch.sum(torch.argmax(outputs, dim=1) != test_labels).item())
    #     recall = torch.sum(torch.argmax(outputs, dim=1) == test_labels).item() / (torch.sum(torch.argmax(outputs, dim=1) == test_labels).item() + torch.sum(torch.argmax(outputs, dim=1) != test_labels).item())
    #     f1 = 2 * (precision * recall) / (precision + recall)
    #     print('Test Accuracy: %.4f' % accuracy)
    #     print('Test Precision: %.4f' % precision)
    #     print('Test Recall: %.4f' % recall)

