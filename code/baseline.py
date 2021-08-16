import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import spacy
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from EncoderCNN import EncoderCNN
from Embedding import Embedding

if torch.cuda.is_available():
    spacy.prefer_gpu()
spacy_nlp = spacy.load('en_core_web_sm')
glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.word2vec.txt'
TRAINFILE = "nyt10_train"
TESTFILE = "nyt10_test"

word_embedding_dim = 50
pos_embedding_dim = 5
HIDDEN_DIM = 300
batch_size = 4
num_workers = 0
shuffle = True
n_classes = 20
epoch = 500
max_length = 500
n_layers = 2
bidirectional = True
weight_decay = 1e-5
learning_rate = 1e-2
embedding_dim = word_embedding_dim + pos_embedding_dim*2

#glove2word2vec(glove_input_file, word2vec_output_file)
X_train1 = []
Y_train1 = []
X_test1 = []
Y_test1 = []
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


def get_model_acc(model, loader):
    ys = []
    y_preds = []
    for x, y in loader:
        ys.append(y.cuda())
        y_preds.append(torch.argmax(model((x.permute(0,2,1)).cuda()), dim=1))
    y = torch.cat(ys, dim=0)
    y_pred = torch.cat(y_preds, dim=0)
    accuracy = accuracy_score(y.cpu(), y_pred.cpu())
    recall = recall_score(y.cpu(), y_pred.cpu(), average='weighted')
    precision = precision_score(y.cpu(), y_pred.cpu(), average='weighted')
    f1 = f1_score(y.cpu(), y_pred.cpu(), average='weighted')
    return accuracy, recall, precision, f1


X_train1, Y_train1 = Embedding.get_embedding(TRAINFILE)
X_test1, Y_test1 = Embedding.get_embedding(TESTFILE)
X_train1 = torch.FloatTensor(X_train1)
Y_train1 = torch.LongTensor(Y_train1)
X_test1 = torch.FloatTensor(X_test1)
Y_test1 = torch.LongTensor(Y_test1)
train_dataset = TensorDataset(X_train1, Y_train1)
test_dataset = TensorDataset(X_test1, Y_test1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

model = EncoderCNN(max_length, word_embedding_dim, pos_embedding_dim, HIDDEN_DIM)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

'''
for e in range(epoch):

      loss_epoch = 0
      model.train()

      for x, y in train_loader:
          optimizer.zero_grad()
          y_pred = model((x.permute(0,2,1)).cuda())
          y = y.view(-1,1).cuda()
          loss = criterion(y_pred, y.squeeze(1))
          loss.backward()
          optimizer.step()
          loss_epoch += loss.item()
torch.save(model.state_dict(), "baseline.pth")
'''
model.load_state_dict(torch.load('baseline.pth'))

model.eval()
train_acc = get_model_acc(model, train_loader)
#accuracy, recall, precision, f1= get_model_acc(model, test_loader)
print ("Testing Accuracy:", get_model_acc(model, test_loader))
