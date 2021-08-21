import numpy as np
import pandas as pd
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
train_file = "basetrain.txt"
si_file = "featureset.txt"
test_file="test6.txt"

word_embedding_dim = 50
pos_embedding_dim = 5
HIDDEN_DIM = 300
epoch = 500
max_length = 500
batch_size = 4
num_workers = 0
shuffle = True
n_classes = 20
n_layers = 2
bidirectional = True
THRESHOLD = 2e-8
learning_rate = 1e-2
embedding_dim = word_embedding_dim + pos_embedding_dim*2

#glove2word2vec(glove_input_file, word2vec_output_file)
X_train1 = []
Y_train1 = []
X_test1 = []
Y_test1 = []
train_accuracy = []
test_accuracy = []
output = []
labels = []
list_label_nums = []
list_ave_weights = []
y_predict = []
truth_labels = []
feature_embedding = []
featest_embedding = []
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)



def weights_calculation(output, labels, feature_embedding):
    list_label_num = []
    list_ave_weight = []
    label_unique = set(labels)
   
    for label_data in label_unique:
        ave = np.zeros(shape=(1, HIDDEN_DIM))
        total = 0
    
        for i in range(len(labels)):
            if labels[i] == label_data:
                total = total + 1
                ave = np.sum(([ave, output[i]]), axis = 0)
                
        list_label_num.append(label_data)
        list_ave_weight.append(np.append(ave/total, feature_embedding[label_data - 1]))
        
    return list_label_num, list_ave_weight


def softmax(x):
    exp_x = x
    sm = exp_x/np.sum(exp_x, axis=-1, keepdims=True)
 
    return sm

def distance_calculation(truth_labels, test_embedding, list_label_nums, list_ave_weights, feature_embedding):
   
    label_list = []
    threshold = THRESHOLD
    for i in range(len(Y_test1)):
        dist_list = []
        dist_new_list = []

        for j in range(len(list_ave_weights)):
            dist = np.linalg.norm(test_embedding[i] - list_ave_weights[j])
            dist1 = math.exp(-dist)
            dist_list.append(dist1)

        if (min(dist_list) < threshold):#zeroshot
            test_embedding[i] = test_embedding[i][HIDDEN_DIM:]
            for k in range(len(feature_embedding)):

                dist_new = np.linalg.norm(test_embedding[i] - feature_embedding[k])
                dist1_new = math.exp(-dist_new)
                dist_new_list.append(dist1_new)
            index = np.argmax(softmax(dist_new_list))
            label_list.append(index + 1)

        else:
            index = np.argmax(softmax(dist_list))
            label_list.append(list_label_nums[index])

    return get_acc(label_list, truth_labels)


def get_test_embedding(model, loader, featest_embedding):
    out_embedding = []
    ys = []
    for x, y in loader:
        ys.append(y.cuda())
       
        for i in range(batch_size):
            try:
                out_embedding.append(model((x.permute(0,2,1)).cuda()).cpu().detach().numpy()[i])
            except:
                continue
    y = torch.cat(ys, dim=0)
    for i in range(len(out_embedding)):
        out_embedding[i] = np.append(out_embedding[i], featest_embedding[y[i] - 1])
    
    return y, out_embedding


def get_acc(pred, y):
    y1 = y.cpu().tolist()
    accuracy = accuracy_score(y1, pred)
    recall = recall_score(y1, pred, average='weighted')
    precision = precision_score(y1, pred, average='weighted')
    f1 = f1_score(y1, pred, average='weighted')
    return accuracy, recall, precision, f1


X_train1, Y_train1 = Embedding.get_embedding(train_file)
X_test1, Y_test1 = Embedding.get_embedding(test_file)
X_train1 = torch.FloatTensor(X_train1)
Y_train1 = torch.LongTensor(Y_train1)
X_test1 = torch.FloatTensor(X_test1)
Y_test1 = torch.LongTensor(Y_test1)
train_dataset = TensorDataset(X_train1, Y_train1)
test_dataset = TensorDataset(X_test1, Y_test1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

model = EncoderCNN(max_length, word_embedding_dim, pos_embedding_dim, HIDDEN_DIM)
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


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

          for i in range(batch_size):
              try:

                  output.append(y_pred[i].cpu().detach().numpy())
                  labels.append(y.cpu().squeeze(1)[i].item())
              except:
                  continue
torch.save(model.state_dict(), "zeroshot.pth")
#model.load_state_dict(torch.load('zeroshot.pth'))
model.eval()

feature_embedding = Embedding.featureEmbedding(si_file)
featest_embedding = Embedding.featureEmbedding(test_file)
list_label_nums, list_ave_weights = weights_calculation(output, labels, feature_embedding)
truth_labels, test_embedding = get_test_embedding(model, test_loader, featest_embedding)
print ("Testing Accuracy for zero-shot Prototype:", distance_calculation(truth_labels, test_embedding, list_label_nums, list_ave_weights, feature_embedding))

