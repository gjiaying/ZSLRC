import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import spacy
import math

spacy_nlp = spacy.load('en_core_web_sm')
glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.word2vec.txt'
max_length = 500
word_embedding_dim = 50
pos_embedding_dim = 5
feature_embedding = []
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

class Embedding:
    def __init__(self, file):
        self.file = file

    def get_embedding(file):
        s = open(file,"r",encoding="utf-8")
        X_train = []
        Y_train = []
        while 1:
            line = s.readline()
            if not line:
                break
            line1 = eval(line)
            doc = spacy_nlp(line1['text'])
            pe = line1['pe']
            label = line1['label']
            list = []
            newarray1 = []
            newarray2 = []
            str = ""
            for word in pe:

                if word == '[' or word == ']' or word == ' ' or word == '-':
                    continue
                elif word == ',':
                    list.append(str)
                    str = ""
                else:
                    str = str + word

            list.append(str)
            matrix_len = int(len(list)/2)

            k = 0
            for j in range(matrix_len):

                newarray1.append(int(list[k]))
                newarray2.append(int(list[k+1]))
                k = k + 2

            weights_matrix = np.zeros((max_length, 50))
            position_matrix1 = np.zeros((max_length, 5))
            position_matrix2 = np.zeros((max_length, 5))

            input1 = torch.LongTensor(newarray1)
            input2 = torch.LongTensor(newarray2)
            embedding = torch.nn.Embedding(max_length, pos_embedding_dim)
            position_matrix1 = embedding(input1)
            position_matrix2 = embedding(input2)
            position_matrix1 = F.pad(input=position_matrix1, pad=(0, 0, 0, (max_length-matrix_len)), mode='constant', value=0)
            position_matrix2 = F.pad(input=position_matrix2, pad=(0, 0, 0, (max_length-matrix_len)), mode='constant', value=0)
            position_matrix1 = position_matrix1.type(torch.DoubleTensor)
            position_matrix2 = position_matrix2.type(torch.DoubleTensor)


            i = -1
            for token in doc:
                i = i + 1

                try:
                    weights_matrix[i] = torch.from_numpy(glove_model[(token.lemma_).lower()])
                except:
                    weights_matrix[i] = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)

            X_train.append((torch.cat((torch.from_numpy(weights_matrix), position_matrix1, position_matrix2), 1)).detach().numpy())
            Y_train.append(float(label))

        return X_train, Y_train
    
    def featureEmbedding(file):
        s = open(file,"r",encoding="utf-8")

        while 1:
            line = s.readline()
            if not line:
                break
            line1 = eval(line)
            label = line1['label']
            hyper1 = line1['hyper1']
            hyper2 = line1['hyper2']
            feature1 = line1['feature1']
            feature2 = line1['feature2']
            feature3 = line1['feature3']
            feature4 = line1['feature4']
            try:
                hyper1_vector = torch.from_numpy(glove_model[(hyper1).lower()])
            except:
                hyper1_vector = torch.from_numpy(np.random.rand(50)).type(torch.FloatTensor)
            try:
                hyper2_vector = torch.from_numpy(glove_model[(hyper2).lower()])
            except:
                hyper2_vector = torch.from_numpy(np.random.rand(50)).type(torch.FloatTensor)
            try:
                feature1_vector = torch.from_numpy(glove_model[(feature1).lower()])
            except:
                feature1_vector = torch.from_numpy(np.random.rand(50)).type(torch.FloatTensor)
            try:
                feature2_vector = torch.from_numpy(glove_model[(feature2).lower()])
            except:
                feature2_vector = torch.from_numpy(np.random.rand(50)).type(torch.FloatTensor)
            try:
                feature3_vector = torch.from_numpy(glove_model[(feature3).lower()])
            except:
                feature3_vector = torch.from_numpy(np.random.rand(50)).type(torch.FloatTensor)
            try:
                feature4_vector = torch.from_numpy(glove_model[(feature4).lower()])

            except:
                feature4_vector = torch.from_numpy(np.random.rand(50)).type(torch.FloatTensor)
            feature_embedding.append((torch.cat((hyper1_vector, hyper2_vector, feature1_vector, feature2_vector, feature3_vector, feature4_vector), 0)).detach().numpy())
        #    feature_embedding.append((torch.cat((hyper1_vector, hyper2_vector, feature1_vector, feature2_vector, feature3_vector), 0)).detach().numpy())
        return (feature_embedding)
    
    
    def get_test_embedding(model, loader, featest_embedding, batch_size):
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
