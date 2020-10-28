import numpy as np
import torch.optim as opt
import torch

from code.data import load_word_embedding
from code.data import transform
from code.data import DataSet
from code.model import HUAPA


path = {'imdb-dev': '../dataset/imdb.dev.txt.ss',
        'imdb-test': '../dataset/imdb.test.txt.ss',
        'imdb-train': '../dataset/imdb.train.txt.ss',
        'imdb-w2vec': '../WordEmbedding/imdb-embedding-200d.txt',
        'yelp13-dev': '../dataset/yelp-2013-seg-20-20.dev.ss',
        'yelp13-test': '../dataset/yelp-2013-seg-20-20.test.ss',
        'yelp13-train': '../dataset/yelp-2013-seg-20-20.train.ss',
        'yelp13-w2vec': '../WordEmbedding/yelp-2013-embedding-200d.txt',
        'yelp14-dev': '../dataset/yelp-2014-seg-20-20.dev.ss',
        'yelp14-test': '../dataset/yelp-2014-seg-20-20.test.ss',
        'yelp14-train': '../dataset/yelp-2014-seg-20-20.train.ss',
        'yelp14-w2vec': '../WordEmbedding/yelp-2014-embedding-200d.txt'}

max_doc_len = 50
max_sen_len = 40
learning_rate = 0.005
hidden_size = 100
batch_size = 10

train_data = DataSet(path['yelp13-train'])
test_data = DataSet(path['yelp13-test'])
dev_data = DataSet(path['yelp13-dev'])

all_doc = np.concatenate([train_data.t_docs, test_data.t_docs, dev_data.t_docs])
embedding_file, words_dict = load_word_embedding(path['yelp13-w2vec'], all_doc)

u_dict, p_dict = train_data.usr_prd_dict()
huapa = HUAPA(embedding_file, hidden_size, max_doc_len, max_sen_len, batch_size, len(u_dict), len(p_dict), 5)
train_X = transform(words_dict, train_data.t_docs, max_doc_len, max_sen_len)
u, p = train_data.usr_prd(u_dict, p_dict)

optimizer = opt.Adam(huapa.parameters(), lr=learning_rate)
data_size = train_X.shape[0]
epoch = data_size//batch_size

for i in range(epoch):
    X = dict()
    X['doc'] = torch.LongTensor(train_X[i*batch_size:(i+1)*batch_size])
    X['usr'] = torch.LongTensor(u[i*batch_size:(i+1)*batch_size])
    X['prd'] = torch.LongTensor(p[i*batch_size:(i+1)*batch_size])
    predict_u, predict_p, predict = huapa.forward(X)

    l_train = train_data.t_label[i*batch_size:(i+1)*batch_size]
    loss1 = torch.sum(torch.mul(predict, l_train), -1)
    loss2 = torch.sum(torch.mul(predict_u, l_train), -1)
    loss3 = torch.sum(torch.mul(predict_p, l_train), -1)
    loss = 0.4*loss1+0.3*loss2+0.3*loss3
    loss.backward()
    optimizer.step()
    print('first epoch')
    break
