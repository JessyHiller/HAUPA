import numpy as np
import torch.optim as opt
import torch

from code.data import load_word_embedding
from code.data import transform
from code.data import Data
from code.model import HUAPA
from code.test import test


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

max_doc_len = 40
max_sen_len = 50
learning_rate = 0.005
hidden_size = 100
batch_size = 100

train_data = Data(path['yelp13-train'])
test_data = Data(path['yelp13-test'])
dev_data = Data(path['yelp13-dev'])

all_doc = np.concatenate([train_data.t_docs, test_data.t_docs, dev_data.t_docs])
embedding_file, words_dict = load_word_embedding(path['yelp13-w2vec'], all_doc)

u_dict, p_dict = train_data.usr_prd_dict()

huapa = HUAPA(embedding_file, hidden_size, max_doc_len, max_sen_len, batch_size, len(u_dict), len(p_dict), 5)
train_X, sen_len, doc_len, outlier_index = transform(words_dict, train_data.t_docs, max_doc_len, max_sen_len)
u, p = train_data.usr_prd(u_dict, p_dict)

l = np.array(train_data.t_label.copy())

if not len(outlier_index) == 0:
    print(outlier_index)
    outlier_index.reverse()
    for i in outlier_index:
        train_X = np.delete(train_X, i, 0)
        sen_len = np.delete(sen_len, i, 0)
        doc_len = np.delete(doc_len, i, 0)
        u = np.delete(u, i, 0)
        p = np.delete(p, i, 0)
        l = np.delete(l, i, 0)

optimizer = opt.Adam(huapa.parameters(), lr=learning_rate)
data_size = train_X.shape[0]

assert data_size == len(sen_len) == len(doc_len) == len(u) == len(p) == len(l)

iters = data_size//batch_size
if iters * batch_size < data_size:
    iters += 1

num_epoch = 10

acc_best = 0

for i in range(num_epoch):
    print('epoch {}/{}'.format(i+1, num_epoch))
    shuffle_indices = np.random.permutation(np.arange(data_size))
    train_X = train_X[shuffle_indices]
    u = u[shuffle_indices]
    p = p[shuffle_indices]
    sen_len = sen_len[shuffle_indices]
    doc_len = doc_len[shuffle_indices]
    l = l[shuffle_indices]

    for j in range(iters):
        X = dict()
        start = j * batch_size
        end = min(data_size, (j + 1) * batch_size)
        X['doc'] = torch.LongTensor(train_X[start: end])
        X['usr'] = torch.LongTensor(u[start: end])
        X['prd'] = torch.LongTensor(p[start: end])
        s_l = torch.tensor(sen_len[start: end])
        d_l = torch.tensor(doc_len[start: end])
        predict_u, predict_p, predict = huapa.forward(X, s_l, d_l)

        label = l[start: end]
        l_train = torch.tensor(np.eye(5)[label])

        loss1 = -torch.sum(torch.mul(torch.log(predict), l_train))
        loss2 = -torch.sum(torch.mul(torch.log(predict_u), l_train))
        loss3 = -torch.sum(torch.mul(torch.log(predict_p), l_train))
        loss = 0.4*loss1+0.3*loss2+0.3*loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (j+1) % 25 == 0:
            acc, r = test(dev_data, words_dict, max_doc_len, max_sen_len, u_dict, p_dict, batch_size, huapa)
            print("epoch {}-iter {}:  accuracy {}, RSME {}".format(i+1, j+1, acc, r))
            if acc > acc_best:
                acc_best = acc
                print("new parameter")
                torch.save(huapa.state_dict(), '../checkpoint/yelp-2013-'+str(i+1)+'-'+str(j+1)+'-parameter.pkl')
