import torch
import numpy as np

from code.data import Data
from code.data import transform

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

train_data = Data(path['yelp13-train'])
test_data = Data(path['yelp13-test'])
dev_data = Data(path['yelp13-dev'])


def test(data, words_dict, max_doc_len, max_sen_len, u_dict, p_dict, batch_size, model):
    doc, sen_len, doc_len, outlier_index = transform(words_dict, data.t_docs, max_doc_len, max_sen_len)
    u, p = data.usr_prd(u_dict, p_dict)
    label = data.t_label.copy()
    if not len(outlier_index) == 0:
        print(outlier_index)
        outlier_index.reverse()
        for i in outlier_index:
            doc = np.delete(doc, i, 0)
            sen_len = np.delete(sen_len, i, 0)
            doc_len = np.delete(doc_len, i, 0)
            u.pop(i)
            p.pop(i)
            label.pop(i)

    N = doc.shape[0]
    iters = N//batch_size
    if iters * batch_size < N:
        iters += 1
    c_u = 0
    s_u = 0
    c_p = 0
    s_p = 0
    c = 0
    s = 0
    for i in range(iters):
        X = dict()
        start = i*batch_size
        end = min(N, (i+1)*batch_size)
        X['doc'] = torch.LongTensor(doc[start:end])
        X['usr'] = torch.LongTensor(u[start:end])
        X['prd'] = torch.LongTensor(p[start:end])
        s_l = torch.tensor(sen_len[start:end])
        d_l = torch.tensor(doc_len[start:end])
        predict_u, predict_p, predict = model.forward(X, s_l, d_l)

        l = label[start:end]
        p_u = torch.argmax(predict_u, -1)
        p_p = torch.argmax(predict_p, -1)
        p_ = torch.argmax(predict, -1)
        for j in range(end-start):
            if l[j] == p_u[j]:
                c_u += 1
            if l[j] == p_p[j]:
                c_p += 1
            if l[j] == p_[j]:
                c += 1
            s_u += (l[j] - p_u[j]) ** 2
            s_p += (l[j] - p_p[j]) ** 2
            s += (l[j] - p_[j]) ** 2

    print('test {} {} {}'.format(c_u, c_p, c))
    return torch.true_divide(c, N), np.sqrt(torch.true_divide(s, N))
