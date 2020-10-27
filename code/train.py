import numpy as np

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
batch_size = 200

train_data = DataSet(path['yelp13-train'])
test_data = DataSet(path['yelp13-test'])
dev_data = DataSet(path['yelp13-dev'])

all_doc = np.concatenate([train_data.t_docs, test_data.t_docs, dev_data.t_docs])
embedding_file, words_dict = load_word_embedding(path['yelp13-w2vec'], all_doc)

u_dict, p_dict = train_data.usr_prd_dict()
huapa = HUAPA(embedding_file, hidden_size, max_doc_len, max_sen_len, learning_rate, batch_size, len(u_dict), len(p_dict), 5)
train_X = transform(words_dict, train_data.t_docs, max_doc_len, max_sen_len)
u, p = train_data.usr_prd(u_dict, p_dict)

huapa.train(train_X, u, p, train_data.t_label)

