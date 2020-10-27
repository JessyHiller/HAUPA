import numpy as np


def load_word_embedding(path, corpus):
    word_set = set()
    for docs in corpus:
        words = docs.strip().split()
        for w in words:
            word_set.add(w)

    word_embeddings = []
    word_dict = dict()
    word_dict['$EOF$'] = 0  # add EOF
    word_embeddings.append(np.zeros(200))
    index = 1
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            if len(line) == 2:
                continue
            w = line[0]
            if w not in word_set:
                continue
            word_dict[w] = index
            vector = [float(i) for i in line[-200:]]
            word_embeddings.append(vector)
            index += 1
    return np.array(word_embeddings), word_dict


def transform(word_dict, reviews, max_sen_len, max_doc_len):
    """
    :param word_dict: map word to index
    :param reviews: list of string
    :param max_sen_len:
    :param max_doc_len:
    :return: X(no_reviews, max_doc_len, max_sen_len)
    """
    X = []
    for doc in reviews:
        i = 0
        x = np.zeros((max_doc_len, max_sen_len), type=int)
        for sen in doc.split('<sssss>'):
            if i == max_doc_len:
                break
            j = 0
            for w in sen.strip().split(' '):
                if j == max_sen_len:
                    break
                x[i][j] = word_dict[w]
                j += 1
        i += 1
        X.append(x)
    return np.array(X)


class DataSet(object):
    def __init__(self, data_path):
        self.t_usr = []
        self.t_prd = []
        self.t_label = []
        self.t_docs = []
        with open(data_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                terms = line.strip().split('\t\t')
                self.t_usr.append(terms[0])
                self.t_prd.append(terms[1])
                self.t_label.append(int(terms[2])-1)
                self.t_docs.append(terms[3].lower())
        self.data_size = len(self.t_docs)

    def usr_prd_dict(self):
        usr_dict = dict()
        prd_dict = dict()
        for u in self.t_usr:
            i = 0
            if usr_dict.get(u) is None:
                usr_dict[u] = i
                i += 1
        for p in self.t_prd:
            i = 0
            if prd_dict.get(p) is None:
                prd_dict[p] = i
                i += 1
        return usr_dict, prd_dict

    def usr_prd(self, u_dict, p_dict):
        usr = []
        prd = []
        for i in range(self.data_size):
            usr.append(u_dict[self.t_usr[i]])
            prd.append(p_dict[self.t_prd[i]])
        return usr, prd
