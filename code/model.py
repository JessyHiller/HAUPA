import torch


class HUAPA(object):
    def __init__(self, embedding_file, hidden_size, max_doc_len, max_sen_len, lr, batch_size, u_num, p_num):
        self.embedding_file = embedding_file
        self.embedding_dim = embedding_file.shape()[-1]
        self.hidden_size = hidden_size
        self.max_doc_len = max_doc_len
        self.max_sen_len = max_sen_len
        self.lr = lr
        self.no_review = batch_size
        self.optimizer = torch.optim.Adam(lr=lr)
        self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(embedding_file))
        self.usr_embedding = torch.nn.Embedding(u_num, 2*self.hidden_size)
        self.prd_embedding = torch.nn.Embedding(p_num, 2*self.hidden_size)
        self.w2s_LSTM = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.s2d_LSTM = torch.nn.LSTM(input_size=self.hidden_size*2, hidden_size=hidden_size, bidirectional=True)

    def look_up(self, X, uid, pid):
        """
        :param X: (no_review, max_doc_len, max_sen_len)
        :param uid: int
        :param pid: int
        :return: (no_review, max_doc_len, max_sen_len, embedding_dim)
        """
        return self.embedding(X), self.usr_embedding(uid), self.prd_embedding(pid)

    def w2s(self, X):
        """
        :param: X (no_review, max_doc_len, max_sen_len, embedding_dim)
        biLSTM input of shape(seq_len, batch, input_size)
        output, (h_n, c_n)
                 output of shape(seq_len, batch, num_directions*hidden_size)
        :return (batch, seq_len, num_directions*hidden_size)
        """
        input = X.view((self.no_review*self.max_doc_len, self.max_sen_len, self.embedding_dim))
        input.permute_(0, 1)
        output, (_, _) = self.w2s_LSTM(input)
        output.permute_(0, 1)
        return output


