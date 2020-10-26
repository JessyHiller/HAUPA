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

        self.w2s_u_lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.s2d_u_lstm = torch.nn.LSTM(input_size=self.hidden_size*2, hidden_size=hidden_size, bidirectional=True)

        self.w2s_p_lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.s2d_p_lstm = torch.nn.LSTM(input_size=self.hidden_size * 2, hidden_size=hidden_size, bidirectional=True)

        self.v_wu = torch.empty((2*self.hidden_size, 1)).uniform_(-0.01, 0.01)
        self.W_wuh = torch.empty((2*self.hidden_size, 2*self.hidden_size)).uniform_(-0.01, 0.01)
        self.W_wuu = torch.empty((2*self.hidden_size, 2*self.hidden_size)).uniform_(-0.01, 0.01)
        self.b_wu = torch.empty((2*self.hidden_size, 1)).uniform_(-0.01, 0.01)

        self.v_su = torch.empty((2 * self.hidden_size, 1)).uniform_(-0.01, 0.01)
        self.W_suh = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.W_suu = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.b_su = torch.empty((2 * self.hidden_size, 1)).uniform_(-0.01, 0.01)

        self.v_wp = torch.empty((2 * self.hidden_size, 1)).uniform_(-0.01, 0.01)
        self.W_wph = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.W_wpu = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.b_wp = torch.empty((2 * self.hidden_size, 1)).uniform_(-0.01, 0.01)

        self.v_sp = torch.empty((2 * self.hidden_size, 1)).uniform_(-0.01, 0.01)
        self.W_sph = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.W_spu = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.b_sp = torch.empty((2 * self.hidden_size, 1)).uniform_(-0.01, 0.01)

    def look_up(self, X, uid, pid):
        """
        :param X: (no_review, max_doc_len, max_sen_len)
        :param uid: int
        :param pid: int
        :return: (no_review, max_doc_len, max_sen_len, embedding_dim)
        """
        return self.embedding(X), self.usr_embedding(uid), self.prd_embedding(pid)

    def w2s_u(self, X):
        """
        :param: X (no_review, max_doc_len, max_sen_len, embedding_dim)
        biLSTM input of shape(seq_len, batch, input_size)
        output, (h_n, c_n)
                 output of shape(seq_len, batch, num_directions*hidden_size)
        :return (batch_size = no_review*max_doc_len, max_sen_len, num_directions*hidden_size)
        """
        input_seq = X.view((self.no_review*self.max_doc_len, self.max_sen_len, self.embedding_dim))
        input_seq.permute_(0, 1)
        s_hidden_state, (_, _) = self.w2s_LSTM(input_seq)
        s_hidden_state.permute_(0, 1)
        return s_hidden_state

    def s2d_u(self, X):
        """

        :param X: ()
        :return:
        """

    def ua_w(self, s_hidden_state, u):
        """

        :param s_hidden_state: (batch_size=no_review*max_doc_len, max_sen_len, 2*hidden_size)
        :param u: (batch_size, max_sen_len, 2*hidden_size)
        :return: s_u: (batch_size, 2*hidden_size)
        """
        s_hidden_state.unsqueeze_(-1)
        linear = torch.matmul(self.W_wuh, s_hidden_state)
        u.unsqueeze_(-1)
        linear += torch.matmul(self.W_wuu, u)
        linear = torch.add(linear, self.b_wu)
        t = torch.tanh(linear)
        e = torch.matmul(self.v_wu.transpose(0, 1), t)
        e.squeeze_()
        alpha = torch.nn.functional.softmax(e, dim=-1)
        alpha.unsqueeze_(-1)
        s_hidden_state.squeeze_()
        u.squeeze_()
        s_hidden_state.permute_(1, 2)
        s_u = torch.matmul(s_hidden_state, alpha)
        return s_u.squeeze()

    def ua_s(self, d_hidden_state, u):
        """

        :param d_hidden_state: (batch_size=no_reviews, max_doc_len, 2*hidden_size)
        :param u: (batch_size, max_doc_len, 2*hidden_size)
        :return: d_u
        """
        d_hidden_state.unsqueeze_(-1)
        linear = torch.matmul(self.W_suh, d_hidden_state)
        u.unsqueeze_(-1)
        linear += torch.matmul(self.W_suu, u)
        linear = torch.add(linear, self.b_su)
        t = torch.tanh(linear)
        e = torch.matmul(self.v_wu.transpose(0, 1), t)
        e.squeeze_()
        beta = torch.nn.functional.softmax(e, dim=-1)
        beta.unsqueeze_(-1)
        d_hidden_state.squeeze_()
        u.squeeze_()
        d_hidden_state.permute_(1, 2)
        d_u = torch.matmul(d_hidden_state, beta)
        return d_u.squeeze()

    def