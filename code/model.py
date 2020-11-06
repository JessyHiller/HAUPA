import torch
import numpy as np


class HUAPA(torch.nn.Module):
    def __init__(self, embedding_file, hidden_size, max_doc_len, max_sen_len, batch_size, u_num, p_num, num_class):

        super(HUAPA, self).__init__()
        self.embedding_file = embedding_file
        self.embedding_dim = embedding_file.shape[-1]
        self.hidden_size = hidden_size
        self.max_doc_len = max_doc_len
        self.max_sen_len = max_sen_len
        self.num_class = num_class
        self.no_review = batch_size

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
        self.b_wu = torch.empty((2*self.hidden_size)).uniform_(-0.01, 0.01)

        self.v_su = torch.empty((2 * self.hidden_size, 1)).uniform_(-0.01, 0.01)
        self.W_suh = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.W_suu = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.b_su = torch.empty((2 * self.hidden_size)).uniform_(-0.01, 0.01)

        self.v_wp = torch.empty((2 * self.hidden_size, 1)).uniform_(-0.01, 0.01)
        self.W_wph = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.W_wpu = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.b_wp = torch.empty((2 * self.hidden_size)).uniform_(-0.01, 0.01)

        self.v_sp = torch.empty((2 * self.hidden_size, 1)).uniform_(-0.01, 0.01)
        self.W_sph = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.W_spu = torch.empty((2 * self.hidden_size, 2 * self.hidden_size)).uniform_(-0.01, 0.01)
        self.b_sp = torch.empty((2 * self.hidden_size)).uniform_(-0.01, 0.01)

        self.predict = torch.nn.Linear(4*self.hidden_size, self.num_class)
        self.predict_u = torch.nn.Linear(2*self.hidden_size, self.num_class)
        self.predict_p = torch.nn.Linear(2*self.hidden_size, self.num_class)

    def forward(self, X):
        """

        :param X: (total_no_reviews, max_doc_len, max_sen_len) int
        :param usr: list of index
        :param prd: list of index
        :param labels: list of lable
        :return:
        """
        # assert len(usr) == len(prd) and len(usr) == len(prd) and len(usr) == np.shape(X)[0], "wrong data!"

        x_tr, u_train, p_train = self.look_up(X['doc'], X['usr'], X['prd'])

        x_train = x_tr.float()

        sh_u = self.w2s_u(x_train)
        sp_u = self.ua_w(sh_u, u_train)
        dh_u = self.s2d_u(sp_u)
        dp_u = self.ua_s(dh_u, u_train)

        sh_p = self.w2s_p(x_train)
        sp_p = self.pa_w(sh_p, p_train)
        dh_p = self.s2d_p(sp_p)
        dp_p = self.pa_s(dh_p, p_train)

        pre_u = torch.nn.functional.softmax(self.predict_u(dp_u), -1)
        pre_p = torch.nn.functional.softmax(self.predict_u(dp_p), -1)
        dp = torch.cat((dp_u, dp_p), -1)
        pre = torch.nn.functional.softmax(self.predict(dp), -1)

        return pre_u, pre_p, pre

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
        :return (no_review, max_doc_len, max_sen_len, num_directions*hidden_size)
        """
        input_seq = X.view((self.no_review*self.max_doc_len, self.max_sen_len, self.embedding_dim))
        inpt = input_seq.transpose(0, 1)
        s_hidden_state, (_, _) = self.w2s_u_lstm(inpt)
        s_h_state = s_hidden_state.transpose(0, 1)
        output_seq = s_h_state.view((self.no_review, self.max_doc_len, self.max_sen_len, 2*self.hidden_size))
        return output_seq

    def s2d_u(self, X):
        """

        :param X: (batch_size=no_review, max_doc_len, 2*hidden_size)
        :return:(batch_size, max_doc_len, 2*hidden_size)
        """
        d_hidden_state, (_, _) = self.s2d_u_lstm(X.transpose(0, 1))
        re = d_hidden_state.transpose(0, 1)
        return re

    def w2s_p(self, X):
        input_seq = X.view((self.no_review * self.max_doc_len, self.max_sen_len, self.embedding_dim))
        inpt = input_seq.transpose(0, 1)
        s_hidden_state, (_, _) = self.w2s_p_lstm(inpt)
        s_h_state = s_hidden_state.transpose(0, 1)
        output_seq = s_h_state.view((self.no_review, self.max_doc_len, self.max_sen_len, 2*self.hidden_size))
        return output_seq

    def s2d_p(self, X):
        d_hidden_state, (_, _) = self.s2d_p_lstm(X.transpose(0, 1))
        re = d_hidden_state.transpose(0, 1)
        return re

    def ua_w(self, s_hidden_state, u):
        """

        :param s_hidden_state: (no_review, max_doc_len, max_sen_len, 2*hidden_size)
        :param u: (no_review, 2*hidden_size)
        :return: s_u: (no_review, max_doc_len, 2*hidden_size)
        """
        h_projection = torch.matmul(s_hidden_state, self.W_wuh)
        u_projection = torch.matmul(u, self.W_wuh)
        add_projection = h_projection + u_projection[:, None, None, :]
        linear = torch.add(add_projection, self.b_wu)
        t = torch.tanh(linear)
        e = torch.matmul(t, self.v_wu)
        alpha = torch.nn.functional.softmax(e, dim=-2)
        s_u = torch.matmul(s_hidden_state.transpose(-1, -2), alpha).squeeze()
        return s_u

    def ua_s(self, d_hidden_state, u):
        """

        :param d_hidden_state: (batch_size=no_reviews, max_doc_len, 2*hidden_size)
        :param u: (batch_size, 2*hidden_size)
        :return: d_u: (batch_size, 2*hidden_size)
        """
        h_projection = torch.matmul(d_hidden_state, self.W_suh)
        u_projection = torch.matmul(u, self.W_suh)
        add_projection = h_projection + u_projection[:, None, :]
        linear = torch.add(add_projection, self.b_su)
        t = torch.tanh(linear)
        e = torch.matmul(t, self.v_wu)
        beta = torch.nn.functional.softmax(e, dim=-2)
        d_u = torch.matmul(d_hidden_state.transpose(-1, -2), beta).squeeze()
        return d_u

    def pa_w(self, s_hidden_state, p):
        h_projection = torch.matmul(s_hidden_state, self.W_wph)
        p_projection = torch.matmul(p, self.W_wpu)
        add_projection = h_projection + p_projection[:, None, None, :]
        linear = torch.add(add_projection, self.b_wp)
        t = torch.tanh(linear)
        e = torch.matmul(t, self.v_wp)
        alpha = torch.nn.functional.softmax(e, dim=-2)
        s_p = torch.matmul(s_hidden_state.transpose(-1, -2), alpha).squeeze()
        return s_p

    def pa_s(self, d_hidden_state, p):
        h_projection = torch.matmul(d_hidden_state, self.W_sph)
        p_projection = torch.matmul(p, self.W_spu)
        add_projection = h_projection + p_projection[:, None, :]
        linear = torch.add(add_projection, self.b_sp)
        t = torch.tanh(linear)
        e = torch.matmul(t, self.v_wp)
        beta = torch.nn.functional.softmax(e, dim=-2)
        d_p = torch.matmul(d_hidden_state.transpose(-1, -2), beta).squeeze()
        return d_p
