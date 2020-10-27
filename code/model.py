import torch
import numpy as np


class HUAPA(object):
    def __init__(self, embedding_file, hidden_size, max_doc_len, max_sen_len, lr, batch_size, u_num, p_num, num_class):
        self.embedding_file = embedding_file
        self.embedding_dim = embedding_file.shape[-1]
        self.hidden_size = hidden_size
        self.max_doc_len = max_doc_len
        self.max_sen_len = max_sen_len
        self.lr = lr
        self.num_class = num_class
        self.no_review = batch_size
        self.optimizer = torch.optim.Adam(self.parameter, lr=lr)

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

        self.predict = torch.nn.Linear(4*self.hidden_size, self.num_class)
        self.predict_u = torch.nn.Linear(2*self.hidden_size, self.num_class)
        self.predict_p = torch.nn.Linear(2*self.hidden_size, self.num_class)

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
        input_seq.permute_(0, 1)
        s_hidden_state, (_, _) = self.w2s_u_lstm(input_seq)
        s_hidden_state.permute_(0, 1)
        output_seq = s_hidden_state.view((self.no_review, self.max_doc_len, self.max_sen_len, 2*self.hidden_size))
        return output_seq

    def s2d_u(self, X):
        """

        :param X: (batch_size=no_review, max_doc_len, 2*hidden_size)
        :return:(batch_size, max_doc_len, 2*hidden_size)
        """
        d_hidden_state, (_, _) = self.s2d_u_lstm(X.permute(0, 1))
        d_hidden_state.permute_(0, 1)
        return d_hidden_state

    def w2s_p(self, X):
        input_seq = X.view((self.no_review * self.max_doc_len, self.max_sen_len, self.embedding_dim))
        input_seq.permute_(0, 1)
        s_hidden_state, (_, _) = self.w2s_p_lstm(input_seq)
        s_hidden_state.permute_(0, 1)
        output_seq = s_hidden_state.view((self.no_review, self.max_doc_len, self.max_sen_len, 2*self.hidden_size))
        return output_seq

    def s2d_p(self, X):
        d_hidden_state, (_, _) = self.s2d_p_lstm(X.permute(0, 1))
        d_hidden_state.permute_(0, 1)
        return d_hidden_state

    def ua_w(self, s_hidden_state, u):
        """

        :param s_hidden_state: (no_review, max_doc_len, max_sen_len, 2*hidden_size)
        :param u: (no_review, 2*hidden_size)
        :return: s_u: (no_review, max_doc_len, 2*hidden_size)
        """
        s_hidden_state.unsqueeze_(-1)
        linear = torch.matmul(self.W_wuh, s_hidden_state)
        u.unsqueeze_(-1)
        linear = torch.add(linear, torch.matmul(self.W_wuu, u))
        linear = torch.add(linear, self.b_wu)
        t = torch.tanh(linear)
        e = torch.matmul(self.v_wu.transpose(0, 1), t)
        e.squeeze_()
        alpha = torch.nn.functional.softmax(e, dim=-1)
        alpha.unsqueeze_(-1)
        s_hidden_state.squeeze_()
        u.squeeze_()
        s_hidden_state.permute_(-1, -2)
        s_u = torch.matmul(s_hidden_state, alpha)
        return s_u.squeeze()

    def ua_s(self, d_hidden_state, u):
        """

        :param d_hidden_state: (batch_size=no_reviews, max_doc_len, 2*hidden_size)
        :param u: (batch_size, max_doc_len, 2*hidden_size)
        :return: d_u: (batch_size, 2*hidden_size)
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

    def pa_w(self, s_hidden_state, p):
        s_hidden_state.unsqueeze_(-1)
        linear = torch.matmul(self.W_wph, s_hidden_state)
        p.unsqueeze_(-1)
        linear += torch.matmul(self.W_wpu, p)
        linear = torch.add(linear, self.b_wp)
        t = torch.tanh(linear)
        e = torch.matmul(self.v_wp.transpose(0, 1), t)
        e.squeeze_()
        alpha = torch.nn.functional.softmax(e, dim=-1)
        alpha.unsqueeze_(-1)
        s_hidden_state.squeeze_()
        p.squeeze_()
        s_hidden_state.permute_(1, 2)
        s_p = torch.matmul(s_hidden_state, alpha)
        return s_p.squeeze()

    def pa_s(self, d_hidden_state, p):
        d_hidden_state.unsqueeze_(-1)
        linear = torch.matmul(self.W_sph, d_hidden_state)
        p.unsqueeze_(-1)
        linear += torch.matmul(self.W_spu, p)
        linear = torch.add(linear, self.b_sp)
        t = torch.tanh(linear)
        e = torch.matmul(self.v_wp.transpose(0, 1), t)
        e.squeeze_()
        beta = torch.nn.functional.softmax(e, dim=-1)
        beta.unsqueeze_(-1)
        d_hidden_state.squeeze_()
        p.squeeze_()
        d_hidden_state.permute_(1, 2)
        d_p = torch.matmul(d_hidden_state, beta)
        return d_p.squeeze()

    def predict(self, X):
        """

        :param X: (batch_size=no_review, 4*self.hidden_size)
        :return: y_: (batch_size, )
        """
        y_ = self.predict(X)
        return torch.nn.functional.softmax(y_, -1)

    def predict_u(self, X):
        y_ = self.predict_u(X)
        return torch.nn.functional.softmax(y_, -1)

    def predict_p(self, X):
        y_ = self.predict_p(X)
        return torch.nn.functional.softmax(y_, -1)

    def train(self, X, usr, prd, labels):
        """

        :param X: (total_no_reviews, max_doc_len, max_sen_len) int
        :param usr: list of index
        :param prd: list of index
        :param labels: list of lable
        :return:
        """
        assert len(usr) == len(prd) and len(usr) == len(prd) and len(usr) == np.shape(X)[0], "wrong data!"
        data_size = len(usr)
        epoch = data_size//self.no_review

        for i in range(epoch):
            x = X[i*self.no_review:(i+1)*self.no_review]
            u = usr[i*self.no_review:(i+1)*self.no_review]
            p = prd[i*self.no_review:(i+1)*self.no_review]
            label = labels[i*self.no_review:(i+1)*self.no_review]

            x_train, u_train, p_train = self.look_up(x, u, p)

            sh_u = self.w2s_u(x_train)
            sp_u = self.ua_w(sh_u, u_train)
            dh_u = self.s2d_u(sp_u)
            dp_u = self.ua_s(dh_u, u_train)

            sh_p = self.w2s_p(x_train)
            sp_p = self.pa_w(sh_p, p_train)
            dh_p = self.s2d_p(sp_p)
            dp_p = self.pa_s(dh_p, p_train)

            l_train = np.eye(self.num_class)[label]
            predict_u = self.predict_u(dp_u)
            predict_p = self.predict_p(dp_p)
            predict = self.predict(torch.cat(dp_u, dp_p), -1)
            loss1 = torch.sum(torch.mul(predict, l_train), -1)
            loss2 = torch.sum(torch.mul(predict_u, l_train), -1)
            loss3 = torch.sum(torch.mul(predict_p, l_train), -1)
            loss = 0.4*loss1+0.3*loss2+0.3*loss3
            loss.backward()
            self.optimizer.step()
            print("here")
            break
