import torch
import torch.nn as nn


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

        self.w2s_u_lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size,
                                        bidirectional=True, batch_first=True)
        self.s2d_u_lstm = torch.nn.LSTM(input_size=self.hidden_size*2, hidden_size=hidden_size,
                                        bidirectional=True, batch_first=True)

        self.w2s_p_lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size,
                                        bidirectional=True, batch_first=True)
        self.s2d_p_lstm = torch.nn.LSTM(input_size=self.hidden_size*2, hidden_size=hidden_size,
                                        bidirectional=True, batch_first=True)

        self.linear_wuu = torch.nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.linear_wuh = torch.nn.Linear(2*self.hidden_size, 2*self.hidden_size, bias=False)
        self.v_wu = torch.nn.Linear(2*self.hidden_size, 1, bias=False)

        self.linear_suu = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.linear_suh = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=False)
        self.v_su = torch.nn.Linear(2 * self.hidden_size, 1, bias=False)

        self.linear_wpp = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.linear_wph = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=False)
        self.v_wp = torch.nn.Linear(2 * self.hidden_size, 1, bias=False)

        self.linear_spp = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.linear_sph = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=False)
        self.v_sp = torch.nn.Linear(2 * self.hidden_size, 1, bias=False)

        self.predict = torch.nn.Linear(4*self.hidden_size, self.num_class)
        self.predict_u = torch.nn.Linear(2*self.hidden_size, self.num_class)
        self.predict_p = torch.nn.Linear(2*self.hidden_size, self.num_class)

    def forward(self, X, sen_len, doc_len):
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

        sorted_doc_len, doc_indices = torch.sort(doc_len, descending=True)
        _, desorted__doc_indices = torch.sort(doc_indices, descending=False)
        x = x_train[doc_indices]
        u = u_train[doc_indices]
        p = p_train[doc_indices]

        sen_l = sen_len[doc_indices].transpose(0, 1).reshape(self.no_review * self.max_doc_len)
        s_l = sen_l[torch.nonzero(sen_l, as_tuple=True)]

        x_w2s_packed = torch.nn.utils.rnn.pack_padded_sequence(x, sorted_doc_len, batch_first=True)

        x_w2s = x_w2s_packed   # (sum(doc_len), max_sen_len, 200)

        sh_u_packed = self.w2s_u(x_w2s, s_l)
        sh_u_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(sh_u_packed, batch_first=True)
        mask0 = torch.zeros(sh_u_padded.shape[:3])

        for i in range(self.no_review):
            l = sorted_doc_len[i]
            for j in range(sh_u_padded.shape[1]):
                if j < l:
                    mask0[i, j, s_l[sum(sorted_doc_len[:i])+j]:] = 1
                else:
                    mask0[i, j, :] = 1

        sp_u = self.ua_w(sh_u_padded, u, mask0)
        dh_u = self.s2d_u(sp_u, sorted_doc_len)

        mask1 = torch.zeros((self.no_review, dh_u.shape[1]))
        for i in range(self.no_review):
            mask1[i, sorted_doc_len[i]:] = 1

        dp_u = self.ua_s(dh_u, u, mask1)

        sh_p_packed = self.w2s_p(x_w2s, s_l)
        sh_p_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(sh_p_packed, batch_first=True)
        sp_p = self.pa_w(sh_p_padded, p, mask0)
        dh_p = self.s2d_p(sp_p, sorted_doc_len)
        dp_p = self.pa_s(dh_p, p, mask1)

        pre_u = torch.nn.functional.softmax(self.predict_u(dp_u), -1)[desorted__doc_indices]
        pre_p = torch.nn.functional.softmax(self.predict_u(dp_p), -1)[desorted__doc_indices]
        dp = torch.cat((dp_u, dp_p), -1)
        pre = torch.nn.functional.softmax(self.predict(dp), -1)[desorted__doc_indices]

        return pre_u, pre_p, pre

    def look_up(self, X, uid, pid):
        """
        :param X: (no_review, max_doc_len, max_sen_len)
        :param uid: int
        :param pid: int
        :return: (no_review, max_doc_len, max_sen_len, embedding_dim)
        """
        return self.embedding(X), self.usr_embedding(uid), self.prd_embedding(pid)

    def w2s_u(self, X, sen_len):
        """
        :param: X (batch, max_sen_len, embedding_dim)
        :param: sen_len (batch): length of each sentence in the reviews, padded with 0
        biLSTM input of shape(batch, seq_len, input_size)
        output, (h_n, c_n)
                 output of shape(batch, seq_len, num_directions*hidden_size)
        :return (no_review, max_doc_len, max_sen_len, num_directions*hidden_size)
        """

        sorted_seq_lengths, indices = torch.sort(sen_len, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        inputs = X.data[indices]
        packed_input = nn.utils.rnn.pack_padded_sequence(input=inputs, lengths=sorted_seq_lengths, batch_first=True)
        s_hidden_state, (_, _) = self.w2s_u_lstm(packed_input)
        s_h_state, _ = nn.utils.rnn.pad_packed_sequence(s_hidden_state, batch_first=True)
        output_seq = s_h_state[desorted_indices]
        return torch.nn.utils.rnn.PackedSequence(output_seq, X.batch_sizes)

    def s2d_u(self, X, doc_len):
        """

        :param X: (batch_size=no_review, -1, 2*hidden_size)
        :param doc_len (no_review) : length of reviews
        :return:(batch_size, -1, 2*hidden_size)
        """
        packed_input = nn.utils.rnn.pack_padded_sequence(input=X, lengths=doc_len, batch_first=True)
        d_hidden_state, (_, _) = self.s2d_u_lstm(packed_input)
        d_h_state, _ = nn.utils.rnn.pad_packed_sequence(d_hidden_state, batch_first=True)
        return d_h_state

    def w2s_p(self, X, sen_len):
        sorted_seq_lengths, indices = torch.sort(sen_len, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        inputs = X.data[indices]
        packed_input = nn.utils.rnn.pack_padded_sequence(input=inputs, lengths=sorted_seq_lengths, batch_first=True)
        s_hidden_state, (_, _) = self.w2s_p_lstm(packed_input)
        s_h_state, _ = nn.utils.rnn.pad_packed_sequence(s_hidden_state, batch_first=True)
        output_seq = s_h_state[desorted_indices]
        return torch.nn.utils.rnn.PackedSequence(output_seq, X.batch_sizes)

    def s2d_p(self, X, doc_len):
        packed_input = nn.utils.rnn.pack_padded_sequence(input=X, lengths=doc_len, batch_first=True)
        d_hidden_state, (_, _) = self.s2d_p_lstm(packed_input)
        d_h_state, _ = nn.utils.rnn.pad_packed_sequence(d_hidden_state, batch_first=True)
        return d_h_state

    def ua_w(self, s_hidden_state, u, m):
        """

        :param s_hidden_state: (, max_sen_len, 2*hidden_size)
        :param u: (no_review, 2*hidden_size)
        :return: s_u: (no_review, -1, 2*hidden_size)
        """

        h_projection = self.linear_wuh(s_hidden_state)
        u_projection = self.linear_wuu(u)
        add_projection = h_projection + u_projection[:, None, None, :]
        t = torch.tanh(add_projection)
        mask = m.unsqueeze(-1)
        e = self.v_wu(t)
        e_ = e.masked_fill(mask == 1, 1e-9)
        alpha = torch.nn.functional.softmax(e_, dim=-2)
        s_u = torch.matmul(s_hidden_state.transpose(-1, -2), alpha).squeeze()
        return s_u

    def ua_s(self, d_hidden_state, u, m):
        """

        :param d_hidden_state: (batch_size=no_reviews, -1, 2*hidden_size)
        :param u: (batch_size, 2*hidden_size)
        :return: d_u: (batch_size, 2*hidden_size)
        """
        h_projection = self.linear_suh(d_hidden_state)
        u_projection = self.linear_suu(u)
        add_projection = h_projection + u_projection[:, None, :]
        t = torch.tanh(add_projection)
        mask = m.unsqueeze(-1)
        e = self.v_su(t)
        e_ = e.masked_fill(mask == 1, 1e-9)
        beta = torch.nn.functional.softmax(e_, dim=-2)
        d_u = torch.matmul(d_hidden_state.transpose(-1, -2), beta).squeeze()
        return d_u

    def pa_w(self, s_hidden_state, p, m):
        h_projection = self.linear_wph(s_hidden_state)
        p_projection = self.linear_wpp(p)
        add_projection = h_projection + p_projection[:, None, None, :]
        t = torch.tanh(add_projection)
        mask = m.unsqueeze_(-1)
        e = self.v_wp(t)
        e_ = e.masked_fill(mask == 1, 1e-9)
        alpha = torch.nn.functional.softmax(e_, dim=-2)
        s_p = torch.matmul(s_hidden_state.transpose(-1, -2), alpha).squeeze()
        return s_p

    def pa_s(self, d_hidden_state, p, m):
        h_projection = self.linear_sph(d_hidden_state)
        p_projection = self.linear_spp(p)
        add_projection = h_projection + p_projection[:, None, :]
        t = torch.tanh(add_projection)
        mask = m.unsqueeze(-1)
        e = self.v_sp(t)
        e_ = e.masked_fill(mask == 1, 1e-9)
        beta = torch.nn.functional.softmax(e_, dim=-2)
        d_p = torch.matmul(d_hidden_state.transpose(-1, -2), beta).squeeze()
        return d_p
