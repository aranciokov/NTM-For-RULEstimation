import torch
import torch.nn as nn
import torch.nn.functional as F

 
class Net(nn.Module):

    def __init__(self, input_size, hidden_size, memory_bank_size,
                 ff_size, output_size, batch_size, dropout, is_cuda, num_layers,
                 write_fc_drop=0.10, read_fc_drop=0.10, dec_drop=0.25):
        super(Net, self).__init__()
        self.name = "NTM"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_bank_size = memory_bank_size
        self.num_layers = num_layers
        self.is_cuda = is_cuda
        self.batch_size = batch_size

        self.hidden_to_content = nn.Linear(input_size + hidden_size, hidden_size)
        self.write_gate = nn.Linear(hidden_size + input_size, 1)
        self.write_prob = nn.Linear(hidden_size + hidden_size, memory_bank_size)

        self.read_gate = nn.Linear(hidden_size + input_size, 1)
        self.read_prob = nn.Linear(hidden_size + hidden_size, memory_bank_size)
        
        self.write_fc_drop = nn.Dropout(p=write_fc_drop)
        self.read_fc_drop = nn.Dropout(p=read_fc_drop)
        
        self.decoder1 = nn.Linear(hidden_size, ff_size)
        self.dec_drop = nn.Dropout(p=dec_drop)
        self.decoder2 = nn.Linear(ff_size, output_size)

        self.Wxh = nn.Parameter(torch.FloatTensor(input_size, hidden_size), requires_grad=True)
        self.Wrh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.wa = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.wb = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(1, hidden_size), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        self.Wxh.data.normal_(0.0, 0.01)
        self.Wrh.data.normal_(0.0, 0.01)
        self.Whh.data.normal_(0.0, 0.01)
        self.wa.data.normal_(0.0, 0.01)
        self.wb.data.normal_(0.0, 0.01)
        self.bh.data.fill_(0)

    def forward(self, sensor_measurements, seq_length):
        memory_bank = torch.zeros(self.batch_size, self.memory_bank_size, self.hidden_size).cuda()
        
        h_t = torch.zeros(self.batch_size, 1, self.hidden_size).cuda()

        hiddens = torch.zeros(self.batch_size, seq_length, self.hidden_size).cuda()

        for t in range(seq_length):
            # collect the sensor values measured at time t
            o_t = sensor_measurements[:, t:t + 1, :]
            # since the concatenation of o_t and h_{t-1} is used in several equations, we concat them once in o_h_t
            o_h_t = torch.cat([o_t, h_t], dim=-1)

            # Eq(1), creation of the content vector c_t
            c_t = self.write_fc_drop(torch.sigmoid(self.hidden_to_content(o_h_t)))

            # Eq(2), creation of the write vector a_t
            c_h_t = torch.cat([c_t, h_t], dim=-1)
            a_t = self.write_fc_drop(torch.tanh(self.write_prob(c_h_t)))
            # Eq(3), computation of the (write) attention weights
            alpha_t = torch.softmax(torch.bmm(self.wa.unsqueeze(0).expand(a_t.shape[0], -1, -1), a_t), dim=-1)
            alpha_t = alpha_t.view(self.batch_size, self.memory_bank_size, 1)
            gw = self.write_fc_drop(torch.sigmoid(self.write_gate(o_h_t)))

            # creation of the read vector and computation of the (read) attention weights
            tmp_r = self.read_fc_drop(torch.tanh(self.read_prob(c_h_t)))
            ar = torch.softmax(torch.bmm(self.wb.unsqueeze(0).expand(tmp_r.shape[0], -1, -1), tmp_r), dim=-1)
            go = self.read_fc_drop(torch.sigmoid(self.read_gate(o_h_t)))
            r = go * torch.bmm(ar, memory_bank)

            # Eq(5), update of the hidden state h_t
            m1 = torch.bmm(o_t, self.Wxh.unsqueeze(0).expand(o_t.shape[0], -1, -1))
            m2 = torch.bmm(r, self.Wrh.unsqueeze(0).expand(r.shape[0], -1, -1))
            m3 = torch.bmm(h_t, self.Whh.unsqueeze(0).expand(h_t.shape[0], -1, -1))
            h_t_p1 = torch.sigmoid(m1 + m2 + m3 + self.bh.unsqueeze(0).expand(m1.shape[0], -1, -1))

            # Eq(4), update of the memory
            memory_bank = gw * alpha_t * c_t + (1.0 - alpha_t) * memory_bank

            h_t = h_t_p1
            
            # each of the hidden states is kept, in order to use them as the automatically extracted features of the input time series
            hiddens[:, t, :] = h_t_p1.squeeze(1)

        # Eq(6) and Eq(7) are used to compute the final prediction for the RUL
        decoded = self.dec_drop(torch.sigmoid(self.decoder1(hiddens)))  # seq_length x hidden_size -> seq_length x ff_size
        final_decoded = self.decoder2(decoded)  # seq_length x 1

        return final_decoded  # batch_size x seq_length x 1
