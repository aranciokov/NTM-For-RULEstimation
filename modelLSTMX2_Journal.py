import torch
import torch.nn as nn
import torch.nn.functional as F

 
class Net(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_size2, ff_size, output_size,
                 batch_size, is_cuda, num_layers, dec_drop=0.25, dec_sigm=False):
        super(Net, self).__init__()
        self.name = "LSTMx2"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_cuda = is_cuda
        self.batch_size = batch_size
        self.dec_sigm = dec_sigm

        self.lstm1 = nn.LSTM(input_size, hidden_size2, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size2, hidden_size, batch_first=True)
        self.decoder1 = nn.Linear(hidden_size, ff_size)
        self.dec_drop = nn.Dropout(p=dec_drop)
        self.decoder2 = nn.Linear(ff_size, output_size)

    def forward(self, hidden_frames, seq_length):
        # hidden_frames: (batch, seq, input_size)

        output, (hn, cn) = self.lstm1(hidden_frames)
        output, (hn, cn) = self.lstm2(output)

        if self.dec_sigm:
            decoded = self.dec_drop(torch.sigmoid(self.decoder1(output)))  # seq_length x hidden_size -> seq_length x ff_size
        else:
            decoded = self.dec_drop(self.decoder1(output))  # seq_length x hidden_size -> seq_length x ff_size
        final_decoded = self.decoder2(decoded)  # seq_length x 1

        return final_decoded  # batch_size x seq_length x 1

    def init_hidden(self, hs):
        if self.is_cuda:
            x, y = torch.zeros((self.num_layers, self.batch_size, hs)),\
                  torch.zeros((self.num_layers, self.batch_size, hs))
            return x.cuda(), y.cuda()
        else:
            return (torch.zeros((self.num_layers, self.batch_size, hs)),
                    torch.zeros((self.num_layers, self.batch_size, hs)))
