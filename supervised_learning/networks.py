import torch.nn as nn
import torch
from torch.autograd import Variable

class MLP(nn.Module):

    def __init__(self, n_input, n_output, n_hidden):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def reset(self):
        pass


class RNN(nn.Module):

    def __init__(self, n_input, n_output, n_hidden):

        super(RNN, self).__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_output)

        self.h0 = self.c0 = None

    def reset(self):
        self.h0 = self.c0 = None

    def detach_(self):
        self.h0.detach_()
        self.c0.detach_()

    def forward(self, x):

        # Set initial states
        if not self.h0:
            self.h0 = Variable(torch.zeros(x.size(0), 1, self.lstm.hidden_size))
            self.c0 = Variable(torch.zeros(x.size(0), 1, self.lstm.hidden_size))

        out, (self.h0, self.c0) = self.lstm(torch.unsqueeze(x,0), (self.h0, self.c0))

        out = self.fc(torch.squeeze(out))

        return out