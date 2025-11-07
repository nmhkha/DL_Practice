import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #Pytorch's built-in LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #Fully connected layer to map hidden state to ouput
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden):
        # Input shape: (batch_size, seq_len, input_size)
        # Hidden shape: (num_layers, batch_size, hidden_size) - initialized outside
        # Cell state shape: (num_layers, batch_size, hidden_size) - initialized outside

        # out shape: (batch_size, seq_len, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell state shape: (num_layers, batch_size, hidden_size
        if hidden is None:
            batch_size = input_seq.size(0)
            hidden = self.init_hidden(batch_size, input_seq.device)
            
        out, hidden = self.lstm(input_seq, hidden)

        # We often want the output of the last time step for sequence classification
        # out[:, -1, :] has shape (batch_size, hidden_size)\
        output = self.fc(out[:, -1, :])
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden and cell state with zeros
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        return (h0, c0)