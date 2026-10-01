import torch
import torch.nn as nn


class LSTM(nn.Module):
      def __init__(self, vocab_size, embedding_unit, lstm_unit, num_layer):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_unit)
            self.lstm = nn.LSTM(embedding_unit, lstm_unit, batch_first=True, num_layers=num_layer)
            self.linear = nn.Linear(lstm_unit, vocab_size)
      

      def forward(self, x):
            embedding_output = self.embedding(x)
            _, (h_n, _) = self.lstm(embedding_output)
            return self.linear(h_n[-1])