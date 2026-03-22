import torch
import torch.nn as nn



class LSTM(nn.Module):

      def __init__(self, vocab_size, embedding_unit, lstm_unit):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_unit)
            self.lstm = nn.LSTM(embedding_unit, lstm_unit, batch_first=True)
            self.linear = nn.Linear(lstm_unit, vocab_size)
      

      def forward(self, x):
            embedding_output = self.embedding(x)
            intermediate_hidden_state, (final_hidden_state, final_cell_states) = self.lstm(embedding_output)
            return self.linear(final_hidden_state.squeeze(0))