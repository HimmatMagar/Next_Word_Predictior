import os
import torch
import torch.nn as nn
from pathlib import Path
from nextWordPrediction import logger
from nextWordPrediction.components.model_building import LSTM
from nextWordPrediction.components.data_loader import CreateDataLoader
from nextWordPrediction.utils import load_file
from nextWordPrediction.entity import ModelBuildingConfig


class TrainModel:

      def __init__(self, config: ModelBuildingConfig):
            self.config = config
            self.vocab = load_file(Path("artifact/data_transformation/tokenizer.pkl"))
      
      def prepare_data(self):
            text = load_file(Path(self.config.input_file))
            chunk = CreateDataLoader(text, self.vocab)
            return chunk

      def train_model(self):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            criterion = nn.CrossEntropyLoss()
            model = LSTM(
                  len(self.vocab),
                  embedding_unit=self.config.embedding_units,
                  lstm_unit=self.config.lstm_unit,
                  num_layer=self.config.num_layer
            ).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=self.config.learning_rate)
            chunk = self.prepare_data()


            for i in range(self.config.epochs):
                  total_loss = 0
                  for x, y in chunk:
                        x, y = x.to(device), y.tp(device)
                        optimizer.zero_grad()

                        output = model(x)
                        loss = criterion(output, y[:, -1])
                        loss.backward()

                        optimizer.step()
                        total_loss += loss.item()
                  print(f"Epochs {i}: Loss {total_loss / len(chunk)}")
            model_path = os.path.join(self.config.root_dir, self.config.model)
            with open(model_path, "wb") as f:
                  torch.save(model, f)
            logger.info(f"PyTorch model saved in {model_path}")
            
            loss = total_loss/len(chunk)
            return model, loss