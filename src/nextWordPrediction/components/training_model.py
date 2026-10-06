import os
import math
import torch
import torch.nn as nn
from pathlib import Path
from nextWordPrediction import logger
from nextWordPrediction.components.model_building import LSTM
from nextWordPrediction.components.data_loader import CreateDataLoader
from nextWordPrediction.utils import load_file, load_text
from nextWordPrediction.entity import ModelBuildingConfig


class TrainModel:

      def __init__(self, config: ModelBuildingConfig):
            self.config = config
            self.vocab = load_file(Path("artifact/data_transformation/tokenizer.pkl"))
      
      def prepare_data(self):
            train_text = load_text(Path(self.config.input_file))
            val_text = load_text(Path(self.config.val_file))
            train_chunk = CreateDataLoader(train_text, self.vocab)
            val_chunk = CreateDataLoader(val_text, self.vocab)
            return train_chunk, val_chunk
      

      @torch.no_grad()
      def _evaluate(self, model, dataloader, criterion, device):
            model.eval();
            total_loss, total_tokens = 0, 0

            for x, y in dataloader:
                  x, y = x.to(device), y.to(device)
                  output = model(x)
                  loss = criterion(output, y[:, -1])

                  total_loss += loss.item() * y.numel()
                  total_tokens += y.numel()

            avg_loss = total_loss / total_tokens
            return avg_loss
      

      def train_model(self):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            criterion = nn.CrossEntropyLoss()
            model = LSTM(
                  len(self.vocab),
                  embedding_unit=self.config.embedding_units,
                  lstm_unit=self.config.lstm_unit,
                  num_layer=self.config.num_layer,
                  dropout=0.3
            ).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=self.config.learning_rate, weight_decay=1e-2)
            train_loader, val_loader = self.prepare_data()


            for epoch in range(self.config.epochs):
                  model.train()
                  for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        output = model(x)
                        loss = criterion(output, y[:, -1])
                        loss.backward()
                        optimizer.step()
                  
                  train_loss= self._evaluate(
                        model,
                        train_loader,
                        criterion,
                        device
                  )

                  val_loss = self._evaluate(
                        model,
                        val_loader,
                        criterion,
                        device
                  )

                  print(
                        f"Epoch {epoch+1} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f}"
                  )
            model_path = os.path.join(self.config.root_dir, self.config.model)
            with open(model_path, "wb") as f:
                  torch.save(model, f)
            logger.info(f"PyTorch model saved in {model_path}")

            perplexity = math.exp(val_loss)
            
            return model, perplexity