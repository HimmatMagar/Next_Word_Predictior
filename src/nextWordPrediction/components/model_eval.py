import math
import torch
import torch.nn as nn
from pathlib import Path
from nextWordPrediction.utils import *
from nextWordPrediction.entity import ModelEvalConfig
from nextWordPrediction.components.data_loader import CreateDataLoader

class ModelEval:
      def __init__(self, config: ModelEvalConfig):
            self.config = config
            self.vocab = load_file(Path("artifact/data_transformation/tokenizer.pkl"))

      
      def _prepare_data(self):
            test_text = load_text(Path(self.config.test_file_path))
            test_chunk = CreateDataLoader(test_text, self.vocab, shuffle=False)
            return test_chunk

      
      @torch.no_grad()
      def _evaluate(self, model, dataloader, criterion, device):
            model.eval()
            total_loss, total_tokens = 0, 0
        
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y[:, -1])
            
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
        
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            return avg_loss, perplexity



      def _evaluate_model(self):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            criterion = nn.CrossEntropyLoss()
            model = torch.load(self.config.model, weights_only=False)
            model.to(device)
            model.eval();
            test_dataset = self._prepare_data()
            loss, perplexity = self._evaluate(model, test_dataset, criterion, device)

            performance_report = {
                  "test_loss": loss,
                  "perplexity": perplexity
            }
            save_json(Path(self.config.metric), performance_report)
            logger.info(f"Model performance report saved successfully in {self.config.metric}")
            return performance_report