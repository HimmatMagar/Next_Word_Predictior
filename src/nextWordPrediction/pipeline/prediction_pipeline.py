import mlflow
import torch
import pickle
import numpy as np
from nltk import word_tokenize

class PredictionPipeline:
      def __init__(self):
            self.model = mlflow.pytorch.load_model("models:/PyTorchModel/Production")

            with open("artifact/run_id.txt", "r") as f:
                  run_id = f.read()

            vocab_path = mlflow.artifacts.download_artifacts(
                  run_id=run_id,
                  artifact_path="vecob/tokenizer.pkl"
            )

            with open(vocab_path, "rb") as f:
                  self.vocab = pickle.load(f)


      def transform_input(self, text):
            text = text.lower()
            input_seq = []
            for token in word_tokenize(text):
                  if token in self.vocab:
                        input_seq.append(self.vocab[token])
                  else:
                        input_seq.append(self.vocab['<UNK>'])

            return input_seq


      def predict_next_word(self, input_seq, k=5):
            self.model.eval()

            # Build reverse vocab map
            idx2word = {v: k for k, v in self.vocab.items()}

            # Prepare input — add batch dimension + send to correct device
            device = next(self.model.parameters()).device
            x = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad():
                  output = self.model(x)
                  probs  = torch.softmax(output, dim=1)
                  top_probs, top_indices = torch.topk(probs, k, dim=1)

            return [idx2word[idx.item()] for idx, prob in zip(top_indices[0], top_probs[0])]