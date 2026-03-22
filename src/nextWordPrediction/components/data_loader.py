import torch
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
      
      def __init__(self, x, y):
            self.x = torch.tensor(x, dtype=torch.long)
            self.y = torch.tensor(y, dtype=torch.long)
      

      def __len__(self):
            return len(self.x)


      def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

