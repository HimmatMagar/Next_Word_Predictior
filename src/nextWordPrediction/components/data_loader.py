import torch
from torch.utils.data import Dataset, DataLoader


def numericalize(sentences, vocab):
      return torch.tensor(
            [vocab.get(word, vocab['<UNK>']) for word in sentences.split()],
            dtype=torch.long
      )

class CustomDataSet(Dataset):
      def __init__(self, text, vocab, max_length, stride):
            self.input_seq = []
            self.target_seq = []

            token = numericalize(text, vocab)
            for i in range(0, len(token) - max_length, stride):
                input_chunk = token[i:i + max_length]
                target_chunk = token[i + 1: i + max_length + 1]
                self.input_seq.append(input_chunk)
                self.target_seq.append(target_chunk)

      def __len__(self):
            return len(self.input_seq)

      def __getitem__(self, index):
            return self.input_seq[index], self.target_seq[index]


def CreateDataLoader(txt, vocab, batch_size=4, max_length=32, stride=20, shuffle=True, drop_last=True, num_workers=0):

    dataset = CustomDataSet(txt, vocab, max_length, stride)
    dataloder = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=shuffle,
        drop_last = drop_last,
        num_workers=num_workers
    )
    return dataloder