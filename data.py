import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import Config
from tokenizer import Tokenizer


class De2EnDataset(Dataset):

    def __init__(self, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        self.de_token_ids = []
        self.en_token_ids = []
        for de_sentence, en_sentence in tokenizer.train_dataset:
            de_ids = tokenizer.encode(de_sentence, 'de')
            en_ids = tokenizer.encode(en_sentence, 'en')
            if len(de_ids) > config.max_seq_len or len(
                    en_ids) > config.max_seq_len:
                continue
            self.de_token_ids.append(de_ids)
            self.en_token_ids.append(en_ids)

    def __len__(self):
        return len(self.de_token_ids)

    def __getitem__(self, index):
        return self.de_token_ids[index], self.en_token_ids[index]


def collate_fn(batch):
    de_batch, en_batch = [], []
    for de, en in batch:
        de_batch.append(torch.tensor(de, dtype=torch.long))
        en_batch.append(torch.tensor(en, dtype=torch.long))

    pad_de = pad_sequence(de_batch,
                          batch_first=True,
                          padding_value=Config.PAD_IDX)
    pad_en = pad_sequence(en_batch,
                          batch_first=True,
                          padding_value=Config.PAD_IDX)
    return pad_de, pad_en


tokenizer = Tokenizer()
dataset = De2EnDataset(Config, tokenizer)
dataloader = DataLoader(dataset,
                        Config.batch_size,
                        shuffle=True,
                        num_workers=4,
                        persistent_workers=True,
                        collate_fn=collate_fn)

if __name__ == '__main__':

    for pad_de, pad_en in dataloader:
        print('de_ids', pad_de[0])
        print('en_ids', pad_en[0])
        break
