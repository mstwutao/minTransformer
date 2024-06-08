from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

from config import Config


class Tokenizer:

    def __init__(self):
        self.train_dataset = list(
            Multi30k(split='train', language_pair=('de', 'en')))
        self.de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.de_vocab_size, self.en_vocab_size = self.build_vocab()

    def build_vocab(self):
        de_tokens, en_tokens = [], []
        for de_sentence, en_sentence in self.train_dataset:
            de_tokens.append(self.de_tokenizer(de_sentence))
            en_tokens.append(self.en_tokenizer(en_sentence))

        self.de_vocab = build_vocab_from_iterator(iterator=de_tokens,
                                                  specials=[
                                                      Config.UNK_SYM,
                                                      Config.PAD_SYM,
                                                      Config.BOS_SYM,
                                                      Config.EOS_SYM
                                                  ],
                                                  special_first=True)
        self.de_vocab.set_default_index(Config.UNK_IDX)

        self.en_vocab = build_vocab_from_iterator(iterator=en_tokens,
                                                  specials=[
                                                      Config.UNK_SYM,
                                                      Config.PAD_SYM,
                                                      Config.BOS_SYM,
                                                      Config.EOS_SYM
                                                  ],
                                                  special_first=True)
        self.en_vocab.set_default_index(Config.UNK_IDX)
        de_vocab_size = len(self.de_vocab)
        en_vocab_size = len(self.en_vocab)
        return de_vocab_size, en_vocab_size

    def encode(self, inputs, language='de'):
        if language == 'de':
            tokens = self.de_tokenizer(inputs)
            token_ids = self.de_vocab(tokens)
        elif language == 'en':
            tokens = self.en_tokenizer(inputs)
            token_ids = self.en_vocab(tokens)
        token_ids = [Config.BOS_IDX] + token_ids + [Config.EOS_IDX]
        return token_ids


if __name__ == '__main__':

    def test_Tokenizer():
        tokenizer = Tokenizer()
        print('de_vocab_size', tokenizer.de_vocab_size)
        print('en_vocab_size', tokenizer.en_vocab_size)

        de_sentence, en_sentence = tokenizer.train_dataset[0]
        de_ids = tokenizer.encode(de_sentence, 'de')
        en_ids = tokenizer.encode(en_sentence, 'en')
        print('de_ids', de_ids)
        print('en_ids', en_ids)

    test_Tokenizer()
