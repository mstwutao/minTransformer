import math

import torch
from torch import nn

from config import Config
from tokenizer import Tokenizer

de_vocab_size, en_vocab_size = Tokenizer().build_vocab()


class PositionalEmbedding(nn.Module):

    def __init__(self, config, vocab_size, dropout=0.1):
        super().__init__()
        self.seq_emb = nn.Embedding(vocab_size, config.d_model)
        pe = torch.zeros(config.max_seq_len, config.d_model)
        position_idx = torch.arange(0, config.max_seq_len).unsqueeze(-1)
        position_fill = position_idx * torch.exp(
            -torch.arange(0, config.d_model, 2) * math.log(10000.0) /
            config.d_model)
        pe[:, 0::2] = torch.sin(position_fill)
        pe[:, 1::2] = torch.cos(position_fill)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.seq_emb(x)  # x: (batch_size, seq_len, d_model)
        x = x + self.pe.unsqueeze(0)[:, :x.size(1), :]
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_qk = config.d_qk
        self.d_v = config.d_v

        self.w_q = nn.Linear(config.d_model, config.n_heads * self.d_qk)
        self.w_k = nn.Linear(config.d_model, config.n_heads * self.d_qk)
        self.w_v = nn.Linear(config.d_model, config.n_heads * self.d_v)

    def forward(self, q, k, v, attn_mask):
        q = self.w_q(q)  # q: (batch_size, seq_len, n_heads * d_qk)
        k = self.w_k(k)  # k: (batch_size, seq_len, n_heads * d_qk)

        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_qk).transpose(
            1, 2)  # q: (batch_size, n_heads, seq_len, d_qk)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_qk).transpose(
            1, 2).transpose(2, 3)  # k: (batch_size, n_heads, d_qk, seq_len)

        attn = torch.matmul(q, k) / math.sqrt(
            self.d_qk)  # attn: (batch_size, n_heads, seq_len, seq_len)

        attn_mask = attn_mask.unsqueeze(1).expand(
            -1, self.n_heads, -1,
            -1)  # attn_mask: (batch_size, n_heads, seq_len, seq_len)
        attn = attn.masked_fill(attn_mask == 0, -1e9)

        attn_scores = torch.softmax(
            attn,
            dim=-1)  # attn_scores: (batch_size, n_heads, seq_len, seq_len)

        v = self.w_v(v)  # v: (batch_size, seq_len, n_heads * d_v)
        v = v.view(v.size()[0],
                   v.size()[1], self.n_heads, self.d_v).transpose(
                       1, 2)  # v: (batch_size, n_heads, seq_len, d_v)

        z = torch.matmul(attn_scores,
                         v)  # z: (batch_size, n_heads, seq_len, d_v)
        z = z.transpose(1, 2)  # z: (batch_size, seq_len, n_heads, d_v)
        z = z.reshape(z.size(0), z.size(1),
                      -1)  # z: (batch_size, seq_len, n_heads * d_v)
        return z


class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.multihead_attn = MultiHeadAttention(config)
        self.z_linear = nn.Linear(config.d_model, config.d_model)
        self.addnorm1 = nn.LayerNorm(config.d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim), nn.ReLU(),
            nn.Linear(config.hidden_dim, config.d_model))
        self.addnorm2 = nn.LayerNorm(config.d_model)

    def forward(self, x, attn_mask):  # x: (batch_size,seq_len,emb_size)
        z = self.multihead_attn(
            x, x, x, attn_mask)  # z: (batch_size,seq_len,head*v_size)
        z = self.z_linear(z)  # z: (batch_size,seq_len,emb_size)
        output1 = self.addnorm1(z + x)  # z: (batch_size,seq_len,emb_size)

        z = self.feedforward(output1)  # z: (batch_size,seq_len,emb_size)
        return self.addnorm2(z + output1)  # (batch_size,seq_len,emb_size)


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = PositionalEmbedding(config, de_vocab_size)

        self.encoder_blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            self.encoder_blocks.append(EncoderBlock(config))

    def forward(self, x):  # x: (batch_size, seq_len)
        pad_mask = (x == self.config.PAD_IDX).unsqueeze(
            1)  # pad_mask: (batch_size, 1, seq_len)
        pad_mask = pad_mask.expand(
            x.size()[0],
            x.size()[1],
            x.size()[1],
        )  # pad_mask: (batch_size, seq_len, seq_len)

        pad_mask = pad_mask.to(Config.device)

        x = self.emb(x)
        for block in self.encoder_blocks:
            x = block(x, pad_mask)  # x: (batch_size, seq_len, emb_size)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.first_multihead_attn = MultiHeadAttention(config)
        self.z_linear1 = nn.Linear(config.d_model, config.d_model)
        self.addnorm1 = nn.LayerNorm(config.d_model)

        self.second_multihead_attn = MultiHeadAttention(config)
        self.z_linear2 = nn.Linear(config.d_model, config.d_model)
        self.addnorm2 = nn.LayerNorm(config.d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim), nn.ReLU(),
            nn.Linear(config.hidden_dim, config.d_model))
        self.addnorm3 = nn.LayerNorm(config.d_model)

    def forward(self, dec_input, enc_output, first_attn_mask,
                second_attn_mask):

        # z: (batch_size,seq_len,head*v_size) , first_attn_mask pad
        z = self.first_multihead_attn(dec_input, dec_input, dec_input,
                                      first_attn_mask)
        z = self.z_linear1(z)  # z: (batch_size,seq_len,emb_size)
        output1 = self.addnorm1(z +
                                dec_input)  # x: (batch_size,seq_len,emb_size)

        # z: (batch_size,seq_len,head*v_size)   , second_attn_mask
        z = self.second_multihead_attn(output1, enc_output, enc_output,
                                       second_attn_mask)
        z = self.z_linear2(z)  # z: (batch_size,seq_len,emb_size)
        output2 = self.addnorm2(z +
                                output1)  # x: (batch_size,seq_len,emb_size)

        # feedforward
        z = self.feedforward(output2)  # z: (batch_size,seq_len,emb_size)
        return self.addnorm3(z + output2)  # (batch_size,seq_len,emb_size)


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = PositionalEmbedding(config, en_vocab_size)

        self.decoder_blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            self.decoder_blocks.append(DecoderBlock(config))

        # prediction logits
        self.linear = nn.Linear(config.d_model, en_vocab_size)

    def forward(self, dec_input, enc_output, enc_input):
        first_attn_mask = (
            dec_input == self.config.PAD_IDX).unsqueeze(1).expand(
                dec_input.size()[0],
                dec_input.size()[1],
                dec_input.size()[1]).to(self.config.device)  # padding mask
        first_attn_mask = first_attn_mask | torch.triu(
            torch.ones(
                dec_input.size()[1],
                dec_input.size()[1]), diagonal=1).bool().unsqueeze(0).expand(
                    dec_input.size()[0], -1, -1).to(
                        self.config.device)  # subsequent mask

        second_attn_mask = (
            enc_input == self.config.PAD_IDX).unsqueeze(1).expand(
                enc_input.size()[0],
                dec_input.size()[1],
                enc_input.size()[1]).to(self.config.device)  # padding mask

        x = self.emb(dec_input)
        for block in self.decoder_blocks:
            x = block(x, enc_output, first_attn_mask, second_attn_mask)

        return self.linear(x)  # (batch_size, target_len, vocab_size)


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, enc_input):
        enc_output = self.encoder(enc_input)
        return enc_output

    def decode(self, dec_input, enc_output, enc_input):
        decoder_z = self.decoder(dec_input, enc_output, enc_input)
        return decoder_z

    def forward(self, enc_input, dec_input):
        enc_output = self.encode(enc_input)
        return self.decode(dec_input, enc_output, enc_input)


if __name__ == '__main__':

    def test_PositionalEmbedding():
        from tokenizer import Tokenizer
        emb = PositionalEmbedding(Config, de_vocab_size)
        tokenizer = Tokenizer()
        de_sentence, en_sentence = tokenizer.train_dataset[0]
        de_ids = tokenizer.encode(de_sentence, 'de')
        de_ids_tensor = torch.tensor(de_ids, dtype=torch.long).unsqueeze(0)
        emb_result = emb(de_ids_tensor)
        print('de_tensor_size:', de_ids_tensor.size(), 'emb_size:',
              emb_result.size())

    #test_PositionalEmbedding()

    def test_MultiHeadAttention():
        from tokenizer import Tokenizer
        emb = PositionalEmbedding(Config, de_vocab_size)
        tokenizer = Tokenizer()
        de_sentence, en_sentence = tokenizer.train_dataset[0]
        de_ids = tokenizer.encode(de_sentence, 'de')
        de_ids_tensor = torch.tensor(de_ids, dtype=torch.long).unsqueeze(0)
        emb_result = emb(de_ids_tensor)
        multihead = MultiHeadAttention(Config)
        attn_mask = torch.zeros((1, 15, 15))
        multihead_result = multihead(emb_result,
                                     emb_result,
                                     emb_result,
                                     attn_mask=attn_mask)
        print('multihead_result:', multihead_result.size())

    #test_MultiHeadAttention()

    def test_EncoderBlock():
        from tokenizer import Tokenizer
        emb = PositionalEmbedding(Config, de_vocab_size)
        tokenizer = Tokenizer()
        de_sentence, en_sentence = tokenizer.train_dataset[0]
        de_ids = tokenizer.encode(de_sentence, 'de')
        de_ids_tensor = torch.tensor(de_ids, dtype=torch.long).unsqueeze(0)
        emb_result = emb(de_ids_tensor)
        eb = EncoderBlock(Config)
        attn_mask = torch.zeros((1, 15, 15))
        eb_outputs = eb(emb_result, attn_mask)
        print('encoder_outputs:', eb_outputs.size())

    #test_EncoderBlock()

    def test_Encoder():
        from tokenizer import Tokenizer
        emb = PositionalEmbedding(Config, de_vocab_size)
        tokenizer = Tokenizer()
        de_sentence, en_sentence = tokenizer.train_dataset[0]
        de_ids = tokenizer.encode(de_sentence, 'de')
        de_ids_tensor = torch.tensor(de_ids, dtype=torch.long).unsqueeze(0)
        enc = Encoder(Config)
        enc_outputs = enc(de_ids_tensor)
        print('encoder_outputs:', enc_outputs.size())

    #test_Encoder()

    def test_DecoderBlock():
        from tokenizer import Tokenizer
        emb = PositionalEmbedding(Config, en_vocab_size)
        tokenizer = Tokenizer()
        de_sentence, en_sentence = tokenizer.train_dataset[0]
        en_ids = tokenizer.encode(en_sentence, 'en')
        en_ids_tensor = torch.tensor(en_ids, dtype=torch.long).unsqueeze(0)
        emb_result = emb(en_ids_tensor)
        eb = EncoderBlock(Config)
        attn_mask = torch.zeros(
            (1, en_ids_tensor.size(1), en_ids_tensor.size(1)))
        eb_outputs = eb(emb_result, attn_mask)
        print('encoder_outputs:', eb_outputs.size())

    #test_EncoderBlock()

    def test_Decoder():
        from tokenizer import Tokenizer
        emb = PositionalEmbedding(Config, de_vocab_size)
        tokenizer = Tokenizer()
        de_sentence, en_sentence = tokenizer.train_dataset[0]
        de_ids = tokenizer.encode(de_sentence, 'de')
        de_ids_tensor = torch.tensor(de_ids, dtype=torch.long).unsqueeze(0)
        enc = Encoder(Config)
        enc_outputs = enc(de_ids_tensor)
        print('encoder_outputs:', enc_outputs.size())

        en_ids = tokenizer.encode(en_sentence, 'en')
        en_ids_tensor = torch.tensor(en_ids, dtype=torch.long).unsqueeze(0)
        dec = Decoder(Config)
        dec_outputs = dec(en_ids_tensor, enc_outputs, de_ids_tensor)
        print('decoder outputs:', dec_outputs.size())

    #test_Decoder()

    def test_Transformer():
        from tokenizer import Tokenizer
        emb = PositionalEmbedding(Config, de_vocab_size)
        tokenizer = Tokenizer()
        de_sentence, en_sentence = tokenizer.train_dataset[0]
        de_ids = tokenizer.encode(de_sentence, 'de')
        de_ids_tensor = torch.tensor(de_ids, dtype=torch.long).unsqueeze(0)
        enc = Encoder(Config)
        enc_outputs = enc(de_ids_tensor)
        print('encoder_outputs:', enc_outputs.size())

        en_ids = tokenizer.encode(en_sentence, 'en')
        en_ids_tensor = torch.tensor(en_ids, dtype=torch.long).unsqueeze(0)
        dec = Decoder(Config)
        dec_outputs = dec(en_ids_tensor, enc_outputs, de_ids_tensor)
        print('decoder outputs:', dec_outputs.size())

        transformer = Transformer(Config)
        transformer_output = transformer(de_ids_tensor, en_ids_tensor)
        print('transformer output:', transformer_output.size())

    test_Transformer()
