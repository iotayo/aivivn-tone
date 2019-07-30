import unidecode
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphabet import AEIOUYD_VN, A_LIST, E_LIST, I_LIST, O_LIST, U_LIST, Y_LIST, D_LIST
from dataset import src_field_name, tgt_field_name, SOS, EOS, PAD, PUNCT, OTHER


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Encoder(nn.Module):
    """ BiLSTM encoder with variable-length input """
    def __init__(self, vocab, max_len, hidden_size, n_layers,
                 input_dropout_p=0.1, dropout_p=0.1):
        super(Encoder, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_len = max_len
        self.embed_size = hidden_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        self.input_dropout = nn.Dropout(p=input_dropout_p)

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.n_layers, batch_first=True,
                            bidirectional=True, dropout=self.dropout_p)

    def forward(self, inputs, input_lengths):
        embedded = self.embed(inputs)

        embedded = self.input_dropout(embedded)

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        output, hidden = self.lstm(embedded)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # output = (batch, seq_len, 2*hidden_size)
        # hidden = (n_layers*2, batch, hidden_size)
        return output, hidden


class Attention(nn.Module):
    """ Dot attention """
    def __init__(self, dec_hidden_size):
        super(Attention, self).__init__()

        self.dec_hidden_size = dec_hidden_size
        self.linear = nn.Linear(dec_hidden_size * 2, dec_hidden_size)

    def forward(self, dec_output, enc_output, mask):
        batch_size = dec_output.size(0)
        dec_hidden_size = dec_output.size(2)
        enc_seq_len = enc_output.size(1)

        # dot-product attention
        # dec_seq_len should be 1
        # in this model, dec_hidden_size = 2*enc_hidden_size
        # (batch, dec_seq_len, dec_hidden_size) * (batch, 2*enc_hidden_size, enc_seq_len) = (batch, dec_seq_len, enc_seq_len)
        attn_w = torch.bmm(dec_output, enc_output.transpose(1, 2))

        # don't attend to pads
        mask = None
        if mask is not None:
            attn_w.data.masked_fill_(mask, -float('inf'))

        attn_w = F.softmax(attn_w.view(-1, enc_seq_len), dim=1).view(batch_size, -1, enc_seq_len)

        # (batch, dec_seq_len, enc_seq_len) * (batch, enc_seq_len, dec_hidden_size) = (batch, dec_seq_len, dec_hidden_size)
        attn = torch.bmm(attn_w, enc_output)

        # concat -> (batch, dec_seq_len, 2*dec_hidden_size)
        combined = torch.cat((attn, dec_output), dim=2)

        # output = (batch, dec_seq_len, dec_hidden_size)
        output = torch.tanh(self.linear(combined.view(-1, 2*dec_hidden_size))).view(batch_size, -1, dec_hidden_size)

        return output


class Decoder(nn.Module):
    """ LSTM decoder """
    def __init__(self, vocab, max_len, hidden_size, n_layers,
                 input_dropout_p=0.1, dropout_p=0.1, reverse=False):
        super(Decoder, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_len = max_len
        self.embed_size = hidden_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.reverse = reverse

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        self.input_dropout = nn.Dropout(p=input_dropout_p)

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.n_layers,
                            batch_first=True, dropout=self.dropout_p)

        self.attention = Attention(self.hidden_size)

        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

        # get indices of each aeiouyd token in vocab
        self.aeiouyd_indices = {}
        for li in [A_LIST, E_LIST, I_LIST, O_LIST, U_LIST, Y_LIST, D_LIST]:
            li_indices = torch.zeros(len(li)).to(device)
            for i in range(len(li)):
                li_indices[i] = self.vocab.stoi[li[i]]
            self.aeiouyd_indices[li[0]] = li_indices

        # create softmax mask for each token in the vocab
        self.softmax_masks = torch.ones(self.vocab_size, self.vocab_size).to(device)

        # create teacher forcing masks
        self.teacher_forcing_masks = torch.ones(self.vocab_size).long().to(device)

        for idx in range(self.vocab_size):
            mask = torch.ones(self.vocab_size).to(device)

            token = self.vocab.itos[idx]

            if token in AEIOUYD_VN:
                # don't fill logits at these positions with -inf
                token_indices = self.aeiouyd_indices[unidecode.unidecode(token)]
                mask[token_indices.long()] = 0

                # don't use teacher forcing on aeiouyd tokens
                self.teacher_forcing_masks[idx] = 0
            else:
                mask[self.vocab.stoi[token]] = 0

            self.softmax_masks[idx] = mask

    def forward(self, targets, enc_output, enc_hidden, mask, mask_softmax=0.5, teacher_forcing=0.5):
        # assert inputs.size(1) == targets.size(1)

        batch_size = targets.size(0)
        max_len = targets.size(1)

        # first input hidden to decoder is enc_hidden
        # (n_layers, batch, 2*enc_hidden_size) = (n_layers, batch, dec_hidden_size)
        dec_hidden = tuple([self._cat_directions(h) for h in enc_hidden])

        # if reverse then reverse inputs and targets
        if self.reverse:
            targets = targets.flip(1)
            enc_output = torch.cat((enc_output[:, :, (self.hidden_size//2):], enc_output[:, :, :(self.hidden_size//2)]), dim=2)

        # first input token to decoder is first target token
        # (batch, 1)
        dec_input = targets[:, 0].unsqueeze(1)

        # tensor to store decoder output probabilities after softmax layer
        # not to be confused with decoder lstm output
        dec_probs = torch.zeros(batch_size, max_len, self.vocab_size).to(device)

        # tensor to store decoder lstm output
        dec_outputs = torch.zeros(batch_size, max_len, self.hidden_size).to(device)

        # tensor to store softmax masks for entire seq_len
        softmax_masks = torch.ones(batch_size, max_len, self.vocab_size).to(device)
        _, first_softmax_mask = self._check_token_(targets[:, 0], mask_softmax=mask_softmax, teacher_forcing=teacher_forcing)
        softmax_masks[:, 0, :] = first_softmax_mask

        for t in range(1, max_len):
            # (batch, 1, dec_hidden_size)
            embedded = self.embed(dec_input)
            embedded = self.input_dropout(embedded)

            # dec_output = (batch, 1, dec_hidden_size)
            # dec_hidden = (n_layers, batch, dec_hidden_size)
            dec_output, dec_hidden = self.lstm(embedded, dec_hidden)

            # (batch, 1, dec_hidden_size)
            dec_output = self.attention(dec_output, enc_output, mask)

            # check aeiouyd and get softmax mask
            is_aeiouyd, softmax_mask = self._check_token_(targets[:, t], mask_softmax=mask_softmax, teacher_forcing=teacher_forcing)

            # prediction layer
            # (batch, tgt_vocab_size)
            logits = self.linear(dec_output.view(-1, self.hidden_size))

            # apply mask before softmax
            logits.masked_fill_(softmax_mask.byte(), -float('inf'))

            # log_softmax to go with NLLLoss
            probs = F.log_softmax(logits, dim=-1)

            # (batch,)
            predicted = probs.max(dim=1)[1]

            # teacher forcing for non-aeiouyd tokens
            if teacher_forcing > 0.0:
                dec_input = targets[:, t] * is_aeiouyd + predicted * (1 - is_aeiouyd)
                dec_input = dec_input.unsqueeze(1)
            else:
                dec_input = predicted.unsqueeze(1)

            # return probs and dec_output
            dec_probs[:, t, :] = probs
            dec_outputs[:, t, :] = dec_output.squeeze(1)
            softmax_masks[:, t, :] = softmax_mask

        # if reverse then reverse dec_probs to match original sequence order
        if self.reverse:
            dec_probs = dec_probs.flip(1)
            dec_outputs = dec_outputs.flip(1)
            softmax_masks = softmax_masks.flip(1)

        # dec_outputs = (batch, max_len, dec_hidden_size)
        return dec_probs, dec_outputs, softmax_masks

    def _cat_directions(self, h):
        # (2*n_layers, batch, enc_hidden_size) -> (n_layers, batch, 2*enc_hidden_size)
        if self.reverse:
            h = torch.cat([h[1:h.size(0):2], h[0:h.size(0):2]], 2)
        else:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _check_token_(self, indices, mask_softmax=0.5, teacher_forcing=0.5):
        # indices = (batch_size,)
        batch_size = indices.size(0)

        # get softmax mask
        softmax_mask = torch.index_select(self.softmax_masks, 0, indices)

        # get teacher forcing mask
        teacher_forcing_mask = torch.index_select(self.teacher_forcing_masks, 0, indices)

        # apply softmax mask ratio
        if 0.0 <= mask_softmax < 1.0:
            softmax_mask_ratio = torch.rand(batch_size, 1, device=device).lt(mask_softmax).repeat(1, self.vocab_size).float()
            softmax_mask = softmax_mask.mul(softmax_mask_ratio)

        # apply teacher forcing ratio
        if 0.0 <= teacher_forcing < 1.0:
            teacher_forcing_ratio = torch.rand(batch_size, device=device).lt(teacher_forcing).long()
            teacher_forcing_mask = teacher_forcing_mask.mul(teacher_forcing_ratio)

        return teacher_forcing_mask, softmax_mask


class Seq2SeqConcat(nn.Module):
    """ Encoder-decoder seq2seq model """
    def __init__(self, encoder, decoder, reverse_decoder, pad_idx=1):
        super(Seq2SeqConcat, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reverse_decoder = reverse_decoder
        self.pad_idx = pad_idx

        self.linear = nn.Linear(self.decoder.hidden_size * 2, self.decoder.vocab_size)

    def forward(self, batch, has_targets=True, mask_softmax=0.5, teacher_forcing=0.5):
        inputs, input_lengths = getattr(batch, src_field_name)
        if has_targets:
            targets = getattr(batch, tgt_field_name)
        else:
            targets = inputs.clone()
            targets2 = inputs.clone()
            # map src_vocab to tgt_vocab
            for i in range(len(self.encoder.vocab)):
                targets[targets2 == i] = self.decoder.vocab.stoi[self.encoder.vocab.itos[i]]

        # create attention mask; don't attend to pads
        mask = (inputs == self.pad_idx).unsqueeze(1)

        enc_output, enc_hidden = self.encoder(inputs, input_lengths)

        dec_probs, dec_outputs, softmax_masks = self.decoder(targets, enc_output, enc_hidden, mask,
                                                             mask_softmax=mask_softmax, teacher_forcing=teacher_forcing)

        reverse_dec_probs, reverse_dec_outputs, reverse_softmax_masks = self.reverse_decoder(targets, enc_output, enc_hidden, mask,
                                                                                             mask_softmax=mask_softmax, teacher_forcing=teacher_forcing)

        # (batch, seq_len, dec_hidden_size*2)
        concat_outputs = torch.cat((dec_outputs, reverse_dec_outputs), dim=2)

        # prediction layer
        # (batch, seq_len, tgt_vocab_size)
        logits = self.linear(concat_outputs)

        # apply mask before softmax
        # assert torch.eq(softmax_masks, reverse_softmax_masks).sum().item() == softmax_masks.numel()
        combined_softmax_masks = softmax_masks + reverse_softmax_masks
        combined_softmax_masks = (combined_softmax_masks == 2)
        logits.masked_fill_(combined_softmax_masks.byte(), -float('inf'))

        # log_softmax to go with NLLLoss
        probs = F.log_softmax(logits, dim=-1)

        return dec_probs, reverse_dec_probs, probs
