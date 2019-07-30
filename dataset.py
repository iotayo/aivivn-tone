import codecs
import torchtext
from torchtext.data import Field


src_field_name = "src"
tgt_field_name = "tgt"

SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"
PUNCT = "-"
OTHER = "?"


def _read_corpus(path):
    with codecs.open(path, "r", "utf-8") as file:
        for line in file:
            yield line


class Seq2SeqDataset(torchtext.data.Dataset):

    def __init__(self, examples, src_field, tgt_field=None, **kwargs):
        # construct fields
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.fields = [(src_field_name, src_field)]
        if tgt_field is not None:
            self.fields.append((tgt_field_name, tgt_field))

        # construct examples
        examples = [torchtext.data.Example.fromlist(list(data) + [i], self.fields)
                    for i, data in enumerate(examples)]

        super(Seq2SeqDataset, self).__init__(examples, self.fields, **kwargs)

    @staticmethod
    def from_file(src_path, tgt_path=None, share_fields_from=None, **kwargs):
        src_list = _read_corpus(src_path)
        if tgt_path is not None:
            tgt_list = _read_corpus(tgt_path)
        else:
            tgt_list = None

        return Seq2SeqDataset.from_list(src_list, tgt_list, share_fields_from, **kwargs)

    @staticmethod
    def from_list(src_list, tgt_list=None, share_fields_from=None, **kwargs):
        if tgt_list is None:
            corpus = zip(src_list)
        else:
            corpus = zip(src_list, tgt_list)

        if share_fields_from is not None:
            src_field = share_fields_from.fields[src_field_name]
            if tgt_list is None:
                tgt_field = None
            else:
                tgt_field = share_fields_from.fields[tgt_field_name]
        else:
            # tokenize by character
            src_field = Field(batch_first=True, include_lengths=True, tokenize=list,
                              init_token=SOS, eos_token=EOS, unk_token=None)
            if tgt_list is None:
                tgt_field = None
            else:
                tgt_field = Field(batch_first=True, tokenize=list,
                                  init_token=SOS, eos_token=EOS, unk_token=None)

        return Seq2SeqDataset(corpus, src_field, tgt_field, **kwargs)

    def build_vocab(self, max_size):
        self.src_field.build_vocab(self, max_size=max_size)
        self.tgt_field.build_vocab(self, max_size=max_size)
