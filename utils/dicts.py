import sys
import os
from collections import Counter

from pytorch_pretrained_bert import BertTokenizer


class WordAlphabet(object):

    PAD_SIGN = "[PAD]"
    UNK_SIGN = "[UNK]"
    SOS_SIGN = "[SOS]"
    SPACE_SIGN = "[SPACE]"

    def __init__(self):
        self._idx_to_item = []
        self._item_to_idx = {}

        self._item_to_freq = Counter()

        self.add(self.PAD_SIGN, 1000000)
        self.add(self.UNK_SIGN, 1000000)
        self.add(self.SOS_SIGN, 1000000)
        self.add(self.SPACE_SIGN)

    def add(self, item, occurs=1):
        self._item_to_freq[item] += occurs

        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        try:
            return self._item_to_idx[item]
        except KeyError:
            return self._item_to_idx[self.UNK_SIGN]

    def freq(self, item):
        return self._item_to_freq[item]

    def __str__(self):
        return str(self._item_to_idx)

    def __len__(self):
        return len(self._idx_to_item)


class LabelAlphabet(object):

    def __init__(self):
        self._idx_to_item = []
        self._item_to_idx = {}

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        return self._item_to_idx[item]

    def __str__(self):
        return str(self._item_to_idx)

    def __len__(self):
        return len(self._idx_to_item)


class PieceAlphabet(object):

    CLS_SIGN, SEP_SIGN = "[CLS]", "[SEP]"
    PAD_SIGN, SOS_SIGN = "[PAD]", "[SOS]"
    SPACE_SIGN = "[SPACE]"

    def __init__(self, resource_dir, is_chinese):
        if is_chinese:
            dict_path = os.path.join(resource_dir, "pretrained_lm", "bert-base-chinese", "vocab.txt")
            no_splits = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[SOS]", "[SPACE]"]
        else:
            dict_path = os.path.join(resource_dir, "pretrained_lm", "bert-base-cased", "vocab.txt")
            no_splits = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[SOS]"]

        print("Loading the BERT vocabulary from path {}".format(dict_path), file=sys.stderr)
        self._segment_model = BertTokenizer(vocab_file=dict_path, never_split=no_splits, do_lower_case=False)

    def tokenize(self, item):
        return self._segment_model.tokenize(item)

    def index_seq(self, items):
        return self._segment_model.convert_tokens_to_ids(items)

    def get_seq(self, indexes):
        return self._segment_model.convert_ids_to_tokens(indexes)

    def __len__(self):
        return len(self._segment_model.vocab.items())
