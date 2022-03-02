import json
import os
import time
from tqdm import tqdm
import argparse

import torch
from torch.optim import Adam
from pytorch_pretrained_bert import BertAdam

from utils import fix_random_seed
from utils import get_corpus_iterator
from utils import bert_data_to_tensor
from utils import lstm_data_to_tensor
from utils import Procedure
from utils import warmup_linear
from utils import WordAlphabet
from utils import PieceAlphabet
from utils import LabelAlphabet

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default='bert', help='bert|lstm|distilbert|halfbert|3layer|1layer|6random')
parser.add_argument("--chinese", "-c", action="store_true", default=False)

parser.add_argument("--data_dir", "-dd", type=str, required=True)
parser.add_argument("--save_dir", "-sd", type=str, required=True)
parser.add_argument("--resource_dir", "-rd", type=str, required=True)
parser.add_argument("--random_state", "-rs", type=int, default=0)
parser.add_argument("--epoch_num", "-en", type=int, default=15)
parser.add_argument("--batch_size", "-bs", type=int, default=8)

parser.add_argument("--embedding_dim", "-ed", type=int, default=128)
parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
parser.add_argument("--dropout_rate", "-dr", type=float, default=0.3)

args = parser.parse_args()
if args.model == 'lstm':
    args.bert = False
else:
    args.bert = True
print(json.dumps(args.__dict__, ensure_ascii=False, indent=True), end="\n\n")

fix_random_seed(args.random_state)

token_vocab = PieceAlphabet(args.resource_dir, args.chinese) if args.bert else WordAlphabet()
label_vocab = LabelAlphabet()

train_loader = get_corpus_iterator(os.path.join(args.data_dir, "train.json"), args.batch_size, True,
                                   label_vocab, token_vocab if not args.bert else None)
dev_loader = get_corpus_iterator(os.path.join(args.data_dir, "dev.json"), args.batch_size, False)
test_loader = get_corpus_iterator(os.path.join(args.data_dir, "test.json"), args.batch_size, False)

model = torch.load(os.path.join(args.save_dir, "model.pt"))
model = model.cuda() if torch.cuda.is_available() else model.cpu()

if args.model == 'lstm':
    WEIGHT_DECAY = 1e-8
    optimizer = Adam(model.parameters(), weight_decay=WEIGHT_DECAY)
else:
    all_parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    LEARNING_RATE, WEIGHT_DECAY = (5e-5, 1e-5) if args.chinese else (1e-4, 0.01)
    WARMUP_PROPORTION = 0.1
    grouped_param = [
        {'params': [p for n, p in all_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    total_steps = int(len(train_loader) * (args.epoch_num + 1))
    optimizer = BertAdam(grouped_param, lr=LEARNING_RATE, warmup=WARMUP_PROPORTION, t_total=total_steps)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

dev_score, _, dev_time = Procedure.evaluation(model, dev_loader, token_vocab, label_vocab, args.resource_dir,
                                              args.bert)
print("(Epoch {:5d}) dev score: {:.6f}, dev time: {:.3f}".format(0, dev_score, dev_time))
test_score, msg, test_time = Procedure.evaluation(model, test_loader, token_vocab, label_vocab, args.resource_dir,
                                                  args.bert)
print("[Epoch {:5d}] test score: {:.6f}, test time: {:.3f}".format(0, test_score, test_time))
