# -*- coding: utf-8 -*-



# %%
import sys
import os
import time
# import importlib
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

from torch.utils.data.distributed import DistributedSampler
from torch.utils import data

# from tqdm import tqdm, trange
import collections
from models.bert_crf import BERT_CRF_NER
from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
import pickle
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.bio_data import *

cuda_yes = torch.cuda.is_available()
device = torch.device("cuda" if cuda_yes else "cpu")
print('Device:', device)

data_dir = './datasets/conll-03'
# "Whether to run training."
do_train = True
# "Whether to run eval on the dev set."
do_eval = True
# "Whether to run the model in inference mode on the test set."
do_predict = True
# Whether load checkpoint file before train model
load_checkpoint = False
# "The vocabulary file that the BERT model was trained on."
max_seq_length = 200  # 256
batch_size = 32  # 32
# "The initial learning rate for Adam."
# learning_rate0 = 5e-5
# lr0_crf_fc = 5e-3  # 8e-5

bert_layers = 12
learning_rate0 = 5e-5 * (12 / bert_layers)
lr0_crf_fc = 5e-3 * (12 / bert_layers)  # 8e-5
weight_decay_finetune = 1e-5  # 0.01
weight_decay_crf_fc = 5e-6  # 0.005
total_train_epochs = 15
gradient_accumulation_steps = 1
warmup_proportion = 0.1
output_dir = './dump/bert_crf'
bert_model_scale = 'bert-base-cased'
do_lower_case = False
ner_model_path = output_dir + '/bert_crf.pt'

print(ner_model_path)

# %%
'''
Functions and Classes for read and organize data set
'''

# %%
'''
Prepare data set
'''
# random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
if cuda_yes:
    torch.cuda.manual_seed_all(44)

# Load pre-trained model tokenizer (vocabulary)
conllProcessor = CoNLLDataProcessor()
label_list = conllProcessor.get_labels()
label_map = conllProcessor.get_label_map()
train_examples = conllProcessor.get_train_examples(data_dir)
dev_examples = conllProcessor.get_dev_examples(data_dir)
test_examples = conllProcessor.get_test_examples(data_dir)

total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)

print("***** Running training *****")
print("  Num examples = %d" % len(train_examples))
print("  Batch size = %d" % batch_size)
print("  Num steps = %d" % total_train_steps)

tokenizer = BertTokenizer.from_pretrained('resource/pretrained_lm/bert-base-cased/bert-base-cased-vocab.txt',
                                          do_lower_case=do_lower_case)
# tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

train_dataset = NerDataset(train_examples, tokenizer, label_map, max_seq_length)
dev_dataset = NerDataset(dev_examples, tokenizer, label_map, max_seq_length)
test_dataset = NerDataset(test_examples, tokenizer, label_map, max_seq_length)

train_dataloader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   collate_fn=NerDataset.pad)

dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=NerDataset.pad)

test_dataloader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=NerDataset.pad)
'''
#####  Use BertModel + CRF  #####
CRF is for transition and the maximum likelyhood estimate(MLE).
Bert is for latent label -> Emission of word embedding.
'''
print('*** Use BertModel + CRF ***')


def log_sum_exp_1vec(vec):  # shape(1,m)
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_mat(log_M, axis=-1):  # shape(n,m)
    return torch.max(log_M, axis)[0] + torch.log(torch.exp(log_M - torch.max(log_M, axis)[0][:, None]).sum(axis))


def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    # print("***** Running prediction *****")
    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
            # _, predicted = torch.max(out_scores, -1)
            valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct / total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend:%.2f s for evaluation' \
          % (epoch_th, 100. * test_acc, 100. * precision, 100. * recall, 100. * f1, dataset_name, end - start))
    return test_acc, f1


start_label_id = conllProcessor.get_start_label_id()
stop_label_id = conllProcessor.get_stop_label_id()

bert_path = 'resource/pretrained_lm/bert-base-cased/model.pt'
bert_model = BertModel.from_pretrained(bert_path)
bert_model.encoder.layer = bert_model.encoder.layer[:bert_layers]
model = BERT_CRF_NER(bert_model, start_label_id, stop_label_id, len(label_list), max_seq_length, batch_size, device)
model.to(device)

# %%
if load_checkpoint and os.path.exists(ner_model_path):
    checkpoint = torch.load(ner_model_path)
    start_epoch = checkpoint['epoch'] + 1
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    pretrained_dict = checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    print('Loaded the pretrain NER_BERT_CRF model, epoch:', checkpoint['epoch'], 'valid acc:',
          checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
    evaluate(model, test_dataloader, 32, start_epoch, 'test set')
else:
    start_epoch = 0
    valid_acc_prev = 0
    valid_f1_prev = 0

# Prepare optimizer
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
                and not any(nd in n for nd in new_param)], 'weight_decay': weight_decay_finetune},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
                and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')] \
        , 'lr': lr0_crf_fc, 'weight_decay': weight_decay_crf_fc},
    {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
        , 'lr': lr0_crf_fc, 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion,
                     t_total=total_train_steps)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


# %%
# train procedure
global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)

# train_start=time.time()
# for epoch in trange(start_epoch, total_train_epochs, desc="Epoch"):
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

        neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)

        if gradient_accumulation_steps > 1:
            neg_log_likelihood = neg_log_likelihood / gradient_accumulation_steps

        neg_log_likelihood.backward()

        tr_loss += neg_log_likelihood.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = learning_rate0 * warmup_linear(global_step_th / total_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
        #
        # print("Epoch:{}-{}/{}, Negative loglikelihood: {} ".format(epoch, step, len(train_dataloader),
        #                                                            neg_log_likelihood.item()))

    print('--------------------------------------------------------------')
    print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss,
                                                                             (time.time() - train_start) / 60.0))
    valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')

    # Save a checkpoint
    if valid_f1 > valid_f1_prev:
        # torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
        #             'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
        #            ner_model_path)
        valid_f1_prev = valid_f1
        print('save--------------------------')
    evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')
