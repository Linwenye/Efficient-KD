# -*- coding: utf-8 -*-


# %%
import time
import numpy as np
from models.bert_crf import BERT_CRF_NER
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertModel, modeling
from utils.bio_data import *
import copy

cuda_yes = torch.cuda.is_available()
device = torch.device("cuda" if cuda_yes else "cpu")
print('Device:', device)

max_seq_length = 200  # 256
batch_size = 32  # 32
load_checkpoint = False

weight_decay_finetune = 1e-5  # 0.01
weight_decay_crf_fc = 5e-6  # 0.005
total_train_epochs = 15
gradient_accumulation_steps = 1
warmup_proportion = 0.1
bert_model_scale = 'bert-base-cased'
do_lower_case = False
bert_layers = 3  # number of layers; base:12
distil_alpha = 0.9
distil = 'struct_wo_trans'  # topk, none, struct_wo_trans, struct_with_trans, margin_prob
learning_rate0 = 5e-5 * (12 / bert_layers)
# lr0_crf_fc = 5e-3 * (12 / bert_layers)  # 8e-5
lr0_crf_fc = 5e-3 # 8e-5

output_dir = './dump/chunking/bert_crf'
ner_model_path = output_dir + '/ner_3layers_distil.pt'
teacher_model_path = output_dir + '/bert_crf.pt'
bert_path = 'resource/pretrained_lm/bert-base-cased/model.pt'
token_path = 'resource/pretrained_lm/bert-base-cased/bert-base-cased-vocab.txt'
data_dir = './datasets/conll-03'
# data_dir = './datasets/conll-00'

print(ner_model_path)
print('distil_alpha', distil_alpha, 'distil_type', distil)


def get_dataloader():
    label_map = conllProcessor.get_label_map()
    train_examples = conllProcessor.get_train_examples(data_dir)
    dev_examples = conllProcessor.get_dev_examples(data_dir)
    test_examples = conllProcessor.get_test_examples(data_dir)

    tokenizer = BertTokenizer.from_pretrained(token_path, do_lower_case=do_lower_case)
    train_dataset = NerDataset(train_examples, tokenizer, label_map, max_seq_length)
    dev_dataset = NerDataset(dev_examples, tokenizer, label_map, max_seq_length)
    test_dataset = NerDataset(test_examples, tokenizer, label_map, max_seq_length)

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=8,
                                       collate_fn=NerDataset.pad,pin_memory=True)

    dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=8,
                                     collate_fn=NerDataset.pad,pin_memory=True)

    test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=8,
                                      collate_fn=NerDataset.pad,pin_memory=True)
    return train_dataloader, dev_dataloader, test_dataloader, len(train_examples)


def load_model():
    label_list = conllProcessor.get_labels()
    start_label_id = conllProcessor.get_start_label_id()
    stop_label_id = conllProcessor.get_stop_label_id()

    config = modeling.BertConfig(vocab_size_or_config_json_file=28996, hidden_size=768,
                                 num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    teacher_model = BERT_CRF_NER(modeling.BertModel(config), start_label_id, stop_label_id, len(label_list),
                                 max_seq_length,
                                 batch_size, device)

    teacher_checkpoint = torch.load(teacher_model_path)
    pretrained_dict = teacher_checkpoint['model_state']
    net_state_dict = teacher_model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    print('Loaded the pretrain teacher NER_BERT_CRF model, epoch:', teacher_checkpoint['epoch'], 'from',
          teacher_model_path, 'valid acc:', teacher_checkpoint['valid_acc'], 'valid f1:',
          teacher_checkpoint['valid_f1'])
    teacher_model.load_state_dict(net_state_dict)
    teacher_model.eval()
    teacher_model.to(device)

    bert_model = BertModel.from_pretrained(bert_path)
    # bert_model = copy.deepcopy(teacher_model.bert)
    bert_model.encoder.layer = bert_model.encoder.layer[:bert_layers]
    model = BERT_CRF_NER(bert_model, start_label_id, stop_label_id, len(label_list), max_seq_length, batch_size, device)
    model.to(device)

    global valid_f1_best

    if load_checkpoint and os.path.exists(ner_model_path):
        checkpoint = torch.load(ner_model_path)
        start_epoch = checkpoint['epoch'] + 1
        pretrained_dict = checkpoint['model_state']
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        print('Loaded the pretrain NER_BERT_CRF model, epoch:', checkpoint['epoch'], 'valid acc:',
              checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
        valid_f1_best = checkpoint['valid_f1']
        evaluate(model, test_dataloader, batch_size, start_epoch - 1, 'Test_set')
    else:
        start_epoch = 0
        valid_f1_best = 0

    # if distil == 'struct_wo_trans':
    #     model.transitions = teacher_model.transitions
    #     model.transitions.requires_grad = False

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

    return model, teacher_model, optimizer, start_epoch, valid_f1_best


def train():
    global valid_f1_best, global_step_th

    tr_loss1, tr_loss2 = 0, 0
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    t_time, s_forward, s_backward = 0, 0, 0
    model.train()
    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        teacher_scores, teacher_feats, teacher_margin, topk_score, teacher_topk = None, None, None, None, None
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            if distil == 'struct_with_trans':
                teacher_scores = teacher_model.get_score_as_teacher(input_ids, segment_ids, input_mask)
            elif distil == 'struct_wo_trans':
                teacher_feats = teacher_model._get_bert_features(input_ids, segment_ids, input_mask)
            elif distil == 'topk':
                _feats = teacher_model._get_bert_features(input_ids, segment_ids, input_mask)
                topk_score, teacher_topk = teacher_model._viterbi_decode_nbest(_feats, input_mask, nbest=3)
            elif distil == 'topk_act':
                pass
            elif distil == 'margin_prob':
                _feats = teacher_model._get_bert_features(input_ids, segment_ids, input_mask)
                teacher_margin = teacher_model.marginal_probabilities(_feats, input_mask)
        torch.cuda.synchronize()
        t_stop = time.perf_counter()
        if distil=='none' or distil=='topk':
            loss = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids,
                                                teacher_feats=teacher_feats, teacher_scores=teacher_scores,
                                                teacher_margin=teacher_margin, topk_score=topk_score, teacher_topk=teacher_topk)
            tr_loss1 += loss.item()

        else:
            loss1, loss2 = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids,
                                                teacher_feats=teacher_feats, teacher_scores=teacher_scores,
                                                teacher_margin=teacher_margin, topk_score=topk_score, teacher_topk=teacher_topk)
            loss = (1 - distil_alpha) * loss1 + distil_alpha * loss2
            tr_loss1 += loss1.item()
            tr_loss2 += loss2.item()
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        torch.cuda.synchronize()
        s_forward_stop = time.perf_counter()
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = learning_rate0 * warmup_linear(global_step_th / total_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
        torch.cuda.synchronize()
        s_backward_stop = time.perf_counter()
        t_time += t_stop - t_start
        s_forward += s_forward_stop - t_stop
        s_backward += s_backward_stop - s_forward_stop
    print('-------------------------------------------------------------------------------------------------')
    print("Epoch:{} completed, Total training's Loss1: {}, Loss2:{}. Spend: {}s".format(epoch, tr_loss1, tr_loss2, (
            time.perf_counter() - train_start)))
    print('t_time:{}s, s_forward:{}s, s_backward:{}s'.format(t_time, s_forward, s_backward))
    valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')

    # Save a checkpoint
    if valid_f1 > valid_f1_best:
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
        #             'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
        #            ner_model_path)
        print('########best########')
        valid_f1_best = valid_f1


def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    start = time.perf_counter()
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
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct / total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.perf_counter()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend:%.3f s for evaluation' \
          % (epoch_th, 100. * test_acc, 100. * precision, 100. * recall, 100. * f1, dataset_name, end - start))
    return test_acc, f1


# def adjust_alpha():
#     global distil_alpha
#     distil_alpha -= 1.0 / (total_train_epochs-1)


if __name__ == '__main__':

    np.random.seed(44)
    torch.manual_seed(44)
    if cuda_yes:
        torch.cuda.manual_seed_all(44)

    # conllProcessor = CoNLLDataProcessor()
    conllProcessor = CoNLLChunkingProcessor()
    train_dataloader, dev_dataloader, test_dataloader, train_examples_len = get_dataloader()

    total_train_steps = int(train_examples_len / batch_size / gradient_accumulation_steps * total_train_epochs)
    model, teacher_model, optimizer, start_epoch, valid_f1_best = load_model()
    global_step_th = int(train_examples_len / batch_size / gradient_accumulation_steps * start_epoch)
    print('model loaded, start training ----------')
    # evaluate(teacher_model, test_dataloader, batch_size, 0, 'Test_set')
    # exit(0)
    for epoch in range(start_epoch, total_train_epochs):
        train()
        evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')
        # adjust_alpha()
        # exit(0)
