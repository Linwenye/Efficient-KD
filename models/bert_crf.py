import torch.nn.functional as F
from utils.kl import DistillKL
from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
import torch.autograd as autograd
from utils.kl import DistillKL
import torch
from torch import nn
from typing import Optional


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1,
                                                                                                                m_size)  # B * M


def log_sum_exp_dim(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


class BERT_CRF_NER(nn.Module):

    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, max_seq_length, batch_size, device):
        super(BERT_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        # self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        self.gpu = True
        # use pretrainded BertModel
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.3)
        # Maps the output of the bert into label space.
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)

        # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        # self.transitions.data[:, start_label_id] = -10000
        # self.transitions.data[stop_label_id, :] = -10000

        self.distil = DistillKL(10)  # T=10
        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)
        # self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_score_as_teacher(self, input_ids, segment_ids, mask):
        feats = self._get_bert_features(input_ids, segment_ids, mask)

        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        return scores

    def _forward_alg(self, feats, mask):
        """
           input:
               feats: (batch, seq_len, tag_size)
               masks: (batch, seq_len)
       """
        batch_size, seq_len, tag_size = feats.shape
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        mask = mask.transpose(1, 0).contiguous()
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_label_id, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        ## add start score (from start to all tag, duplicate to batch_size)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            ## effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx == 1)
            ## let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            ## replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx == 1, masked_cur_partition)
        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + \
                     partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.stop_label_id]
        return final_partition.sum(), scores

    def marginal_probabilities(self,
                               emissions: torch.Tensor,
                               mask: Optional[torch.ByteTensor] = None) -> torch.FloatTensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            marginal_probabilities: (sequence_length, batch_size, num_tags)
        """
        if mask is None:
            batch_size, sequence_length, _ = emissions.data.shape
            mask = torch.ones([batch_size, sequence_length], dtype=torch.uint8, device=emissions.device)
        # print('margin')
        alpha = self.alpha_beta(emissions,
                                mask,
                                reverse_direction=False)
        beta = self.alpha_beta(emissions,
                               mask,
                               reverse_direction=True)
        z = log_sum_exp_dim(alpha[alpha.size(0) - 1] + self.transitions[:, self.stop_label_id], dim=1)  # p(y_t|x)
        proba = torch.exp(alpha + beta - z.view(1, -1, 1))
        return proba

    def alpha_beta(self,
                   emissions: torch.Tensor,
                   mask: torch.ByteTensor,
                   reverse_direction: bool = False) -> torch.FloatTensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
            reverse_direction: This parameter decide algorithm direction.
        Returns:
            log_probabilities: (sequence_length, batch_size, num_tags)
        """
        batch_size, sequence_length, num_tags = emissions.data.shape

        broadcast_emissions = emissions.transpose(0, 1).unsqueeze(
            2).contiguous()  # (sequence_length, batch_size, 1, num_tags)
        mask = mask.float().transpose(0, 1).contiguous()  # (sequence_length, batch_size)
        broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
        sequence_iter = range(1, sequence_length)

        # backward algorithm
        if reverse_direction:
            # Transpose transitions matrix and emissions
            broadcast_transitions = broadcast_transitions.transpose(1, 2)  # (1, num_tags, num_tags)
            broadcast_emissions = broadcast_emissions.transpose(2, 3)  # (sequence_length, batch_size, num_tags, 1)
            sequence_iter = reversed(sequence_iter)

            # It is beta
            log_proba = [self.transitions[:, self.stop_label_id].expand(batch_size, num_tags)]
        # forward algorithm
        else:
            # It is alpha
            log_proba = [emissions.transpose(0, 1)[0] + self.transitions[self.start_label_id].view(1, -1)]

        for i in sequence_iter:
            # Broadcast log probability
            broadcast_log_proba = log_proba[-1].unsqueeze(2)  # (batch_size, num_tags, 1)

            # Add all scores
            # inner: (batch_size, num_tags, num_tags)
            # broadcast_log_proba:   (batch_size, num_tags, 1)
            # broadcast_transitions: (1, num_tags, num_tags)
            # broadcast_emissions:   (batch_size, 1, num_tags)
            inner = broadcast_log_proba \
                    + broadcast_transitions \
                    + broadcast_emissions[i]

            # Append log proba
            log_proba.append((log_sum_exp_dim(inner, 1) * mask[i].view(batch_size, 1) +
                              log_proba[-1] * (1 - mask[i]).view(batch_size, 1)))

        if reverse_direction:
            log_proba.reverse()

        return torch.stack(log_proba)

    def _get_bert_features(self, input_ids, segment_ids, input_mask):
        '''
            (batch_size, sequence_length, num_tags)
        '''
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    output_all_encoded_layers=False)
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        return bert_feats

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        ## convert tag value into a new format, recorded label bigram information to index
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                ## start -> first score
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        ## transition for label to STOP_TAG
        end_transition = self.transitions[:, self.stop_label_id].contiguous().view(1, tag_size).expand(batch_size,
                                                                                                       tag_size)
        ## length for batch,  last word position = length - 1
        lens = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## index the label id of last word
        end_ids = torch.gather(tags, 1, lens - 1)
        ## index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)
        ## convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        ### need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len,
                                                                                         batch_size)  # seq_len * bat_size
        ## mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0) == 1)
        ## add all score together
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids, teacher_feats=None, teacher_scores=None, teacher_margin=None,
                           teacher_topk=None, topk_score=None):
        """
            loss function: predict-gold
        """
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        forward_score, scores = self._forward_alg(bert_feats, input_mask)
        if teacher_topk is not None:
            pseudo_label_loss = torch.zeros(forward_score.shape).cuda()
            loss_factor = 1
            for i, pseudo_labels in enumerate(teacher_topk):
                if i != 0:
                    loss_factor = 0.1
                pseudo_score = self._score_sentence(scores, input_mask, pseudo_labels)
                pseudo_label_loss += torch.mean(forward_score - pseudo_score) * loss_factor
            return pseudo_label_loss
        gold_score = self._score_sentence(scores, input_mask, label_ids)
        loss1 = torch.mean(forward_score - gold_score)
        if teacher_feats is not None:
            # loss2 = torch.mean((bert_feats - teacher_feats) ** 2)
            loss2 = torch.mean(nn.MSELoss(reduction='mean')(bert_feats, teacher_feats))
            # loss2 = torch.mean(torch.sum(nn.MSELoss(reduction='none')(bert_feats, teacher_feats), dim=(2, 1)))
            # loss2 = self.distil(bert_feats,teacher_feats)
            return loss1, loss2
        if teacher_scores is not None:
            loss2 = torch.mean((scores - teacher_scores) ** 2)
            # loss2 = self.distil(scores.view(scores.size(0), scores.size(1), -1),
            #                     teacher_scores.view(scores.size(0), scores.size(1), -1))
            return loss1, loss2
        if teacher_margin is not None:
            # print('margin')
            student_margin = self.marginal_probabilities(bert_feats, input_mask)
            loss2 = nn.KLDivLoss(reduction='batchmean')(torch.log(student_margin), teacher_margin)
            return loss1, loss2

        return loss1

    def _viterbi_decode(self, feats, mask):
        """
           input:
               feats: (batch, seq_len, self.tag_size+2)
               mask: (batch, seq_len)
           output:
               decode_idx: (batch, seq_len) decoded sequence
               path_score: (batch, 1) corresponding score for each sequence (to be implementated)
       """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        ## calculate sentence length for each sentence
        lens = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).bool()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_label_id, :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        # print "init part:",partition.size()
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            ## forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            ## cur_bp: (batch_size, tag_size) max source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        ### add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1, 0).contiguous()  # (batch_size,seq_len,tagsize)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = lens.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size, tag_size).expand(batch_size,
                                                                                                                                  tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        ## select end ids in STOP_TAG
        pointer = last_bp[:, self.stop_label_id]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def _viterbi_decode_nbest(self, feats, mask, nbest):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, nbest, seq_len) decoded sequence
                path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
                nbest decode for sentence with one token is not well supported, to be optimized

                remove the difference of start and end difference.
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        ## calculate sentence length for each sentence
        lens = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).bool()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_label_id, :].clone()  # bat_size * to_target_size
        ## initial partition [batch_size, tag_size]
        partition_history.append(partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest))
        # iter over last scores
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(
                    batch_size, tag_size, tag_size)
            else:
                # previous to_target is current from_target
                # partition: previous results log(exp(from_target)), #(batch_size * nbest * from_target)
                # cur_values: batch_size * from_target * to_target
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest,
                                                                                       tag_size) + \
                             partition.contiguous().view(batch_size, tag_size, nbest, 1).expand(batch_size, tag_size,
                                                                                                nbest, tag_size)
                ## compare all nbest and all from target
                cur_values = cur_values.view(batch_size, tag_size * nbest, tag_size)
                # print "cur size:",cur_values.size()
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            ## cur_bp/partition: [batch_size, nbest, tag_size], id should be normize through nbest in following backtrace step
            # print partition[:,0,:]
            # print cur_bp[:,0,:]
            # print "nbest, ",idx
            if idx == 1:
                cur_bp = cur_bp * nbest
            partition = partition.transpose(2, 1)
            cur_bp = cur_bp.transpose(2, 1)

            # partition: (batch_size * to_target * nbest)
            # cur_bp: (batch_size * to_target * nbest) Notice the cur_bp number is the whole position of tag_size*nbest, need to convert when decode
            partition_history.append(partition)
            ## cur_bp: (batch_size,nbest, tag_size) topn source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            ## mask[idx] ? mask[idx-1]
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0)
            # print cur_bp[0]
            back_points.append(cur_bp)
        ### add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, tag_size, nbest).transpose(1,
                                                                                                                 0).contiguous()  ## (batch_size, seq_len, nbest, tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = lens.view(batch_size, 1, 1, 1).expand(batch_size, 1, tag_size, nbest) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, nbest, 1)
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, nbest, tag_size) + self.transitions.view(1, tag_size,
                                                                                                           1,
                                                                                                           tag_size).expand(
            batch_size, tag_size, nbest, tag_size)
        last_values = last_values.view(batch_size, tag_size * nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        ## end_partition: (batch, nbest, tag_size)
        end_bp = end_bp.transpose(2, 1)
        # end_bp: (batch, tag_size, nbest)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size, nbest)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

        ## select end ids in STOP_TAG
        pointer = end_bp[:, self.stop_label_id, :]  ## (batch_size, nbest)
        insert_last = pointer.contiguous().view(batch_size, 1, 1, nbest).expand(batch_size, 1, tag_size, nbest)
        back_points = back_points.transpose(1, 0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        ## copy the ids of last position:insert_last to back_points, though the last_position index
        ## last_position includes the length of batch sentences
        back_points.scatter_(1, last_position, insert_last)
        ## back_points: [batch_size, seq_length, tag_size, nbest]
        '''
        back_points: in simple demonstratration
        x,x,x,x,x,x,x,x,x,7
        x,x,x,x,x,4,0,0,0,0
        x,x,6,0,0,0,0,0,0,0
        '''

        back_points = back_points.transpose(1, 0).contiguous()
        ## back_points: (seq_len, batch, tag_size, nbest)
        ## decode from the end, padded position ids are 0, which will be filtered in following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size, nbest))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data / nbest
        # use old mask, let 0 means has token
        for idx in range(len(back_points) - 2, -1, -1):
            new_pointer = torch.gather(back_points[idx].view(batch_size, tag_size * nbest), 1,
                                       pointer.contiguous().view(batch_size, nbest))
            decode_idx[idx] = new_pointer.data / nbest
            # # use new pointer to remember the last end nbest ids for non longest
            pointer = new_pointer + pointer.contiguous().view(batch_size, nbest) * mask[idx].view(batch_size, 1).expand(
                batch_size, nbest).long()

        path_score = None
        decode_idx[-1] = 0

        decode_idx = decode_idx.transpose(1, 0)
        decode_idx[torch.arange(batch_size), lens.squeeze() - 1] = self.stop_label_id
        ## decode_idx: [batch, nbest, seq_len]
        decode_idx = decode_idx.transpose(1, 2).transpose(0, 1)

        ### calculate probability for each sequence
        scores = end_partition[:, :, self.stop_label_id]
        ## scores: [batch_size, nbest]
        max_scores, _ = torch.max(scores, 1)
        minus_scores = scores - max_scores.view(batch_size, 1).expand(batch_size, nbest)
        path_score = F.softmax(minus_scores, 1)
        ## path_score: [batch_size, nbest]
        return path_score, decode_idx

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask):
        # Get the emission scores from the BiLSTM
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)

        # Find the best path, given the features.
        score, label_seq_ids = self._viterbi_decode(bert_feats, input_mask)
        return score, label_seq_ids
