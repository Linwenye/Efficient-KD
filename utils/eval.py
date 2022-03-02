import os
import random
import time


def flat_list(h_list):
    e_list = []

    for item in h_list:
        if isinstance(item, (list, tuple, set)):
            e_list.extend(flat_list(item))
        else:
            e_list.append(item)
    return e_list


def CWS_f1(pred_list, gold_list):
    """Refer to https://arxiv.org/pdf/1911.00720.pdf."""

    y_pred = flat_list(pred_list)
    y = flat_list(gold_list)

    cor_num = 0
    yp_wordnum = y_pred.count('E') + y_pred.count('S')
    yt_wordnum = y.count('E') + y.count('S')
    start = 0
    for i in range(len(y)):
        if y[i] == 'E' or y[i] == 'S':
            flag = True
            for j in range(start, i + 1):
                if y[j] != y_pred[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i + 1

    P = cor_num / float(yp_wordnum)
    R = cor_num / float(yt_wordnum)
    return 2.0 * P * R / (P + R) if P + R > 0.0 else 0.0


def miulab_f1(pred_list, gold_list):
    """Refer to https://github.com/MiuLab/SlotGated-SLU."""

    def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
        if prevTag == 'B' and tag == 'B':
            chunkStart = True
        if prevTag == 'I' and tag == 'B':
            chunkStart = True
        if prevTag == 'O' and tag == 'B':
            chunkStart = True
        if prevTag == 'O' and tag == 'I':
            chunkStart = True

        if prevTag == 'E' and tag == 'E':
            chunkStart = True
        if prevTag == 'E' and tag == 'I':
            chunkStart = True
        if prevTag == 'O' and tag == 'E':
            chunkStart = True
        if prevTag == 'O' and tag == 'I':
            chunkStart = True

        if tag != 'O' and tag != '.' and prevTagType != tagType:
            chunkStart = True
        return chunkStart

    def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
        if prevTag == 'B' and tag == 'B':
            chunkEnd = True
        if prevTag == 'B' and tag == 'O':
            chunkEnd = True
        if prevTag == 'I' and tag == 'B':
            chunkEnd = True
        if prevTag == 'I' and tag == 'O':
            chunkEnd = True

        if prevTag == 'E' and tag == 'E':
            chunkEnd = True
        if prevTag == 'E' and tag == 'I':
            chunkEnd = True
        if prevTag == 'E' and tag == 'O':
            chunkEnd = True
        if prevTag == 'I' and tag == 'O':
            chunkEnd = True

        if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
            chunkEnd = True
        return chunkEnd

    def __splitTagType(tag):
        s = tag.split('-')
        if len(s) > 2 or len(s) == 0:
            raise ValueError('tag format wrong. it must be B-xxx.xxx')
        if len(s) == 1:
            tag = s[0]
            tagType = ""
        else:
            tag = s[0]
            tagType = s[1]
        return tag, tagType

    correctChunk = {}
    correctChunkCnt = 0.0
    foundCorrect = {}
    foundCorrectCnt = 0.0
    foundPred = {}
    foundPredCnt = 0.0
    correctTags = 0.0
    tokenCount = 0.0
    for correct_slot, pred_slot in zip(gold_list, pred_list):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                        (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1.0
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1.0
                    else:
                        correctChunk[lastCorrectType] = 1.0
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                        (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                    __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                    (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1.0
                else:
                    foundCorrect[correctType] = 1.0

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1.0
                if predType in foundPred:
                    foundPred[predType] += 1.0
                else:
                    foundPred[predType] = 1.0

            if correctTag == predTag and correctType == predType:
                correctTags += 1.0
            tokenCount += 1.0

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1.0
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1.0
            else:
                correctChunk[lastCorrectType] = 1.0

    if foundPredCnt > 0:
        precision = 1.0 * correctChunkCnt / foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 1.0 * correctChunkCnt / foundCorrectCnt
    else:
        recall = 0

    if precision + recall > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1  # return (f1 Y), (precision N), (recall N)


def CoNLL_f1(sent_list, pred_list, gold_list, script_path):
    fn_out = 'eval_%04d.txt' % random.randint(0, 10000)
    if os.path.isfile(fn_out):
        os.remove(fn_out)

    text_file = open(fn_out, mode='w', encoding="utf-8")
    for i, words in enumerate(sent_list):
        tags_1 = gold_list[i]
        tags_2 = pred_list[i]
        for j, word in enumerate(words):
            tag_1 = tags_1[j]
            tag_2 = tags_2[j]
            text_file.write('%s %s %s\n' % (word, tag_1, tag_2))
        text_file.write('\n')
    text_file.close()

    cmd = 'perl %s < %s' % (os.path.join('.', script_path), fn_out)
    msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
    msg += ''.join(os.popen(cmd).readlines())
    time.sleep(1.0)
    if fn_out.startswith('eval_') and os.path.exists(fn_out):
        os.remove(fn_out)

    final = float(msg.split('\n')[3].split(':')[-1].strip()) * 0.01
    return final, msg
