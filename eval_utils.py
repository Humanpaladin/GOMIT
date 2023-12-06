from tkinter import _flatten

import pandas as pd
from sklearn.metrics import classification_report, jaccard_score, hamming_loss
import numpy as np


def jaccard(y_gold, y_pred):
    y_gold = np.asarray(y_gold).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    assert len(y_gold) == len(y_pred)
    tmp_sum = 0
    num_sample = len(y_gold)
    for i in range(num_sample):
        if y_pred[i][-1] == 1 and y_pred[i].sum() > 1 and y_gold[i][-1] != 1:
            # if y_gold[i][-1] != 1:
            y_gold_i = y_gold[i][:-1]
            y_pred_i = np.zeros((len(y_gold_i),), dtype=np.int)
        else:
            y_gold_i = y_gold[i][:-1]
            y_pred_i = y_pred[i][:-1]
        if sum(np.logical_or(y_gold_i, y_pred_i)) == 0:
            tmp_sum += 1
        else:
            tmp_sum += sum(y_gold_i & y_pred_i) / sum(np.logical_or(y_gold_i, y_pred_i))
    return tmp_sum / num_sample


def compute_classification_eval_metrics(metrics, predictions, labels, dataset):
    report = classification_report(labels, predictions, digits=4, zero_division=0, output_dict=True)
    metrics['classification'] = report

    if dataset == 'GoEmotions':
        metrics['jaccard_score'] = jaccard_score(labels, predictions, average='samples')

    return metrics


def extract_SemEvalEc(seq_list, emotion_dict):
    extractions, neutral_extractions = [], []
    for seqs in seq_list:
        for seq in seqs:
            num1 = np.zeros((11,), dtype=int).tolist()
            num2 = np.zeros((len(emotion_dict),), dtype=int).tolist()
            # if "neutral" in seq:
            #     extractions.append(num)
            #     break
            targets = seq.split(' [SSEP] ')
            # targets = seq.split()
            for target in targets:
                words = target.split()
                try:
                    emo = words[2].strip(".")
                    # emo = target
                except IndexError:
                    print(target)
                    emo = ''

                if emo != '' and emo in emotion_dict.keys():
                    if emo == "neutral":
                        idx = emotion_dict[emo]
                        num2[idx] = 1
                    else:
                        idx = emotion_dict[emo]
                        num1[idx] = 1
                        num2[idx] = 1
                else:
                    # print(targets)
                    num1 = np.zeros((11,), dtype=int).tolist()
                    num2 = np.zeros((len(emotion_dict),), dtype=int).tolist()
                    break
            if num2[11] == 1:
                num1 = np.zeros((11,), dtype=int).tolist()
            extractions.append(num1)
            neutral_extractions.append(num2)

    return extractions, neutral_extractions


def evaluate_SemEvalEc(outputs, targets, dataset):
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness",
                "surprise", "trust", "neutral"]
    emotion_dict = {emo: idx for idx, emo in enumerate(emotions)}
    pred_pt, jaccord_pred = extract_SemEvalEc(outputs, emotion_dict)
    gold_pt, jaccord_gold = extract_SemEvalEc(targets, emotion_dict)

    pred_pt = np.array(pred_pt)
    gold_pt = np.array(gold_pt)
    metrics = {}
    metrics = compute_classification_eval_metrics(metrics, pred_pt, gold_pt, dataset=dataset)
    metrics['jaccard_score'] = jaccard(jaccord_gold, jaccord_pred)
    # print(metrics)

    return metrics


def evaluate_GoEmotion(outputs, targets, dataset):
    emotion_label = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                     'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                     'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                     'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    emotion_dict = {emo: idx for idx, emo in enumerate(emotion_label)}

    pred = extract_GoEmotion(outputs, emotion_dict)
    # outputs:
    #   为所有样本的预测结果句子, 以 batch 分开. 共 5427/16=340 个 batch. 如:
    #       [["..", ..., ".."],
    #        ["..", ..., ".."],
    #        ...
    #        ["..", ..., ".."]]      一个 batch 包含 16 个句子.
    #   其中, 每个句子形如 "The emotion remorse is ... [SSEP] The emotion sadness is ..."
    # pred:
    #   为每个句子所对应的 label, 在相应的位置置 1. 如:
    #       [[0, 0, 1, ..., 0, 0, 0],
    #        [0, 0, 1, ..., 0, 1, 0],
    #        ...
    #        [0, 1, 0, ..., 0, 0, 1]]
    gold = extract_GoEmotion(targets, emotion_dict)     # targets: 所有样本的 gold 结果句子, 以 batch 分开
    # targets:
    #   为所有样本的 gold 结果句子, 形制同 outputs
    # gold:
    #   为所有样本的对应的 label, 在相应位置置 1. 形制同 pred

    metrics = {}
    metrics = compute_classification_eval_metrics(metrics, pred, gold, dataset=dataset)
    print(metrics)

    # metrics = compute_metrics_triplet(gold, pred)

    return metrics



def extract_GoEmotion(batch_list, emotion_dict):
    """

    Args:
        batch_list:
            所有样本的 pred / gold 结果句子, 以 batch 分开. 如:
                [["The emotion ...", "The emotion ...", ..., "The emotion ..."],    # 一个 batch (16个句子) 的 pred / gold 结果句子
                 ["The emotion ...", "The emotion ...", ..., "The emotion ..."],
                  ...,
                 ["The emotion ...", "The emotion ...", ..., "The emotion ..."]]
        emotion_dict:
            {emo: idx}

    Returns:
        extractions:
            为所有句子的 label, 为一个 list. 以下的方式表示:
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 ...
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            extractions 的每一个元素(也为一个 list)的长度为 28, 用 1 表示在该位置的 label 被取得, 即为对应样本句子的 pred/gold label

    """
    extractions = []
    for batch in batch_list:
        for sequence in batch:
            # num = np.zeros((len(emotion_dict),), dtype=int).tolist()
            num = len(emotion_dict) * [0]
            sentences = sequence.split(' [SSEP] ')      # 如果 pred / gold 句子中包含多个句子
            for sen in sentences:
                words = sen.split()
                try:
                    emo = words[2].strip('.')
                    # emo = sen
                except IndexError:
                    emo = ''

                if emo != '' and emo in emotion_dict.keys():
                    idx = emotion_dict[emo]
                    num[idx] = 1
                else:
                    # num = np.zeros((len(emotion_dict),), dtype=int).tolist()
                    num = len(emotion_dict) * [0]
                    break
            extractions.append(num)

    return extractions


def evaluate(outputs, targets, dataset):
    if dataset == 'SemEvalEc':
        results = evaluate_SemEvalEc(outputs, targets, dataset)
        return results
    elif dataset == 'GoEmotions':
        results = evaluate_GoEmotion(outputs, targets, dataset)
        return results
    elif dataset == "Triplets" or dataset == "Triplets_Restaurant":
        golds, preds = extract_Triplets(targets, outputs)
        # results = compute_metrics_triplet(golds, preds)
        # results = compute_metrics_target(golds, preds)
        # results = compute_metrics_opinion(golds, preds)
        results = compute_metrics_sentiment(golds, preds)
        return results


def extract_Triplets(targets, outputs):
    golds = list(_flatten(targets))
    preds = list(_flatten(outputs))

    gold_triplets = []
    for gold in golds:
        # print(f"gold: {gold}")
        triplets = []
        items = gold.split(" [SSEP] ")
        for item in items:
            words = item.split(" ")

            s = words[1]

            o_s = 3
            o_e = words.index("is")
            o = " ".join(words[o_s: o_e])

            t_s = words.index("towards") + 1
            t_e = words.index("in")
            t = " ".join(words[t_s: t_e])

            tri = (t, o, s)
            triplets.append(tri)
        gold_triplets.append(triplets)

    pred_triplets = []
    for pred in preds:
        triplets = []
        items = pred.split(" [SSEP] ")
        for item in items:
            words = item.split(" ")

            try:
                s = words[1]
            except IndexError:
                # print(item)
                s = ''

            o_s = 3
            if "is" in words:
                o_e = words.index("is")
            else:
                o_e = 4
            o = " ".join(words[o_s: o_e])

            if "towards" in words and "in" in words:
                t_s = words.index("towards") + 1
                t_e = words.index("in")
                t = " ".join(words[t_s: t_e])
            else:
                t = ""

            tri = (t, o, s)
            triplets.append(tri)
        pred_triplets.append(triplets)

    return gold_triplets, pred_triplets


def compute_metrics_triplet(golds, preds):
    """
    计算三元组的预测指标
    args:
        golds:
            每个句子所包含的三元组, 如:
                [[(3, 3, 0, 0, 3), (3, 3, 2, 2, 3), (5, 6, 8, 8, 3)], ...]
                                                    |-------------|
                                                          fn
        preds:
            预测每个句子所包含的三元组, 如:
                [[(3, 3, 0, 0, 3), (3, 3, 2, 2, 3), (5, 6, 7, 7, 3)], ...]
                  |-------------|  |-------------|  |-------------|
                        hit              hit              fp
    returns:
        [recall, precision, f1]

    """
    TP, FP, FN = 0, 0, 0            # 最后统计得到的所有样本包含的三元组的累积 tp, fp 和 fn
    num_examples = len(golds)       # 样本的个数
    assert num_examples == len(preds)

    for i in range(num_examples):   # 对于第 i 个样本
        num_gold = len(golds[i])    # 预测第 i 个样本包含几个三元组
        num_pred = len(preds[i])    # 实际第 i 个样本包含几个三元组
        num_hit = 0                 # 第 i 个样本预测命中的个数

        checked_pred = []
        for tri in preds[i]:
            if tri not in checked_pred:
                checked_pred.append(tri)
                if tri in golds[i]:
                    num_hit += 1        # 第 i 个样本预测命中的个数统计
        fp = num_pred - num_hit     # 第 i 个样本的 false positive
        fn = num_gold - num_hit     # 第 i 个样本的 false negative

        TP += num_hit
        FP += fp
        FN += fn

    precision = float(TP) / float(TP + FP + 1e-6)
    recall = float(TP) / float(TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return [precision, recall, f1]


def compute_metrics_target(golds, preds):
    """
    计算 target 的预测指标
    args:
        golds:
            每个句子所包含的三元组, 如:
                [[(3, 3, 0, 0, 3), (3, 3, 2, 2, 3), (5, 6, 8, 8, 3)], ...]
                                                    |-------------|
                                                          fn
        preds:
            预测每个句子所包含的三元组, 如:
                [[(3, 3, 0, 0, 3), (3, 3, 2, 2, 3), (5, 6, 7, 7, 3)], ...]
                  |-------------|  |-------------|  |-------------|
                        hit              hit              fp
    returns:
        [recall, precision, f1]

    """
    TP, FP, FN = 0, 0, 0            # 最后统计得到的所有样本包含的三元组的累积 tp, fp 和 fn
    num_examples = len(golds)       # 样本的个数
    assert num_examples == len(preds)

    gold_triplets = golds
    pred_triplets = preds

    # 从已有的 gold, pred (triplet) 中提取出 gold_target, pred_target, 用以计算 F1
    gold_targets = []  # 以句子为单位, 每个句子所包含的 targets: [[t1, t2], [t1, t2, t3]]
    for sent_golds in gold_triplets:
        sent_targets = []
        for triplet in sent_golds:
            target = triplet[0]
            sent_targets.append(target)
        gold_targets.append(sent_targets)

    pred_targets = []  # 以句子为单位, 每个句子所预测的 targets
    for sent_preds in pred_triplets:
        sent_targets = []
        for triplet in sent_preds:
            target = triplet[0]
            sent_targets.append(target)
        pred_targets.append(sent_targets)

    for i in range(num_examples):   # 对于第 i 个样本
        num_gold = len(gold_targets[i])    # 预测第 i 个样本包含几个三元组
        num_pred = len(pred_targets[i])    # 实际第 i 个样本包含几个三元组
        num_hit = 0                 # 第 i 个样本预测命中的个数

        checked_pred = []
        for tar in pred_targets[i]:
            if tar not in checked_pred:
                checked_pred.append(tar)
                if tar in gold_targets[i]:
                    num_hit += 1        # 第 i 个样本预测命中的个数统计
        fp = num_pred - num_hit     # 第 i 个样本的 false positive
        fn = num_gold - num_hit     # 第 i 个样本的 false negative

        TP += num_hit
        FP += fp
        FN += fn

    precision = float(TP) / float(TP + FP + 1e-6)
    recall = float(TP) / float(TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return [precision, recall, f1]


def compute_metrics_opinion(golds, preds):
    """
    计算 target 的预测指标
    args:
        golds:
            每个句子所包含的三元组, 如:
                [[(3, 3, 0, 0, 3), (3, 3, 2, 2, 3), (5, 6, 8, 8, 3)], ...]
                                                    |-------------|
                                                          fn
        preds:
            预测每个句子所包含的三元组, 如:
                [[(3, 3, 0, 0, 3), (3, 3, 2, 2, 3), (5, 6, 7, 7, 3)], ...]
                  |-------------|  |-------------|  |-------------|
                        hit              hit              fp
    returns:
        [recall, precision, f1]

    """
    TP, FP, FN = 0, 0, 0            # 最后统计得到的所有样本包含的三元组的累积 tp, fp 和 fn
    num_examples = len(golds)       # 样本的个数
    assert num_examples == len(preds)

    gold_triplets = golds
    pred_triplets = preds

    # 从已有的 gold, pred (triplet) 中提取出 gold_target, pred_target, 用以计算 F1
    gold_targets = []  # 以句子为单位, 每个句子所包含的 targets: [[t1, t2], [t1, t2, t3]]
    for sent_golds in gold_triplets:
        sent_targets = []
        for triplet in sent_golds:
            target = triplet[1]
            sent_targets.append(target)
        gold_targets.append(sent_targets)

    pred_targets = []  # 以句子为单位, 每个句子所预测的 targets
    for sent_preds in pred_triplets:
        sent_targets = []
        for triplet in sent_preds:
            target = triplet[1]
            sent_targets.append(target)
        pred_targets.append(sent_targets)

    for i in range(num_examples):   # 对于第 i 个样本
        num_gold = len(gold_targets[i])    # 预测第 i 个样本包含几个三元组
        num_pred = len(pred_targets[i])    # 实际第 i 个样本包含几个三元组
        num_hit = 0                 # 第 i 个样本预测命中的个数

        checked_pred = []
        for tar in pred_targets[i]:
            if tar not in checked_pred:
                checked_pred.append(tar)
                if tar in gold_targets[i]:
                    num_hit += 1        # 第 i 个样本预测命中的个数统计
        fp = num_pred - num_hit     # 第 i 个样本的 false positive
        fn = num_gold - num_hit     # 第 i 个样本的 false negative

        TP += num_hit
        FP += fp
        FN += fn

    precision = float(TP) / float(TP + FP + 1e-6)
    recall = float(TP) / float(TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return [precision, recall, f1]


def compute_metrics_sentiment(golds, preds):
    """
    计算 target 的预测指标
    args:
        golds:
            每个句子所包含的三元组, 如:
                [[(3, 3, 0, 0, 3), (3, 3, 2, 2, 3), (5, 6, 8, 8, 3)], ...]
                                                    |-------------|
                                                          fn
        preds:
            预测每个句子所包含的三元组, 如:
                [[(3, 3, 0, 0, 3), (3, 3, 2, 2, 3), (5, 6, 7, 7, 3)], ...]
                  |-------------|  |-------------|  |-------------|
                        hit              hit              fp
    returns:
        [recall, precision, f1]

    """
    TP, FP, FN = 0, 0, 0            # 最后统计得到的所有样本包含的三元组的累积 tp, fp 和 fn
    num_examples = len(golds)       # 样本的个数
    assert num_examples == len(preds)

    gold_triplets = golds
    pred_triplets = preds

    # 从已有的 gold, pred (triplet) 中提取出 gold_target, pred_target, 用以计算 F1
    gold_targets = []  # 以句子为单位, 每个句子所包含的 targets: [[t1, t2], [t1, t2, t3]]
    for sent_golds in gold_triplets:
        sent_targets = []
        for triplet in sent_golds:
            target = triplet[2]
            sent_targets.append(target)
        gold_targets.append(sent_targets)

    pred_targets = []  # 以句子为单位, 每个句子所预测的 targets
    for sent_preds in pred_triplets:
        sent_targets = []
        for triplet in sent_preds:
            target = triplet[2]
            sent_targets.append(target)
        pred_targets.append(sent_targets)

    for i in range(num_examples):   # 对于第 i 个样本
        num_gold = len(gold_targets[i])    # 预测第 i 个样本包含几个三元组
        num_pred = len(pred_targets[i])    # 实际第 i 个样本包含几个三元组
        num_hit = 0                 # 第 i 个样本预测命中的个数

        checked_pred = []
        for tar in pred_targets[i]:
            if tar not in checked_pred:
                checked_pred.append(tar)
                if tar in gold_targets[i]:
                    num_hit += 1        # 第 i 个样本预测命中的个数统计
        fp = num_pred - num_hit     # 第 i 个样本的 false positive
        fn = num_gold - num_hit     # 第 i 个样本的 false negative

        TP += num_hit
        FP += fp
        FN += fn

    precision = float(TP) / float(TP + FP + 1e-6)
    recall = float(TP) / float(TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return [precision, recall, f1]
