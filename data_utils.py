import json

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import itertools
import data_process
import main
import copy


def get_sem_data(path, data_type):
    sents, labels, label_idx = [], [], []
    df = pd.read_csv(path, sep="	")
    processed_sen_df = pd.read_csv(f'data/SemEvalEc/processed_{data_type}.csv')
    lab = df.iloc[:, 2:]
    all_label = [emotion for emotion in lab.columns]
    sents_array = processed_sen_df['0'].values.tolist()  # (6838,)
    label_array = df.iloc[:, 2:].values.tolist()  # （6838,11)

    for temp in label_array:

        if np.asarray(temp).sum() == 0:
            labels.append('The emotion neutral is expressed in this sentence.')
            label_idx.append(torch.tensor(temp, dtype=torch.float32))
            continue
        # temp.append(0)
        label_idx.append(torch.tensor(temp, dtype=torch.float32))

        label_list = []
        index = [inx for inx, num in enumerate(temp) if num == 1]
        i_label_strs = [all_label[a] for a in index]
        for emo in i_label_strs:
            label_list.append(f'The emotion {emo} is expressed in this sentence.')
        labels.append(' [SSEP] '.join(label_list))

    for sen in sents_array:
        sen = sen.split()
        if sen != '':
            sents.append(sen)

    return sents, labels, label_idx

def get_Triplet_data(path, data_dir, args):
    """
    本函数构建的内容: 一个数据集所有的 输入句子, gold 生成句子 和 gold triplets, 其格式分别为:
        input_sents:
            [['great', 'little', 'laptop', 'and', 'tablet', '!'], [...], ...]
        gold_sents:
            ['A positive sentiment great is expressed towards laptop in this sentence. [SSEP] A postive ...', '...', ...]
        tar_opi_label_idx:
            [?]
        gold_triplets:
            [[(laptop, great, positive), (tablet, great, positive)], [], ...]
    其中, 这里的 gold_triplets 尽管提取了 gold 三元组, 但是在训练中肯定用不上, 而且在 test 中, 也没用上. 因为用的是从 gold_sents 中提取出来的三元组
    Args:
        path:
        data_dir:

    Returns:
        input_sents:
        gold_sents:
        tar_opi_label_idx:

    """

    input_sents, gold_sents = [], []        # 这里仿照 GoEmotions 数据集，返回增加一个 tar_opi_label_idx，如:
                                                                #   tensor([[0, 0, 0, 1, ..., 0, 0, 0],
                                                                #           [0, 1, 0, 0, ..., 0, 0, 0],
                                                                #           ...,
                                                                #           [1, 0, 0, 0, ..., 0, 1, 0]])    (8, 50)
    df = pd.read_csv(path, sep='\t', names=['input_sentences', 'gold_triplets'])
    input_sents_list = df.iloc[:, 0].values.tolist()        # 所有输入句子的列表
    gold_triplets_list = df.iloc[:, 1].values.tolist()      # 每个句子的三元组的列表, 每个句子的三元组可能不止一个, 用 '####' 隔开

    # 构建 input_sents，即将每个输入句子用空格隔开，如:
    #   [['screen', 'is', 'good', '.'], [], ...]
    for sen in input_sents_list:
        sen = sen.strip()
        temp = sen.split()
        if temp != '':
            input_sents.append(temp)

    # 构建 gold_sents，即每个输入句子构建一个 gold 生成句子，如
    #   ['A positive sentiment pretty is expressed towards unit in this sentence. [SSEP] A positive sentiment stylish is expressed towards unit in this sentence.', ...]
    #   gold_sents 为一个列表，其每一个元素为一个 gold 生成句子
    for elem in gold_triplets_list:
        gens = []                           # 一个输入句子的 gold 生成的句子, 可能有多个, 用 [SSEP] 隔开: 'The sentiment ... [SSEP] The sentiment ...'
        triplets = elem.split("####")
        for triplet in triplets:
            t = triplet.split(',')[0].strip()
            o = triplet.split(',')[1].strip()
            s = triplet.split(',')[2].strip()
            gen = f"A {s} sentiment {o} is expressed towards {t} in this sentence."     # 针对一个输入句子的 一个 gold 生成句子 (可能存在和它并列的句子)
            gens.append(gen)
        gens_str = " [SSEP] ".join(gens)
        gold_sents.append(gens_str)

    # 以句子为单位, 构建每个句子中的共现 target, 共现 opinion 以及 共现 target-opinion, 这些 target / opinion 均以它们在 target-opinion 字典中键的索引来表示. 如:
    #   tar_opi_co_occurs_idx:
    #       [[(13, 113), (13, 114)], [(34, 97)], ...[], ...]    # 一个句子的共现 target-opinion 有可能为 [], 因为句子的 target 或者 opinion 可能比较少见而没有进入字典, 尽管每个样本都有 opinion 和 target.
    #   tar_opi_terms:
    #       为 target 和 opinion 的 term 的字典
    tar_co_occurs_idx, opi_co_occurs_idx, tar_opi_co_occurs_idx, tar_terms, opi_terms, tar_opi_terms = build_co_term_idx(f_path=path, args=args)

    # term_co_oc_idx 可以是三个中的任一个
    tar_term_idx = []
    opi_term_idx = []
    tar_opi_term_idx = []

    # tar_opi_label_idx[0:len(tar_labels), 0:len(tar_labels)] = tar_label_idx     # tar_opi_label_idx[0:80, 0:80] 被赋以 target 间的共现
    # tar_opi_label_idx[len(tar_labels):len(tar_labels)+len(opi_labels), len(tar_labels):len(tar_labels)+len(opi_labels)] = opi_label_idx

    for tar_co_occur in tar_co_occurs_idx:
        label = torch.zeros(len(tar_terms), dtype=torch.float32)
        for item in tar_co_occur:
            label[int(item)] = 1
        tar_term_idx.append(label)
    # term_co_oc_idx.append(tar_term_idx)

    for opi_co_occur in opi_co_occurs_idx:
        label = torch.zeros(len(opi_terms), dtype=torch.float32)
        for item in opi_co_occur:
            label[int(item)] = 1
        opi_term_idx.append(label)
    # term_co_oc_idx.append(opi_term_idx)


    for t_o_co_occur in tar_opi_co_occurs_idx:
        label = torch.zeros(len(tar_opi_terms), dtype=torch.float32)
        for pair in t_o_co_occur:
            if pair:
                t_idx = pair[0]
                o_idx = pair[1]
                label[t_idx] = 1  # 这里为什么是 1, 用 += 1 可不可以 ?
                label[o_idx] = 1
        tar_opi_term_idx.append(label)
    return input_sents, gold_sents, tar_term_idx, opi_term_idx, tar_opi_term_idx  # 这里为了测试，只返回 tar_label_idx


def get_Triplet_data_backup(path, data_dir):
    """
    本函数要返回的内容: 一个数据集所有的 输入句子, gold 生成句子 和 gold triplets, 其格式分别为:
        input_sents:
            [['great', 'little', 'laptop', 'and', 'tablet', '!'], [...], ...]
        gold_sents:
            ['A positive sentiment great is expressed towards laptop in this sentence. [SSEP] A postive ...', '...', ...]
        gold_triplets:
            [[(laptop, great, positive), (tablet, great, positive)], [], ...]
    其中, 这里的 gold_triplets 尽管提取了 gold 三元组, 但是在训练中肯定用不上, 而且在 test 中, 也没用上. 因为用的是从 gold_sents 中提取出来的三元组
    Args:
        path:
        data_dir:

    Returns:

    """

    input_sents, gold_sents, label_idx, gold_triplets = [], [], [], []      # 尽管这里收集到了 gold_triplets, 并且返回, 但是后续并没有用它
                                                                            # 这里仿照 GoEmotions 数据集，返回增加一个 label_idx，如:
                                                                            #   tensor([[0, 0, 0, 1, ..., 0, 0, 0],
                                                                            #           [0, 1, 0, 0, ..., 0, 0, 0],
                                                                            #           ...,
                                                                            #           [1, 0, 0, 0, ..., 0, 1, 0]])    (8, 50)

    target_terms = []           # 数据集中 (包括 dev和 test) 出现的所有 target term
    opinion_terms = []          # 数据集中 (包括 dev和 test) 出现的所有 opinion term
    target_terms_dict = {}
    opinion_terms_dict = {}

    df = pd.read_csv(path, sep='\t', names=['input_sentences', 'gold_triplets'])
    input_sents_list = df.iloc[:, 0].values.tolist()        # 所有输入句子的列表
    gold_triplets_list = df.iloc[:, 1].values.tolist()      # 每个句子的三元组的列表, 每个句子的三元组可能不止一个, 用 '####' 隔开

    # 取得数据集级别最多的三元组数目
    num_max_triplets = 1
    for elem in gold_triplets_list:
        tripl = elem.split("####")
        num_triplet = len(tripl)
        if num_triplet > num_max_triplets:
            print(tripl)
            print(num_triplet)
            num_max_triplets = num_triplet

    for elem in gold_triplets_list:
        gens = []                           # 一个输入句子的 gold 生成的句子, 可能有多个, 用 [SSEP] 隔开: 'The sentiment ... [SSEP] The sentiment ...'
        tris = []                           # 一个句子所对应的三元组, 可能有多个: [(t, o, s), (t, o, s), ...]
        triplets = elem.split("####")
        for triplet in triplets:
            t = triplet.split(',')[0].strip()
            o = triplet.split(',')[1].strip()
            s = triplet.split(',')[2].strip()
            gen = f"A {s} sentiment {o} is expressed towards {t} in this sentence."     # 针对一个输入句子的 一个 gold 生成句子 (可能存在和它并列的句子)
            gens.append(gen)
            tris.append((t, o, s))

        num_pad_tri = num_max_triplets - len(triplets)
        pad_tri = num_pad_tri * [("", "", "")]

        tris.extend(pad_tri)

        gens_str = " [SSEP] ".join(gens)
        gold_sents.append(gens_str)
        gold_triplets.append(tris)

    for sen in input_sents_list:
        sen = sen.strip()
        temp = sen.split()
        if temp != '':
            input_sents.append(temp)
    tar_co_occurs_idx, opi_co_occurs_idx, t_o_co_occur_idx, tar_keys, opi_keys, t_o_keys = data_process.build_co_terms(f_path=path, tar_limit=50, opi_limit=50)

    tar_labels = tar_keys
    opi_labels = opi_keys
    t_o_labels = t_o_keys

    # 增加一个返回 ...
    tar_label_idx = []
    opi_label_idx = []
    t_o_label_idx = []

    for tar_co_occur in tar_co_occurs_idx:
        label = torch.zeros(len(tar_labels), dtype=torch.float32)
        for item in tar_co_occur:
            label[int(item)] = 1
        tar_label_idx.append(label)

    for opi_co_occur in opi_co_occurs_idx:
        label = torch.zeros(len(opi_labels), dtype=torch.float32)
        for item in opi_co_occur:
            label[int(item)] = 1
        opi_label_idx.append(label)

    for t_o_co_occur in t_o_co_occur_idx:
        label = torch.zeros(len(t_o_labels), dtype=torch.float32)
        for item in t_o_co_occur:
            label[int(item)] = 1
        t_o_label_idx.append(label)

    return input_sents, gold_sents, t_o_label_idx  # 这里为了测试，只返回 tar_label_idx


def get_transformed_data(data_path, data_dir, data_type, args):
    if data_dir == 'SemEvalEc':
        sents, label, labels_idx = get_sem_data(data_path, data_type)
        return sents, label, labels_idx
    elif data_dir == "GoEmotions":
        sents, label, labels_idx = get_GoEmotions_data(data_path, data_dir)
        return sents, label, labels_idx
    elif data_dir == "Triplets" or data_dir == "Triplets_Restaurant":
        sents, label, tar_idx, opi_idx, tar_opi_idx = get_Triplet_data(data_path, data_dir, args)
        return sents, label, tar_idx, opi_idx, tar_opi_idx
    else:
        print("Wrong data_dir")


class EmotionDataset(Dataset):
    def __init__(self, tokenizer, data_type, data_dir, max_len, args):    # data_type: 'dev', data_dir: 'GoEmotions'
        self.tokenizer = tokenizer
        if data_dir == 'SemEvalEc':
            self.data_path = f'data/{data_dir}/{data_type}.txt'
        elif data_dir == 'GoEmotions':
            self.data_path = f'data/{data_dir}/{data_type}.tsv'
        elif data_dir == "Triplets" or data_dir == "Triplets_Restaurant":
            self.data_path = f'data/{data_dir}/Ex_and_Im/{data_type}.tsv'
        self.max_len = max_len      # 输入句子的最大长度，不包括 prompt
        self.data_dir = data_dir
        self.data_type = data_type
        self.target_length = max_len if data_dir == 'GoEmotions' else 128

        self.inputs = []
        self.targets = []

        if data_dir == "Triplets" or data_dir == "Triplets_Restaurant":
            self.tar_idx = None
            self.opi_idx = None
            self.tar_opi_idx = None
        else:
            self.label_idx = None

        self._build_examples(args)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        if self.data_dir == "Triplets" or self.data_dir == "Triplets_Restaurant":
            tar_idx = self.tar_idx[index]
            opi_idx = self.opi_idx[index]
            tar_opi_idx = self.tar_opi_idx[index]
            return {"source_ids": source_ids, "source_mask": src_mask,
                    "target_ids": target_ids, "target_mask": target_mask,
                    "tar_idx": tar_idx, "opi_idx": opi_idx, "tar_opi_idx": tar_opi_idx}
        else:
            label_idx = self.label_idx[index]
            return {"source_ids": source_ids, "source_mask": src_mask,
                    "target_ids": target_ids, "target_mask": target_mask,
                    "labels_idx": label_idx}

    def _build_examples(self, args):
        inputs, targets, tar_idx, opi_idx, tar_opi_idx = get_transformed_data(self.data_path, self.data_dir, self.data_type, args)
        # inputs:
        #   [['Is', 'this', 'in', 'New', 'Orleans??' ...], ['You', 'know', 'the', 'answer', ...], ...]
        # targets:
        #   ['The emotion neutral is expressed in this sentence.', 'The emotion approval is expressed in this sentence. [SSEP] The emotion neutral is expressed in this sentence.', ...]
        # labels_idx:
        #   [tensor([0, 1, 0, ..., 0]), tensor([0, 0, 1, ..., 0]), ...]

        for i in range(len(inputs)):
            input = ' '.join(inputs[i])
            # print(input)
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt",
            )
            # tokenized_input 是一个 BatchEncoding() 类对象:
            #   它存储下面方法的输出:
            #       meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus`
            #       meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.batch_encode` methods (tokens,
            #           attention_masks, etc.).
            #   它继承自 dict 类，也可被用作一个 dict，包含两个键:
            #       tokenized_input['input_ids'] 如:
            #           tensor([[ 27,     7,    48,    16,   368, 14433,  8546,    27,   310,   473,
            #                    114,    48,    19,   368, 14433,     5,     1,     0,     0,     0,
            #                      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            #                      ...
            #                      0,     0,     0,     0,     0,     0,     0,     0,     0,     0]])      (1, 100)
            #       tokenized_input['attention_mask']:
            #           tensor([[  1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
            #                      1,     1,     1,     1,     1,     1,     1,     0,     0,     0,
            #                      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            #                      ...
            #                      0,     0,     0,     0,     0,     0,     0,     0,     0,     0]])      (1, 100)
            #   In addition, this class
            #       exposes utility methods to map from word/character space to token space.
            #
            # print('type(tokenized_input):', type(tokenized_input))
            # print('tokenized_input:', tokenized_input)
            # print(tokenized_input['input_ids'])
            # print(tokenized_input['attention_mask'])
            # print(tokenized_input.keys())

            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.target_length, padding='max_length', truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)     # tokenized_input: [BatchEncoding(), ...], BatchEncoding() 类对象可
                                                    #   视为一个 dict，有两个键:
                                                    #       tokenized_input['input_ids']:
                                                    #           tensor([[27, 7, 48, ...]])
                                                    #       tokenized_input['attention_mask']:
                                                    #           tensor([[1, 1, 1, ...]])
            self.targets.append(tokenized_target)

            if self.data_dir == "Triplets" or self.data_dir == "Triplets_Restaurant":
                self.tar_idx = tar_idx
                self.opi_idx = opi_idx
                self.tar_opi_idx = tar_opi_idx
            else:
                print("wrong dataset.")
                # self.label_idx = labels_idx             # self.label_idx: [tensor([0, ..., 0, 1, 1, ..., 0]), ... ]

        # print(f"_build_examples completed")


def build_co_term_idx(f_path, args):
    """
    仿照 GoEmotions 数据集的形式:
        You know the answer man ... 4, 27
    即在句子后面接上句中出现的所有共现 label.

    现将一条样本句子, 也在其后接上其句中出现的共现 target, 共现 opinion 以及共现 target-opinion. 如:
        the flip and touchscreen aspects work fine , no problems .	flip, fine, positive####touchscreen, fine, positive
            -->
        the flip and touchscreen aspects work fine , no problems .  (flip, touchscreen), (fine, fine), [(flip, fine), (touchscreen, fine)]
                                                                    |-----------------|  |----------|  |---------------------------------|
                                                                        共现 aspect       共现 opinion          共现 aspect-opinion
    并将它们转为 term 在字典键中的索引.
    在这里要注意的是:
        1. 这里最后得到的共现结果是 各个 term 在字典所有键中的索引
        2. target 或者 opinion 必须出现在统计词典里, 否则因出现频次太少而舍弃
        3. 若共现 term 是重复的, 则舍弃重复的. 如:
            (fine, fine) --> (fine)
    Returns:
        1. tar_co_occurs_idx:
            为一个 list, 长度为 1407, 其每个元素为一个样本中 (同时) 出现的 target 在字典所有键中的索引 (只有出现达到一定次数才可以进入到字典中). 如:
                [[0, 2, 12], [4, 1], [], [], ...]
        2. opi_co_occurs_idx:
            为一个 list, 长度为 1407, 其每个元素为一个样本中 (同时) 出现的 opinion 在字典所有键中的索引 (只有出现达到一定次数才可以进入到字典中). 如:
                [[1, 32, 15], [20, 2], [26], [], ...]
        3. tar_opi_co_occurs_idx:
            为一个 list, 长度为 1407, 其每个元素为一个样本中出现的 target-opinion 对在字典所有键中的索引 (target 和 opinion 必须都出现达到一定次数才可以进入到字典中). 如:
                [[(0, 82), (12, 65), (2, 51)], [(1, 52), (4, 70)], [], [], ...]
        4. tar_keys:
            target 字典的键, 取出现次数前 80 的 target
        5. opi_keys:
            opinion 字典的键, 取出现次数前 80 的 opinion
        6. tar_opi_keys:
            tar_keys 和 opi_keys 的合并
    """

    # 构建数据集中出现的所有 target 和 opinion 的字典, 其中:
    #   键 为 term 本身,
    #   值 为 term 出现的次数.
    # 按出现次数降序排列, 只取出现次数在前 80 位的键值对.
    # tar_opi_dict 为前两个 dict 的简单合并, 也没有对键值对重新排序
    tar_dict, opi_dict, tar_opi_dict = build_term_dict(f_path=f_path, tar_limit=args.tar_limit, opi_limit=args.opi_limit)

    # 以下为一个构造过程, 构造给出的三个空 list
    tar_co_occurs = []          # 为数据集级别的, 其每个元素为将某个句子出现的所有 target 放在一起, 类似于 GoEmotions 数据集中一条样本后的 label:'4, 27'. 如:
                                #   [['chromebook', 'screen'], ['product', 'customer service'], ...]
    opi_co_occurs = []          # 为数据集级别的, 其每个元素为将某个句子出现的所有 opinion 放在一起, 如:
                                #   [['beautiful', 'unusable', 'bummed', 'nice'], ['well constructed', 'solid', 'easy'], ...]
    tar_opi_co_occurs = []      # 为数据集级别的, 其每个元素为将某个句子出现的 target-opinion 放在一起, 如:
                                #   [[('unit', 'pretty'), ('unit', 'stylish')], [('version', 'least favorite')], ...]
    with open(f_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # sent_str = line.split("\t")[0]
            tri_str = line.split("\t")[1]

            tar_in_example = []         # 一个句子里出现的所有 target, 即为找样本里共现的 target, 此时还是单词的形式. target 需要出现在 tar_dict 里
            opi_in_example = []         # 一个句子里出现的所有 opinion, 即为找样本里共现的 opinion, 此时还是单词的形式. opinion 需要出现在 opi_dict 里
            tar_opi_in_example = []     # 一个句子里出现的所有 target-opinion 对, 即为找到样本里的共现 target-opinion. target 和 opinion 都需要出现在各自的字典里
            triplets = tri_str.split("####")
            for tri in triplets:
                tar = tri.split(", ")[0]
                if tar in tar_dict.keys():
                    tar_in_example.append(tar)

                opi = tri.split(", ")[1]
                if opi in opi_dict.keys():
                    opi_in_example.append(opi)

                if tar in tar_dict.keys() and opi in opi_dict.keys():
                    tar_opi = (tar, opi)
                    tar_opi_in_example.append(tar_opi)
                elif tar in tar_dict.keys() and opi not in opi_dict.keys():     # 这里的两个条件需要观察一下 GoEmotions 数据集里对 只有一个标签 是如何处理的
                    pass
                elif tar not in tar_dict.keys() and opi in opi_dict.keys():     # 这里的两个条件需要观察一下 GoEmotions 数据集里对 只有一个标签 是如何处理的
                    pass

            tar_in_example = list(set(tar_in_example))          # 去除掉一个样本包含的重复的 target, 如: ['laptop', 'speakers', 'laptop'] -> ['laptop', 'speakers'].
            opi_in_example = list(set(opi_in_example))
            tar_opi_in_example = list(set(tar_opi_in_example))

            tar_co_occurs.append(tar_in_example)                # [['unit'], ...]
            opi_co_occurs.append(opi_in_example)                # [['pretty', 'stylish'], ...]
            tar_opi_co_occurs.append(tar_opi_in_example)        # [[('unit', 'pretty'), ('unit', 'stylish')], ...]

    # 将上面得到的 tar_opi_co_occurs 的实际单词转换为单词在字典键中的索引
    tar_opi_terms = list(tar_opi_dict.keys())                   # 取得所有的 target 和 opinion term, 作为一个 list
    tar_opi_co_occurs_idx = []                                  # 共现的 target 和 opinion 在上面 list 中的索引
    for line in tar_opi_co_occurs:
        tar_opi_idx_in_line = []
        for tar_opi in line:
            tar = tar_opi[0]
            tar_idx = tar_opi_terms.index(tar)
            opi = tar_opi[1]
            opi_idx = tar_opi_terms.index(opi)
            tar_opi_idx = (tar_idx, opi_idx)
            tar_opi_idx_in_line.append(tar_opi_idx)
        tar_opi_co_occurs_idx.append(tar_opi_idx_in_line)       # [[('unit', 'pretty'), ('unit', 'stylish')], ...]  -->
                                                                # [[(  13,     113   ), (  13,      114   )], ...]

    # 将上面得到的 tar_co_occurs 的实际单词转换为单词在字典键中的索引
    if args.unify_dict:
        tar_terms = list(tar_opi_dict.keys())  # 由 target opinion 联合构建的字典
        tar_terms[args.tar_limit:] = args.tar_limit * [None]
    else:
        tar_terms = list(tar_dict.keys())  # 由 target 单独构建的字典
    tar_co_occurs_idx = []
    for line in tar_co_occurs:
        t_indexes = []
        for t in line:
            idx = tar_terms.index(t)
            t_indexes.append(idx)
        tar_co_occurs_idx.append(t_indexes)

    # 将上面得到的 opi_co_occurs 的实际单词转换为单词在字典键中的索引
    if args.unify_dict:
        opi_terms = list(tar_opi_dict.keys())  # 也可采用 target opinion 联合构建的字典
        opi_terms[0:args.tar_limit] = args.tar_limit * [None]
    else:
        opi_terms = list(opi_dict.keys())  # 这里是采用单独由 opinion 构建的字典
    opi_co_occurs_idx = []

    print(f"f_path: {f_path}")
    for i, line in enumerate(opi_co_occurs):
        # print(f"idx: {idx}, line: {line}")
        o_indexes = []
        for o in line:
            # print(f"\to: {o}")
            # print(f"opi_terms: {opi_terms}")
            idx = opi_terms.index(o)
            # print(f"\tidx: {idx}")
            o_indexes.append(idx)
        opi_co_occurs_idx.append(o_indexes)

    # 是否将 共现 target, 共现 opinion 以及 共现 target-opinion 三者合并, 得到一个 all_co_occurs_idx
    # if args.combine_all_co_occurs:
    #     all_co_occurs = combine_all_co_occurs(tar_co_occurs_idx, opi_co_occurs_idx, tar_opi_co_occurs_idx)

    return tar_co_occurs_idx, opi_co_occurs_idx, tar_opi_co_occurs_idx, tar_terms, opi_terms, tar_opi_terms
    # 1. tar_co_occurs_idx, 为一个 list, 长度为 1407, 其每个元素为一个样本中(同时)出现的 target 在字典所有键中的索引(只有出现达到一定次数才可以进入到字典中). 如:
    #       [[0, 2, 12], [4, 1], [], [], ...]
    # 2. opi_co_occurs_idx, 为一个 list, 长度为 1407, 其每个元素为一个样本中(同时)出现的 opinion 在字典所有键中的索引(只有出现达到一定次数才可以进入到字典中). 如:
    #       [[1, 32, 15], [20, 2], [26], [], ...]
    # 3. tar_opi_co_occurs_idx, 为一个 list, 长度为 1407, 其每个元素为一个样本中出现的 target-opinion 对在字典所有键中的索引(target 和 opinion 必须都出现达到一定次数才可以进入到字典中). 如:
    #       [[(0, 82), (12, 65), (2, 51)], [(1, 52), (4, 70)], [], [], ...]
    # 4. tar_terms: target 字典的键, 取出现次数前 50 的 target
    # 5. opi_terms: opinion 字典的键, 取出现次数前 50 的 opinion
    # 6. tar_opi_terms: tar_keys 和 opi_keys 的合并


def combine_all_co_occurs(tar_co_occurs_idx, opi_co_occurs_idx, tar_opi_co_occurs_idx):
    all_co_occurs = []
    # 将 tar_co_occurs 和 opi_co_occurs 合并进 tar_opi_co_occurs

    assert len(tar_co_occurs_idx) == len(tar_opi_co_occurs_idx)
    assert len(opi_co_occurs_idx) == len(tar_opi_co_occurs_idx)

    for idx, t_o_co_oc in enumerate(tar_opi_co_occurs_idx):
        tmp = copy.deepcopy(t_o_co_oc)
        tar_co_oc = tuple(tar_co_occurs_idx[idx])
        opi_co_oc = tuple(opi_co_occurs_idx[idx])

        tmp.append(tar_co_oc)
        tmp.append(opi_co_oc)

        all_co_occurs.append(tmp)

    return all_co_occurs


def build_term_dict(f_path, tar_limit=80, opi_limit=80):
    """
    构建训练集中 target / opinion 的字典, 其中:
        键为 term 本身,
        值为 term 的出现次数,
    进入字典的 term 需要满足一定的出现次数. 用于筛选部分 term, 避免 term 过多.
    Args:
        f_path:
        tar_limit:
        opi_limit:

    Returns: 返回统计词典
        tar_dict:
            target 及其出现次数的字典, 按出现次数降序排列
        opi_dict:
            opinion 及其出现次数的字典, 按出现次数降序排列
        tar_opi_dict:
            tar_dict 和 opi_dict 的单纯合并, 并没有重新排序
    """

    tar_dict = {}       # 样本集中出现的 target 的 dict (键为 target, 值为出现次数), 按出现次数做降序排序 (取前50个)
    opi_dict = {}       # 样本集中出现的 opinion 的 dict (键为 opinion, 值为出现次数), 按出现次数做降序排序 (取前50个)
    tar_opi_dict = {}   # 为上面 tar_dict 和 opi_dict 的单纯合并

    targets_list = []   # 用来存放训练集中出现的所有 target term,一开始是 2016, 去重之后是 489
    opinions_list = []  # 用来存放训练集中出现的所有 opinion term, 一开始是 2016, 去重之后是 505

    with open(f_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            tar_in_line = []    # 一句话里出现的 target
            opi_in_line = []    # 一句话里出现的 opinion
            sent_str = line.split("\t")[0]
            tri_str = line.split("\t")[1]

            triplets = tri_str.split("####")
            for tri in triplets:            # 对于句子里的每一个三元组
                tar = tri.split(", ")[0]
                tar_in_line.append(tar)
                opi = tri.split(", ")[1]
                opi_in_line.append(opi)
            # tar_in_line = list(set(tar_in_line))    # 句子里出现的 target, 去除重复的
            # opi_in_line = list(set(opi_in_line))    # 句子里出现的 opinion, 去除重复的
            targets_list.extend(tar_in_line)        # 将 tar_in_line 统一添加, 总共 2016 个 target
            opinions_list.extend(opi_in_line)       # 将 opi_in_line 统一添加, 总共 2016 个 opinion

        for target in targets_list:                 # 构造 opinion 出现次数的 dict
            if target not in tar_dict.keys():
                tar_dict[target] = 1
            else:
                tar_dict[target] += 1               # 数据集中每种 target 出现次数的 dict. tar_dict:
                                                            #   {
                                                            #       "screen": 72,
                                                            #       "chromebook": 96,
                                                            #       ...
                                                            #   }

        for opinion in opinions_list:               # 构造 opinion 出现次数的 dict
            if opinion not in opi_dict.keys():
                opi_dict[opinion] = 1
            else:
                opi_dict[opinion] += 1              # 数据集中每种 opinion 出现次数的 dict. opi_dict:
                                                            #   {
                                                            #       "nice": 82,
                                                            #       "good": 104,
                                                            #       ...
                                                            #   }

        tar_dict = dict(sorted(tar_dict.items(), key=lambda x: x[1], reverse=True))     # 将 tar_dict 按出现次数做降序排序
        opi_dict = dict(sorted(opi_dict.items(), key=lambda x: x[1], reverse=True))     # 将 opi_dict 按出现次数做降序排序

        print(sum(tar_dict.values()))
        print(sum(opi_dict.values()))

        tar_json = json.dumps(tar_dict, indent=4)
        with open("tar_dict.json", "w", encoding="utf-8") as f:
            f.write(tar_json)

        opi_json = json.dumps(opi_dict, indent=4)
        with open("opi_dict.json", "w", encoding="utf-8") as f:
            f.write(opi_json)

        tar_dict = dict(itertools.islice(tar_dict.items(), tar_limit))                  # 将 dict 做截断, 只保留出现一定次数 term
        opi_dict = dict(itertools.islice(opi_dict.items(), opi_limit))                  # 将 dict 做截断, 只保留出现一定次数 term

        # 将 tar_dict 和 opi_dict 单纯合并
        tar_opi_dict.update(tar_dict)
        tar_opi_dict.update(opi_dict)

        return tar_dict, opi_dict, tar_opi_dict        # 返回统计词典

