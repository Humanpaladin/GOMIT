import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt


def get_co_existence(dataset, label_size, term_co_occurs_idx=None):
    labels_list = None
    if dataset == 'GoEmotions':
        df = pd.read_csv(f'data/{dataset}/train.tsv', sep="\t", names=['sentence', 'label', 'null'])
        label_size = 28                         # emotion 的个数, 28
        labels_list = df.iloc[:, 1].values      # 每个样本对应的 emotion, 可视为一个 list, 如: ['27' '27' '2' ... '6,22' '6,9,27' '12' '27' ...]
    elif dataset == "SemEvalEc":
        df = pd.read_csv(f'data/{dataset}/train.txt', sep="	")
        label_size = 11
    elif dataset == "Triplets" or dataset == "Triplets_Restaurant":                 # 目的是想像 GoEmotions 一样，得到一个 label_list
        labels_list = term_co_occurs_idx
    relation_matrix = np.zeros((label_size, label_size), dtype=int)     # (28, 28) 的全零矩阵. 用来存放 28 个 label 任意两个之间的共现次数

    def permutation(tup, k):
        # 函数的输入为:
        #   1. tup
        #       tup 为某个样本的 emotion, 如 (8, 20). 意为在该样本中, emotion 8 和 10 共现
        #   2. k
        #       k 为 2, 但不知为何为 2
        # 函数的输出为:
        #   result
        #       result 如 [[8, 20], [20, 8]], 用于构造 weight 矩阵

        lst = list(tup)
        result = []
        tmp = [0] * k

        def next_num(a, ni=0):
            if ni == k:
                result.append(copy.copy(tmp))
                return
            for lj in a:
                tmp[ni] = lj
                b = a[:]
                b.pop(a.index(lj))
                next_num(b, ni + 1)

        c = lst[:]
        next_num(c, 0)
        return result

    if dataset == 'GoEmotions':
        for labels in labels_list:
            if labels != "":
                items = labels.split(",")
                labels_int = list(map(int, items))
                labels_tuple = tuple(labels_int)
                if len(labels_tuple) >= 2:  # 即某个样本对应 emotion 不止一个, 该样本的 emotion 出现共现. 如 (8, 20)
                    all_relation = permutation(labels_tuple, 2)
                    for relation in all_relation:
                        relation_matrix[relation[0]][relation[1]] += 1
            else:
                print(f"labels: {labels}")

    elif dataset == 'Triplets' or dataset == "Triplets_Restaurant":
        for labels in labels_list:
            if labels:
                for co_occ in labels:
                    if len(co_occ) >= 2:
                        relations = permutation(co_occ, 2)
                        for rela in relations:
                            relation_matrix[rela[0]][rela[1]] += 1

    elif dataset == 'SemEvalEc':
        label_array = df.iloc[:, 2:].values.tolist()
        for temp in label_array:
            indexes = [inx for inx, num in enumerate(temp) if num == 1]
            if len(indexes) > 1:
                all_relation = permutation(indexes, 2)
                for relation in all_relation:
                    relation_matrix[relation[0]][relation[1]] += 1

    return relation_matrix


def get_co_existence_tar(dataset, label_size, tar_co_occurs_idx=None):
    if dataset == 'GoEmotions':
        df = pd.read_csv(f'data/{dataset}/train.tsv', sep="\t", names=['sentence', 'label', 'null'])
        label_size = 28                         # emotion 的个数, 28
        labels_list = df.iloc[:, 1].values      # 每个样本对应的 emotion, 可视为一个 list, 如: ['27' '27' '2' ... '6,22' '6,9,27' '12' '27' ...]
    elif dataset == "SemEvalEc":
        df = pd.read_csv(f'data/{dataset}/train.txt', sep="	")
        label_size = 11
    elif dataset == "Triplets" or dataset == "Triplets_Restaurant":                 # 目的是想像 GoEmotions 一样，得到一个 label_list
        df = pd.read_csv(f"data/{dataset}/Ex_and_Im/train.tsv", sep="\t", names=['sentence', 'triplets'])
        labels_list = []
        for item in tar_co_occurs_idx:
            labels_str = ""
            for label in item:
                labels_str += str(label) + ","
            labels_str = labels_str.strip(",")
            labels_list.append(labels_str)

    relation_matrix = np.zeros((label_size, label_size), dtype=int)

    def permutation(tup, k):
        # 函数的输入为:
        #   1. tup
        #       tup 为某个样本的 emotion, 如 (8, 20). 意为在该样本中, emotion 8 和 10 共现
        #   2. k
        #       k 为 2, 但不知为何为 2
        # 函数的输出为:
        #   result
        #       result 如 [[8, 20], [20, 8]], 用于构造 weight 矩阵

        lst = list(tup)
        result = []
        tmp = [0] * k

        def next_num(a, ni=0):
            if ni == k:
                result.append(copy.copy(tmp))
                return
            for lj in a:
                tmp[ni] = lj
                b = a[:]
                b.pop(a.index(lj))
                next_num(b, ni + 1)

        c = lst[:]
        next_num(c, 0)
        return result

    if dataset == 'GoEmotions':
        for labels in labels_list:
            if labels != "":
                items = labels.split(",")
                labels_int = []

                for item in items:
                    item = int(item)
                    labels_int.append(item)

                labels_tuple = tuple(labels_int)
                if len(labels_tuple) >= 2:
                    all_relation = permutation(labels_tuple, 2)
                    for relation in all_relation:
                        relation_matrix[relation[0]][relation[1]] += 1
            else:
                print(f"labels: {labels}")

    elif dataset == 'Triplets' or dataset == "Triplets_Restaurant":
        for labels in labels_list:
            if labels != "":
                items = labels.split(",")
                labels_int = []

                for item in items:
                    item = int(item)
                    labels_int.append(item)

                labels_tuple = tuple(labels_int)
                if len(labels_tuple) >= 2:
                    all_relation = permutation(labels_tuple, 2)
                    for relation in all_relation:
                        relation_matrix[relation[0]][relation[1]] += 1
            else:
                pass
    elif dataset == 'SemEvalEc':
        label_array = df.iloc[:, 2:].values.tolist()
        for temp in label_array:
            indexes = [inx for inx, num in enumerate(temp) if num == 1]
            if len(indexes) > 1:
                all_relation = permutation(indexes, 2)
                for relation in all_relation:
                    relation_matrix[relation[0]][relation[1]] += 1

    return relation_matrix




