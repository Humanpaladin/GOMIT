import pandas as pd


def get_triplet_data(path, data_dir):
    input_sents, gen_sents, gold_triplets = [], [], []      # 本函数要返回的内容: 一个数据集所有的 输入句子, 生成句子 和 gold triplets, 其格式分别为:
                                                            #   input_sents:
                                                            #       [['great', 'little', 'laptop', 'and', 'tablet', '!'], [...], ...]
                                                            #   gen_sents:
                                                            #       ['A positive sentiment great is expressed towards laptop in this sentence. [SSEP] A postive ...', '...', ...]
                                                            #   gold_triplets:
                                                            #       [[(laptop, great, positive), (tablet, great, positive)], [], ...]

    target_terms = []           # 数据集中 (包括 dev和 test) 出现的所有 target term
    opinion_terms = []          # 数据集中 (包括 dev和 test) 出现的所有 opinion term
    target_terms_dict = {}
    opinion_terms_dict = {}

    df = pd.read_csv(path, sep='\t', names=['input_sentences', 'gold_triplets'])
    input_sents_list = df.iloc[:, 0].values.tolist()    # 所有输入句子的列表
    gold_triplets_list = df.iloc[:, 1].values.tolist()      # 每个句子的三元组的列表, 每个句子的三元组可能不止一个, 用 '####' 隔开

    for elem in gold_triplets_list:
        gens = []                           # 针对一个输入句子所生成的句子, 可能有多个, 用 [SSEP] 隔开: 'The sentiment ... [SSEP] The sentiment ...'
        tris = []                           # 一个句子所对应的三元组, 可能有多个: [(t, o, s), (t, o, s), ...]
        triplets = elem.split("####")
        for triplet in triplets:
            t = triplet.split(',')[0].strip()
            o = triplet.split(',')[1].strip()
            s = triplet.split(',')[2].strip()
            gen = f"A {s} sentiment {o} is expressed towards {t} in this sentence."     # 针对一个句子所生成的一个句子 (可能存在和它并列的句子)
            gens.append(gen)
            tris.append((t, o, s))
        gens_str = " [SSEP] ".join(gens)
        gen_sents.append(gens_str)
        gold_triplets.append(tris)

    for sen in input_sents_list:
        sen = sen.strip()
        temp = sen.split()
        if temp != '':
            input_sents.append(temp)

    print("test")

    return input_sents, gen_sents, gold_triplets


