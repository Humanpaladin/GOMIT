import json


def count_categories(f_path):
    """
    统计某个文件里的 category 的种类
    Args:
        f_path:

    Returns:

    """
    with open(f_path, "r", encoding='utf-8') as f:
        categories = []
        for line in f:
            line = line.strip()
            items = line.split("\t")
            sent = items[0]
            triplets = items[1:]

            for tri in triplets:
                category = tri.split(' ')[1]
                if category not in categories:
                    categories.append(category)

    return categories


def filter_examples(file_path):
    """
    找出数据集中 "不含隐式 aspect 且不含隐式 opinion" 的样本
    Args:
        file_path:

    Returns:

    """
    with open(file_path, 'r', encoding="utf-8") as f:
        examples_with_no_ito = []
        for line in f:
            line = line.strip()
            items = line.split("\t")
            sent = items[0]
            triplets = items[1:]

            flag = False
            for tri in triplets:
                if "-1,-1" in tri:
                    flag = True

            if flag is False:
                examples_with_no_ito.append(line)

        return examples_with_no_ito


def convert_dataset(r_path, w_path):
    """
    将初始的 ACOS 数据集文件转换为 Triplet 要用的明确的三元组形式, 如:
        初始的 ACOS 数据用 index 表示:
            great little laptop and tablet !	2,3 LAPTOP#GENERAL 2 0,1	4,5 LAPTOP#GENERAL 2 0,1
                ->
        Triplets 数据直接用单词表示:
            great little laptop and tablet !	laptop, great, positive####tablet, great, positive
    Returns:

    """
    #   great little laptop and tablet !	2,3 LAPTOP#GENERAL 2 0,1	4,5 LAPTOP#GENERAL 2 0,1
    #   ->
    #   great little laptop and tablet !	laptop, great, positive####tablet, great, positive
    with open(r_path, "r", encoding="utf-8") as f:
        senti_dict = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

        new_lines = []
        for line in f:
            line = line.strip()
            items = line.split("\t")
            sent = items[0]
            triplets = items[1:]

            tri_str_temp = []
            for tri in triplets:
                items = tri.split(" ")
                target = items[0]
                category = items[1]
                sentiment = items[2]
                opinion = items[3]

                target_b, target_e = int(target.split(",")[0]), int(target.split(",")[1])
                opinion_b, opinion_e = int(opinion.split(",")[0]), int(opinion.split(",")[1])

                target_term = " ".join(sent.split(" ")[target_b: target_e])
                opinion_term = " ".join(sent.split(" ")[opinion_b: opinion_e])
                sentiment_term = senti_dict[int(sentiment)]

                tri_temp = ", ".join([target_term, opinion_term, sentiment_term])
                tri_str_temp.append(tri_temp)

            tri_str = "####".join(tri_str_temp)

            new_lines.append(sent + "\t" + tri_str)

    with open(w_path, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")


def convert_dataset_update(r_path, w_path):
    """
    本函數主要处理包含隐式 term 的样本, 将它们做如下形式的转换:
    将初始的 ACOS 数据集文件转换为 Triplet 要用的明确的三元组形式, 如:
        初始的 ACOS 数据用 index 表示:
            too large for just two people but nothing was left .	-1,-1 FOOD#STYLE_OPTIONS 0 0,2	-1,-1 FOOD#QUALITY 2 -1,-1  ||||    food, food, good
                ->
        Triplets 数据直接用单词表示:
            too large for just two people but nothing was left .	food, too large, positive####food, good, positive
    Returns:

    """
    #   great little laptop and tablet !	2,3 LAPTOP#GENERAL 2 0,1	4,5 LAPTOP#GENERAL 2 0,1
    #   ->
    #   great little laptop and tablet !	laptop, great, positive####tablet, great, positive

    with open(r_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        for line in data:

            # 1. 简单判断一下标注的隐式 term 数目是否等于样本中的隐式 term 数目:
            it_count = line["data"].count("-1,-1")  # 样本中 -1,-1 出现的次数, 即隐式 term 的个数
            annos_temp = []
            if line["label"]:
                annos_temp = line["label"][0].strip(",").split(",")
            else:
                print(f"Error: This sentence is not annotated.\n\tline: {line}")

            annos = []
            for ann in annos_temp:
                a = ann.strip()
                annos.append(a)
            if it_count != len(annos):
                print(f"The number of annotation is not equal to the number of implicit term in this sentence: {line}")

            """ 
            annos_temp = line.split("||-->")[-1]
            annos_temp = annos_temp.split(",")
            annos = []
            for ann in annos_temp:
                a = ann.strip()
                annos.append(a)
            if it_count != len(annos):
                print(f"The number of annotation is not equal to the number of implicit term in this sentence: {line}")
            print("test")
            """
    with open(r_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        senti_dict = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

        new_lines = []
        for line in data:
            # line = line.strip()
            # example = line.split("||-->")[0].strip()
            example = line["data"].strip()
            # anno = line.split("||-->")[1].strip()
            anno = line["label"][0].strip()
            anno = anno.strip(",")

            sent = example.split("\t")[0]
            triplets = example.split("\t")[1:]

            tri_str_temp = []
            for tri in triplets:
                items = tri.split(" ")
                target = items[0]
                # category = items[1]
                sentiment = items[2]
                opinion = items[3]

                if target != "-1,-1":
                    target_b, target_e = int(target.split(",")[0]), int(target.split(",")[1])
                    target_term = " ".join(sent.split(" ")[target_b: target_e])
                else:
                    target_term = "NONE-TERM"

                if opinion != "-1,-1":
                    opinion_b, opinion_e = int(opinion.split(",")[0]), int(opinion.split(",")[1])
                    opinion_term = " ".join(sent.split(" ")[opinion_b: opinion_e])
                else:
                    opinion_term = "NONE-TERM"

                sentiment_term = senti_dict[int(sentiment)]

                tri_temp = ", ".join([target_term, opinion_term, sentiment_term])
                tri_str_temp.append(tri_temp)

            tri_str = "####".join(tri_str_temp)

            annos = []
            for a in anno.split(","):
                a = a.strip()
                annos.append(a)

            for a in annos:
                tri_str = tri_str.replace("NONE-TERM", a, 1)

            new_lines.append(sent + "\t" + tri_str)

    with open(w_path, "w", encoding="utf-8") as f:
        for line in new_lines:
            # print(f"line: {line}")
            f.write(line + "\n")


def filter_examples_ito(file_path):
    """
    找出数据集中 "包含隐式 aspect 或隐式 opinion" 的样本
    Args:
        file_path:

    Returns:

    """
    with open(file_path, 'r', encoding="utf-8") as f:
        examples_with_ito = []
        for line in f:
            line = line.strip()
            # items = line.split("\t")
            # sent = items[0]
            # triplets = items[1:]
            if "-1," in line:
                examples_with_ito.append(line)

            """
            flag = False
            for tri in triplets:
                if "-1,-1" in tri:
                    flag = True

            if flag is False:
                examples_with_ito.append(line)
            """
        return examples_with_ito


