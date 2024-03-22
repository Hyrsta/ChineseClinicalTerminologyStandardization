# _*_ coding=utf-8 _*_
import pre      # 读取预处理文件
import model    # 读取模型文件
import json     # 读取json数据集
import openpyxl
from gensim.models import Word2Vec


# 文件的开始
if __name__ == '__main__':
    # 查看是否可用cuda

    # 读取停止词文件
    with open('dataset/stop_words.txt', 'r', encoding='utf-8') as file:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = file.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.strip()  # 去掉读取每一行数据的\n
            stop_words.add(i)

    # 导入xlsx文件里的标准病毒
    xlsx_file = openpyxl.load_workbook("dataset/鍥介檯鐤剧梾鍒嗙被 ICD-10鍖椾含涓村簥鐗坴601.xlsx")
    nonP_standard = []
    for index, row in enumerate(xlsx_file.worksheets[0]):
        if row[1].value not in nonP_standard:
            nonP_standard.append(row[1].value)

    # 导入数据集
    train_path = 'dataset/train.json'
    nonP_train_text, nonP_train_result, nonP_train_result_list = pre.dataset_to_list(train_path)
    print("successfully load", train_path)
    dev_path = 'dataset/dev.json'
    nonP_dev_text, nonP_dev_result, nonP_dev_result_list = pre.dataset_to_list(dev_path)
    print("successfully load", dev_path)

    # 数据预处理
    # 删除诊断词里的停用词并tokenize句子
    pre_train_text = [pre.delete_stop_words(item, stop_words) for item in nonP_train_text]
    pre_dev_text = [pre.delete_stop_words(item, stop_words) for item in nonP_dev_text]

    # 删除标准词里的标点符号并tokenize句子
    pre_train_result = [pre.delete_punctuations(item) for item in nonP_train_result_list]
    pre_dev_result = [pre.delete_punctuations(item) for item in nonP_dev_result_list]
    pre_standard = [pre.delete_punctuations(item) for item in nonP_standard]

    # 导入模型
    my_model = Word2Vec.load("model/train.model")
    print("successfully loading model")
    print(my_model)
    print(f"window:{my_model.window}, epochs:{my_model.epochs}")
    # 计算诊断词的句子向量
    train_text_vector = model.count_sentence_vector(pre_train_text, my_model)
    dev_text_vector = model.count_sentence_vector(pre_dev_text, my_model)

    # 计算标准词的句子向量
    train_result_vector = model.count_sentence_vector(pre_train_result, my_model)
    dev_result_vector = model.count_sentence_vector(pre_dev_result, my_model)
    standard_vector = model.count_sentence_vector(pre_standard, my_model)
    print("successfully creating sentence vectors")

    # 通过余弦相似度计算对text_vector的预测答案
    predictions = model.sentence_cosine_similarity(dev_text_vector, train_result_vector, nonP_train_result_list,
                                                   nonP_dev_text, nonP_dev_result, percentage=10, distance=0, max_ans=2)

    save_list = []
    for data, prediction in zip(nonP_dev_text, predictions):
        save_list.append({"text": data, "normalized_result": prediction})

    # 保存预测答案
    with open('prediction/prednum_comparison/m2d0_Dtrain_v10.json', 'w', encoding='utf-8') as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)
