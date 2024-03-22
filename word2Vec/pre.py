# _*_ coding=utf-8 _*_
import jieba
import string
import json


def dataset_to_list(dataset_path):
    # 读取数据集文件
    with open(dataset_path, 'r', encoding='utf-8') as file:
        train_json = json.load(file)

    # 由数据集创建以诊断词为键和标准词为值的字典
    train_dict = {}
    for item in train_json:
        train_dict[item['text']] = item['normalized_result']

    # 创建两个列表，分别存储数据集的诊断词和标准词
    nonP_text = list(train_dict.keys())
    nonP_result = list(train_dict.values())
    nonP_result_list = []
    for item in nonP_result:
        for sentence in item.split('##'):
            if sentence not in nonP_result_list:
                nonP_result_list.append(sentence)

    return nonP_text, nonP_result, nonP_result_list


# 去除停用词并将句子分词
def delete_stop_words(input_string, stop_words_list):
    # chinese_only = re.sub('[^\u4e00-\u9fa5]', '', input_string)
    result = []
    for word in jieba.lcut(input_string):
        if word not in stop_words_list:
            result.append(word)
    return result


def delete_punctuations(input_string):
    result = []
    for word in jieba.lcut(input_string):
        if word not in string.punctuation:
            result.append(word)
    return result


# test
if __name__ == '__main__':
    with open('dataset/stop_words.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = f.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.strip()  # 去掉读取每一行数据的\n
            stop_words.add(i)
    query = '右肺,结节转移c1c1'
    st = delete_stop_words(query, stop_words)
    print(st)
