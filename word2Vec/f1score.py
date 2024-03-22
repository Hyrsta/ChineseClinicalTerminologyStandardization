# _*_ coding=utf-8 _*_
import json     # 读取json数据集


# 计算精确率，召回率和f1分数的函数
def metrics(answer_list, predictions, total_answer):  # 参数分别为：正确答案和预测答案列表以及答案总数
    true_pos = 0        # 存储真阳性
    pos_pred = 0        # 存储阳性
    prediction_list_split = [item.split('##') for item in predictions]  # 存储由'##'分割后的预测答案
    # answer_list_length = len(answer_list)
    # 计算真阳性和阳性
    for prediction, answer in zip(prediction_list_split, answer_list):
        for sentence in prediction:
            if sentence != 'no_similar_sentence':
                if sentence in answer:
                    true_pos += 1
                    pos_pred += 1
                else:
                    pos_pred += 1
    print("真阳性数量: {}\n预测总数: {}\n答案总数: {}\n".format(true_pos, pos_pred, total_answer))

    # 计算精确率，召回率和f1分数
    precision = true_pos / pos_pred
    recall = true_pos / total_answer
    f1score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1score


def data_metric(dataset_paths, prediction_paths):    # 参数分别为：数据集和预测答案的路径
    # 读取数据集文件和预测答案文件
    with open(prediction_paths, 'r', encoding='utf-8') as file:
        json_prediction = json.load(file)

    with open(dataset_paths, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # 由数据集和预测答案分别创建以诊断词为键和标准词为值的字典
    prediction_dict = {}
    for item in json_prediction:
        prediction_dict[item['text']] = item['normalized_result']

    data_dict = {}
    for item in json_data:
        data_dict[item['text']] = item['normalized_result']

    # 创建列表存储预测答案的标准词
    predictions = list(prediction_dict.values())

    # 创建列表存储数据集的标准词
    nonP_result = list(data_dict.values())
    nonP_result_split = []      # 存储由'##'分割后的标准词的列表
    total_answer = 0            # 存储数据集中标准词的总数
    for item in nonP_result:
        nonP_result_split.append(item.split('##'))
        total_answer += len(item.split('##'))

    # 计算精确率，召回率和f1分数
    precision, recall, f1score = metrics(nonP_result_split, predictions, total_answer)
    return precision, recall, f1score


# 文件的开始
if __name__ == '__main__':
    # 数据集的路径
    dataset_paths = ['dataset/dev.json']

    # 预测答案的路径
    # prediction_paths = ['dataset/dev.json']
    prediction_paths = ['prediction/prednum_comparison/m5d0_Dtrain_v10.json']
    # 计算精确率，召回率和f1分数
    for dataset_path, prediction_path in zip(dataset_paths, prediction_paths):
        print("target prediction file: ", prediction_path)
        precision, recall, f1score = data_metric(dataset_path, prediction_path)
        print("精确率:{}\n召回率:{}\nf1值:{}\n".format(precision, recall, f1score))
