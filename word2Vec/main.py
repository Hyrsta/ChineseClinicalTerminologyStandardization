import model    # 读取模型文件
from gensim.models import Word2Vec
import json
import pre      # 读取预处理文件

if __name__ == '__main__':
    # 导入数据集
    train_path = 'dataset/train.json'
    nonP_train_text, nonP_train_result, nonP_train_result_list = pre.dataset_to_list(train_path)
    print("successfully load", train_path)

    # 导入测试集
    test_path = 'dataset/test.json'
    nonP_test_text, _, _ = pre.dataset_to_list(test_path)

    # 读取停止词文件
    with open('dataset/stop_words.txt', 'r', encoding='utf-8') as file:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = file.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.strip()  # 去掉读取每一行数据的\n
            stop_words.add(i)

    # 数据预处理
    # 删除诊断词里的停用词并tokenize句子
    pre_train_text = [pre.delete_stop_words(item, stop_words) for item in nonP_train_text]
    pre_test_text = [pre.delete_stop_words(item, stop_words) for item in nonP_test_text]

    # 删除标准词里的标点符号并tokenize句子
    pre_train_result = [pre.delete_punctuations(item) for item in nonP_train_result_list]

    # 导入模型
    my_model = Word2Vec.load("model/train.model")
    print("successfully loading model")
    print(my_model)
    print(f"window:{my_model.window}, epochs:{my_model.epochs}")

    # 计算诊断词的句子向量
    train_text_vector = model.count_sentence_vector(pre_train_text, my_model)
    test_text_vector = model.count_sentence_vector(pre_test_text, my_model)

    # 计算标准词的句子向量
    train_result_vector = model.count_sentence_vector(pre_train_result, my_model)
    print("successfully creating sentence vectors")

    # 为了和之前创建的函数规范化，创建值为unknown的列表
    test_result = []
    for i in range(len(nonP_test_text)):
        test_result.append("unknown")

    predictions = model.sentence_cosine_similarity(test_text_vector, train_result_vector, nonP_train_result_list,
                                                   nonP_test_text, test_result, percentage=10, distance=0, max_ans=2)

    save_list = []
    for data, prediction in zip(nonP_test_text, predictions):
        save_list.append({"text": data, "normalized_result": prediction})

    # 保存预测答案
    with open('test_answer.json', 'w', encoding='utf-8') as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)
