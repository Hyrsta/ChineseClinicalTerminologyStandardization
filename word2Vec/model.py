# _*_ coding=utf-8 _*_
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# 训练WV模型函数
def w2v_model(sentences):
    model = Word2Vec(sentences=sentences, vector_size=300, window=8, min_count=1, workers=8, epochs=2)
    return model


# 计算句子的句子向量的函数
def count_sentence_vector(tokenized_sentences, my_model):
    preprocessed_text = [" ".join(tokens) for tokens in tokenized_sentences]
    # 创建并拟合TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_text)

    # 将模型里的所有词放入vocabulary变量里
    vocabulary = my_model.wv.index_to_key

    # 获取TF-IDF vectorizer里面的所有单词
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out().tolist()

    weighted_sentence_vectors = []
    for index, sentence in enumerate(tokenized_sentences):
        weighted_vector = np.zeros(my_model.vector_size)     # 存储句子向量
        total_word_used = 0   # 存储用于计算句子向量的词数量
        for word in sentence:
            if word in vocabulary and word in tfidf_feature_names:
                word_vector = my_model.wv[word]     # 获该词的取词向量
                tfidf_score = tfidf_matrix[index, tfidf_feature_names.index(word)]  # 获该词的取tf-idf值
                weighted_vector += (word_vector * tfidf_score)  # 让词向量用tf-idf加权
                total_word_used += tfidf_score
        if total_word_used != 0:
            weighted_vector /= total_word_used  # 计算词向量的平均
        weighted_sentence_vectors.append(weighted_vector)
    return weighted_sentence_vectors


# 按照percentage，distance和max_ans的指标，由相似度查找待查寻语句的标准词
def most_similar_sentence(similarities, result_text_split, *, percentage, distance, max_ans):
    # 匹配每一个相似度与其对应的标准词成元组，用列表保存这些元组
    similarity_sentence_pair = list(zip(similarities, result_text_split))

    # 根据相似度大小进行升序排序
    sorted_value = sorted(similarity_sentence_pair, reverse=True, key=lambda x: x[0])

    highest_similarity = sorted_value[0][0]         # 保存相似度的最大值
    similar_sentence_list = [sorted_value[0][1]]    # 保存相似度满足要求的标准词
    difference_percentage_list = [0.0]              # 保存与最大相似度的百分比差异
    length = 1                                      # 保存相似度满足要求的标准词的数量
    for i in range(1, max_ans):
        # 计算该相似度与最大相似度的百分比差异
        difference_percentage = (highest_similarity - sorted_value[i][0]) / highest_similarity * 100

        # 当百分比小于指定的百分数且百分比减去之前的百分比大于指定的距离
        if difference_percentage <= percentage:
            difference_percentage_list.append(difference_percentage)
            if difference_percentage - difference_percentage_list[-1] >= distance:
                # 将该相似度对应的标准词加大列表
                similar_sentence_list.append(sorted_value[i][1])
                length += 1
            else:
                continue
        else:
            break

    print("answer count: {}".format(length))
    similar_sentence = '##'.join(map(str, similar_sentence_list))

    # 查看最大相似度与下五个相似度的百分差异（为了调参用）
    for j in range(5):
        difference_percentage = (highest_similarity - sorted_value[j][0]) / highest_similarity * 100
        difference_percentage_list.append(difference_percentage)

    print(sorted_value[:5])
    print(difference_percentage_list)
    return similar_sentence


# 基于余弦相似度计算预测答案的函数
def sentence_cosine_similarity(query_vector_list, train_vector_list, train_list, query_list, answer_list, *,
                               percentage, distance, max_ans):
    predictions = []    # 存储预测答案的列表
    for index, query_vector in enumerate(query_vector_list):
        query = query_list[index]               # 待查询语句
        correct_answer = answer_list[index]     # 待查询语句的正确答案
        if np.any(query_vector != 0):
            # 计算余弦相似度
            similarities = [cosine_similarity([query_vector], [vector])[0][0] for vector in train_vector_list]

            # 计算与待查寻语句相似度最高的标准词
            predicted_answer = most_similar_sentence(similarities, train_list, percentage=percentage,
                                                     distance=distance, max_ans=max_ans)
        else:
            predicted_answer = 'no_similar_sentence'    # 'no_similar_sentence'表示没有预测答案
        predictions.append(predicted_answer)

        # 输出待查询语句，预测结果，以及正确结果
        print("query{}\n待查询语句：{}\n预测结果：{}\n正确结果:{}\n".format(index, query, predicted_answer, correct_answer))
    return predictions
