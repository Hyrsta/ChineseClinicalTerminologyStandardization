import model
import openpyxl
import pre


# 文件的开始
if __name__ == '__main__':
    # 读取停止词文件
    with open('dataset/stop_words.txt', 'r', encoding='utf-8') as file:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = file.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.strip()  # 去掉读取每一行数据的\n
            stop_words.add(i)

    # 导入xlsx文件里的标准病毒
    standard_path = "dataset/鍥介檯鐤剧梾鍒嗙被 ICD-10鍖椾含涓村簥鐗坴601.xlsx"
    xlsx_file = openpyxl.load_workbook(standard_path)
    nonP_standard = []
    for index, row in enumerate(xlsx_file.worksheets[0]):
        if row[1].value not in nonP_standard:
            nonP_standard.append(row[1].value)
    print("successfully load", standard_path)

    # 导入数据集
    train_path = 'dataset/train.json'
    nonP_train_text, nonP_train_result, nonP_train_result_list = pre.dataset_to_list(train_path)
    print("successfully load", train_path)
    dev_path = 'dataset/dev.json'
    nonP_dev_text, nonP_dev_result, nonP_dev_result_list = pre.dataset_to_list(dev_path)
    print("successfully load", dev_path)

    # 数据预处理
    # 删除标准词里的标点符号并分词
    pre_train_result = [pre.delete_punctuations(item) for item in nonP_train_result_list]
    pre_dev_result = [pre.delete_punctuations(item) for item in nonP_dev_result_list]
    pre_standard = [pre.delete_punctuations(item) for item in nonP_standard]

    # 建立Word2vec模型
    train_model = model.w2v_model(pre_train_result)
    print("successfully created", train_model, "based on", train_path)
    dev_model = model.w2v_model(pre_dev_result)
    print("successfully created", dev_model, "based on", dev_path)
    standard_model = model.w2v_model(pre_standard)
    print("successfully created", standard_model, "based on", standard_path)

    # 保存模型
    train_save_path = "model/train.model"
    dev_save_path = "model/dev.model"
    standard_save_path = "model/standard.model"

    train_model.save(train_save_path)
    print(train_model, "successfully saved to path", train_save_path)
    dev_model.save(dev_save_path)
    print(dev_model, "successfully saved to path", dev_save_path)
    standard_model.save(standard_save_path)
    print(standard_model, "successfully saved to path", standard_save_path)
