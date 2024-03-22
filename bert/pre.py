# _*_ coding=utf-8 _*_
import re  # 正则表达式
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import jieba 
import string
from torch.nn.utils.rnn import pad_sequence
from nltk import edit_distance
import numpy as np
import torch


# 定义Dataset类，以便dataloader处理
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(
        self, idx
    ):  # 给DataLoader最后的处理参数，对每个idx进行tokenizer处理，返回包含input_ids与attention_mask的字典
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

# collnate_fn函数，用于填充张量
def collate_fn(batch):
    # 获取输入的batch字典中的input_ids，生成列表,获取最长的tensor段
    padded_input_ids = pad_sequence(
        [item["input_ids"].squeeze(dim=0) for item in batch],
        batch_first=True,
        padding_value=0,
    )
    return {"input_ids": padded_input_ids}

# 数据规范化函数
def remove_non_chinese(input_string):
    # 使用正则表达式匹配非汉字字符，并替换为空格
    chinese_only = re.sub("[^\u4e00-\u9fa5]", "", input_string)
    return chinese_only

def delete_punctuations(input_string):
    result = []
    for word in jieba.lcut(input_string):
        if word not in string.punctuation:
            result.append(word)
    return result


# 训练WV模型函数
def wv_model(text):
    model = Word2Vec(sentences=text, vector_size=300, window=10, min_count=1, workers=4)
    return model


# bert_embedding模型
def bert_embedding(batch_input, model):
    outputs = model(**batch_input)
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # 将张量移回CPU以便与其他操作兼容
    return embeddings.cpu().detach()


# 计算Jaccard系数
def jaccard_similarity(array1, array2):
    set1 = set(array1.flatten())
    set2 = set(array2.flatten())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


# 计算编辑距离
def edit_distence(st1, st2):
    edit_dist = edit_distance(st1, st2)
    return edit_dist


# test
if __name__ == "__main__":
    input = ["右肺"]
    query = ["右肺,结节转移" '"可能大"', "肺占位性病变", "肺继发恶性肿瘤"]
    st = []
    for item in query:
        temp = list(remove_non_chinese(item))
        st.append("".join(temp))
    print(st)
    
    model_path = "/home/nekozo/VSCode/model/bert-base-chinese/"
    tokenizer = BertTokenizer.from_pretrained(model_path, num_threads=8)
    model = BertModel.from_pretrained(model_path)
    
    embedded_query = []
    for item in st:
        item_list = [item]
        embedded_item = []
        dataset = TextDataset(item_list, tokenizer, max_length=128)
        Dataloader = DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
        )
        for batch in Dataloader:
            embedded_item.append(bert_embedding(batch, model))
        embedded_query.append(torch.cat(embedded_item, dim=0))
    print(embedded_query)