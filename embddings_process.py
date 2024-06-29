import pickle
import numpy as np
from gensim.models import KeyedVectors


# 将词向量文件保存为二进制文件
def trans_bin(path1, path2):
    # 从文本文件加载词向量
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    # 计算词向量的L2范数，并替换原始词向量
    wv_from_text.init_sims(replace=True)
    # 保存词向量为二进制文件
    wv_from_text.save(path2)

# 构建新的词典和词向量矩阵
def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    # 使用Gensim的KeyedVectors模块加载指定路径的词向量文件
    model = KeyedVectors.load(type_vec_path, mmap='r')
    # 打开并读取包含全部单词的文件
    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())

    # 初始化词典，包含四个特殊词汇：PAD，SOS，EOS，UNK
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 其中0 PAD_ID, 1 SOS_ID, 2 EOS_ID, 3 UNK_ID
    # 初始化无法找到词向量的单词列表
    fail_word = []
    # 创建一个随机状态
    rng = np.random.RandomState(None)
    # 创建特殊词汇的词向量
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    # 初始化词向量列表
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]
   
    # 对所有单词进行遍历
    for word in total_word:
        try:
            # 尝试从模型中获取该单词的词向量，并添加到词向量列表中
            word_vectors.append(model.wv[word])  
            # 将单词添加到词典中
            word_dict.append(word)
        except:
            # 如果无法从模型中获取该单词的词向量，则将该单词添加到失败列表中
            fail_word.append(word)

    # 将词向量列表转换为numpy数组，方便后续的计算和操作
    word_vectors = np.array(word_vectors)

    # 将词典中的单词和其在词典中的索引进行反向映射，形成新的词典
    # 这样做的目的是为了后续能够通过词汇快速查找其在词典中的位置
    word_dict = dict(map(reversed, enumerate(word_dict)))

    # 将词向量矩阵保存到指定的路径，以备后续使用
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    # 将词典保存到指定的路径，以备后续使用
    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

print("完成") 


# 得到词在词典中的位置
def get_index(type, text, word_dict):
    # 初始化一个空列表，用于存储结果
    location = []
    # 判断文本类型
    if type == 'code':
        # 如果类型是'code'，则首先在列表中添加一个元素1
        location.append(1)
        # 获取输入文本的长度
        len_c = len(text)
        # 判断文本长度是否小于349
        if len_c + 1 < 350:
            # 如果长度为1且文本的第一个字符为'-1000'，则在列表中添加一个元素2
            if len_c == 1 and text[0] == '-1000':
                location.append(2)
            else:
                # 否则，遍历文本中的每个字符
                for i in range(0, len_c):
                    # 获取每个字符在词典中的索引，如果字符不在词典中，则返回'UNK'的索引
                    index = word_dict.get(text[i], word_dict['UNK'])
                    # 将索引添加到列表中
                    location.append(index)
                # 在列表末尾添加一个元素2
                location.append(2)
        else:
            # 如果文本长度大于等于349，只获取前348个字符的索引
            for i in range(0, 348):
                index = word_dict.get(text[i], word_dict['UNK'])
                location.append(index)
            # 在列表末尾添加一个元素2
            location.append(2)
    else:
        # 如果类型不是'code'，则根据文本的内容决定添加什么元素
        if len(text) == 0:
            # 如果文本为空，则添加一个元素0
            location.append(0)
        elif text[0] == '-10000':
            # 如果文本的第一个字符为'-10000'，则添加一个元素0
            location.append(0)
        else:
            # 否则，获取文本中每个字符在词典中的索引
            for i in range(0, len(text)):
                index = word_dict.get(text[i], word_dict['UNK'])
                location.append(index)

    return location


# 将训练、测试、验证语料序列化
# 查询：25 上下文：100 代码：350
def serialization(word_dict_path, type_path, final_type_path):
    # 打开并加载词典文件
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    # 打开并读取语料文件
    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    # 初始化一个列表，用于存储序列化后的数据
    total_data = []

    # 遍历语料中的每一条数据
    for i in range(len(corpus)):
        # 提取数据中的各个部分
        qid = corpus[i][0]
        Si_word_list = get_index('text', corpus[i][1][0], word_dict)
        Si1_word_list = get_index('text', corpus[i][1][1], word_dict)
        tokenized_code = get_index('code', corpus[i][2][0], word_dict)
        query_word_list = get_index('text', corpus[i][3], word_dict)
        block_length = 4
        label = 0

        # 对每一部分的长度进行限制，并填充不足的部分
        Si_word_list = Si_word_list[:100] if len(Si_word_list) > 100 else Si_word_list + [0] * (100 - len(Si_word_list))
        Si1_word_list = Si1_word_list[:100] if len(Si1_word_list) > 100 else Si1_word_list + [0] * (100 - len(Si1_word_list))
        tokenized_code = tokenized_code[:350] + [0] * (350 - len(tokenized_code))
        query_word_list = query_word_list[:25] if len(query_word_list) > 25 else query_word_list + [0] * (25 - len(query_word_list))

        # 将处理后的数据添加到列表中
        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    # 将序列化后的数据保存到文件中
    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)


if __name__ == '__main__':
    # 词向量文件路径
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    # ==========================最初基于Staqc的词典和词向量==========================

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
    # get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================

    # sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sqlfinal_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    # get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path,sql_final_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    # Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    # Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)

    # python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    # python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    # get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path,python_final_word_dict_path)

    # 处理成打标签的形式
    staqc_python_f = '../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    # Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')
    # test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)
