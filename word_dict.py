import pickle


# 定义一个函数，用于获取两个语料库的词汇表
def get_vocab(corpus1, corpus2):
    # 初始化一个集合，用于存储词汇
    word_vocab = set()
    # 遍历两个语料库
    for corpus in [corpus1, corpus2]:
        # 遍历语料库中的每个元素
        for i in range(len(corpus)):
            # 更新词汇集合，将元素中的词汇添加到集合中
            word_vocab.update(corpus[i][1][0])
            word_vocab.update(corpus[i][1][1])
            word_vocab.update(corpus[i][2][0])
            word_vocab.update(corpus[i][3])
    # 打印词汇集合的大小
    print(len(word_vocab))
    # 返回词汇集合
    return word_vocab

# 定义一个函数，用于加载pickle文件
def load_pickle(filename):
    # 打开并读取指定的pickle文件
    with open(filename, 'rb') as f:
        # 使用pickle库的load方法加载文件内容
        data = pickle.load(f)
    # 返回加载得到的数据
    return data


# 定义一个函数，用于处理两个文件中的词汇表，并保存结果到指定的路径
def vocab_processing(filepath1, filepath2, save_path):
    # 打开并读取指定的文件1，将文件内容转换为集合
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))
    # 打开并读取指定的文件2，将文件内容转换为列表
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())

    # 使用get_vocab函数获取两个文件中的词汇表
    word_set = get_vocab(total_data2, total_data2)

    # 获取两个词汇表的交集，即在两个文件中都出现过的词汇
    excluded_words = total_data1.intersection(word_set)
    # 从词汇表中移除交集中的词汇，得到只在一个文件中出现过的词汇
    word_set = word_set - excluded_words

    # 打印两个词汇表的大小
    print(len(total_data1))
    print(len(word_set))

    # 将处理后的词汇表保存到指定的路径
    with open(save_path, 'w') as f:
        f.write(str(word_set))


if __name__ == "__main__":
    python_hnn = './data/python_hnn_data_teacher.txt'
    python_staqc = './data/staqc/python_staqc_data.txt'
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = './data/sql_hnn_data_teacher.txt'
    sql_staqc = './data/staqc/sql_staqc_data.txt'
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'

    final_vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)
