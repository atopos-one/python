import pickle
from collections import Counter


# 定义一个函数，用于加载pickle文件
def load_pickle(filename):
    # 打开并读取指定的pickle文件
    with open(filename, 'rb') as f:
        # 使用pickle库的load方法加载文件内容，并指定编码格式为'iso-8859-1'
        data = pickle.load(f, encoding='iso-8859-1')
    # 返回加载得到的数据
    return data

# 定义一个函数，用于将数据根据qid分割成单个和多个的两部分
def split_data(total_data, qids):
    # 使用Counter统计qid的出现次数
    result = Counter(qids)
    # 初始化两个列表，用于存储单个和多个的数据
    total_data_single = []
    total_data_multiple = []
    # 遍历总数据
    for data in total_data:
        # 如果某个数据的qid在结果中的出现次数为1，那么将其添加到单个数据的列表中
        if result[data[0][0]] == 1:
            total_data_single.append(data)
        else:
            # 否则，将其添加到多个数据的列表中
            total_data_multiple.append(data)
    # 返回两个列表
    return total_data_single, total_data_multiple

# 定义一个函数，用于处理数据并将结果保存到指定的路径
def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    # 打开并读取指定的文件
    with open(filepath, 'r') as f:
        total_data = eval(f.read())
    # 提取每个数据的qid
    qids = [data[0][0] for data in total_data]
    # 使用split_data函数将数据分割为单个和多个的两部分
    total_data_single, total_data_multiple = split_data(total_data, qids)

    # 将单个数据的列表保存到指定的路径
    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))
    # 将多个数据的列表保存到指定的路径
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))


# 定义一个函数，用于处理大量数据，并将结果保存到指定的路径
def data_large_processing(filepath, save_single_path, save_multiple_path):
    # 从指定的文件路径加载数据
    total_data = load_pickle(filepath)
    # 提取每个数据的qid
    qids = [data[0][0] for data in total_data]
    # 使用split_data函数将数据分割为单个和多个的两部分
    total_data_single, total_data_multiple = split_data(total_data, qids)

    # 将单个数据的列表保存到指定的路径
    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    # 将多个数据的列表保存到指定的路径
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)

# 定义一个函数，用于将未标记的单个数据转换为已标记的数据，并将结果保存到指定的路径
def single_unlabeled_to_labeled(input_path, output_path):
    # 从指定的文件路径加载数据
    total_data = load_pickle(input_path)
    # 为每个数据添加标签1
    labels = [[data[0], 1] for data in total_data]
    # 按照qid和标签对数据进行排序
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))
    # 将排序后的数据保存到指定的路径
    with open(output_path, "w") as f:
        f.write(str(total_data_sort))


if __name__ == "__main__":
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)
