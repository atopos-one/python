import pickle
import multiprocessing
from python_structured import *
from sqlang_structured import *

# 定义一个函数，用于处理Python查询语句的多进程任务
def multipro_python_query(data_list):
    # 使用列表推导式，对数据列表中的每一条查询语句进行解析
    return [python_query_parse(line) for line in data_list]

# 定义一个函数，用于处理Python代码的多进程任务
def multipro_python_code(data_list):
    # 使用列表推导式，对数据列表中的每一段代码进行解析
    return [python_code_parse(line) for line in data_list]

# 定义一个函数，用于处理Python上下文的多进程任务
def multipro_python_context(data_list):
    # 初始化一个空列表，用于存储结果
    result = []
    # 遍历数据列表
    for line in data_list:
        # 如果数据为'-10000'，则将其添加到结果列表中
        if line == '-10000':
            result.append(['-10000'])
        else:
            # 否则，对数据进行解析，并将解析结果添加到结果列表中
            result.append(python_context_parse(line))
    # 返回结果列表
    return result

# 定义一个函数，用于处理SQL查询语句的多进程任务
def multipro_sqlang_query(data_list):
    # 使用列表推导式，对数据列表中的每一条查询语句进行解析
    return [sqlang_query_parse(line) for line in data_list]

# 定义一个函数，用于处理SQL代码的多进程任务
def multipro_sqlang_code(data_list):
    # 使用列表推导式，对数据列表中的每一段代码进行解析
    return [sqlang_code_parse(line) for line in data_list]

# 定义一个函数，用于处理SQL查询语句的上下文多进程任务
def multipro_sqlang_context(data_list):
    # 初始化一个空列表，用于存储结果
    result = []
    # 遍历数据列表
    for line in data_list:
        # 如果数据为'-10000'，则将其添加到结果列表中
        if line == '-10000':
            result.append(['-10000'])
        else:
            # 否则，对数据进行解析，并将解析结果添加到结果列表中
            result.append(sqlang_context_parse(line))
    # 返回结果列表
    return result

# 定义一个函数，用于处理多进程任务，包括上下文、查询和代码的解析
def parse(data_list, split_num, context_func, query_func, code_func):
    # 创建一个多进程池
    pool = multiprocessing.Pool()
    # 将数据列表分割为多个子列表，每个子列表的长度为split_num
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
    # 使用多进程对每个子列表进行上下文解析
    results = pool.map(context_func, split_list)
    # 将结果合并为一个列表
    context_data = [item for sublist in results for item in sublist]
    # 打印上下文数据的数量

    # 对查询和代码也执行相同的操作
    results = pool.map(query_func, split_list)
    query_data = [item for sublist in results for item in sublist]
    
    results = pool.map(code_func, split_list)
    code_data = [item for sublist in results for item in sublist]
    
    # 关闭进程池并等待所有进程结束
    pool.close()
    pool.join()

    # 返回解析后的上下文、查询和代码数据
    return context_data, query_data, code_data

# 定义一个主函数，用于处理并保存数据
def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    # 打开并加载原始数据文件
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    # 解析原始数据
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)
    # 提取每个数据的qid
    qids = [item[0] for item in corpus_lis]

    # 将qid、上下文、查询和代码组合为一个新的数据列表
    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    # 将新的数据列表保存到指定的路径
    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)

if __name__ == '__main__':
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)

    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)
