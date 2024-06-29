# python
软件工程完美编程

embddings_process.py

     将词向量文件保存为二进制文件:trans_bin()
   
     构建新的词典和词向量矩阵:get_new_dict()
   
     得到词在词典中的位置:get_index()
   
     将训练、测试、验证语料序列化:serialization()

getStru2Vec.py
   
     处理Python查询语句的多进程任务:multipro_python_query()
   
     处理Python代码的多进程任务:multipro_python_code()
   
     处理Python上下文的多进程任务:multipro_python_context()
   
     处理SQL查询语句的多进程任务:multipro_sqlang_query()
   
     处理SQL代码的多进程任务:multipro_sqlang_code()
   
     处理SQL查询语句的上下文多进程任务:multipro_sqlang_context()
   
     处理多进程任务，包括上下文、查询和代码的解析:parse()

process_single.py
   
     加载pickle文件:load_pickle()
   
     将数据根据qid分割成单个和多个的两部分:split_data()
   
     处理数据并将结果保存到指定的路径:data_staqc_processing()
   
     处理大量数据，并将结果保存到指定的路径:data_large_processing()
   
     将未标记的单个数据转换为已标记的数据，并将结果保存到指定的路径:single_unlabeled_to_labeled()

word_dict.py

     获取两个语料库的词汇表:get_vocab()
   
     加载pickle文件:load_pickle()
    
     处理两个文件中的词汇表，并保存结果到指定的路径:vocab_processing()
