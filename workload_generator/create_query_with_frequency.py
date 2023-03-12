import numpy as np
import random

MIN_KEY = 1  # Smallest key
# NUM_KEYS = 1000  # Total number of keys
NUM_KEYS = 1000000  # Total number of keys
max_key = MIN_KEY + NUM_KEYS - 1

component_size = 2000
query_length = 4000
range_query_size = 40000

low_hot = 10000
high_hot = 200000

low_cold = 300000
high_cold = 800000

def create_seed_query(hot_size, cold_size):

    seed_size = hot_size + cold_size

    hot_min_keys = set()
    cold_min_keys = set()

    # uniform_hot_min_key_list = np.random.randint(low, high, seed_size)

    # create hot min key and cold min key
    # idx = 0
    while (len(hot_min_keys) < hot_size) :
        min_k = np.random.randint(low_hot, high_hot, 1)
        min_key = min_k[0]
        if (min_key + query_length - 1) >= max_key:
            min_key = min_key - query_length
        if min_key not in hot_min_keys:
            hot_min_keys.add(min_key)
            # idx += 1

    while (len(cold_min_keys) < cold_size):
        min_k = np.random.randint(low_cold, high_cold, 1)
        min_key = min_k[0]
        if (min_key + query_length - 1) >= max_key:
            min_key = min_key - query_length
        if min_key not in hot_min_keys and min_key not in cold_min_keys:
            cold_min_keys.add(min_key)
            # idx += 1

    # query = "0" + "," + str(query_length) + "-" + str(min_q) + "-" + str(max_q)
    # query_set_1.append(query)

    # create seed query based on min key

    hot_query_seed = []
    cold_query_seed = []

    for min_q in hot_min_keys:
        max_q = min_q + query_length - 1
        query = "0" + "," + str(query_length) + "-" + str(min_q) + "-" + str(max_q)
        hot_query_seed.append(query)

    for min_q in cold_min_keys:
        max_q = min_q + query_length - 1
        query = "0" + "," + str(query_length) + "-" + str(min_q) + "-" + str(max_q)
        cold_query_seed.append(query)

    return hot_query_seed, cold_query_seed

# hot_size: # of hot query in total, e.g. 20K query, exp-1: 20 hot query
# cold_size: # of cold query in total, e.g. 20K query, exp-1: 20*500 cold query
# hot_frequency: hot query frequency in window, e.g. exp-1, window = 100, hot frequency = 500
# hot_num_in_window: # of hot query in current window, e.g. exp-1: 1 hot query
# cold_num_in_window: # of cold query in current window, e.g. exp-1: 500 cold query

def create_query_in_window(hot_size, cold_size, hot_frequency, hot_num_in_window, cold_num_in_window, num_window=20):

    query_set = []

    hot_query_seed, cold_query_seed = create_seed_query(hot_size, cold_size)

    hot_query_idx = 0
    cold_query_idx = 0

    current_window = 0

    while current_window < num_window:
        queries = []
        current_hot_num_in_window = 0
        while current_hot_num_in_window < hot_num_in_window:
            hot_query = hot_query_seed[hot_query_idx]
            cnt = 0
            while cnt < hot_frequency:
                queries.append(hot_query)
                cnt += 1
            hot_query_idx += 1
            current_hot_num_in_window += 1

        current_cold_num_in_window = 0
        while current_cold_num_in_window < cold_num_in_window:
            cold_query = cold_query_seed[cold_query_idx]
            queries.append(cold_query)
            cold_query_idx += 1
            current_cold_num_in_window += 1

        random.shuffle(queries)
        query_set += queries
        current_window += 1
        print("current_window = ", current_window)

    return query_set

# hot_size: # of hot query in total, e.g. 20K query, exp-3: 20*30 hot query
# cold_size: # of cold query in total, e.g. 20K query, exp-3: 20*100 cold query
# hot_frequency_1: hot query frequency in window, e.g. exp-3, window = 1, hot frequency_1 = 6
# hot_frequency_2: hot query frequency in window, e.g. exp-3, window = 2, hot frequency_1 = 24
# hot_num_in_window_1: # of hot query in current window, e.g. exp-3: 30 hot query
# cold_num_in_window_1: # of cold query in current window, e.g. exp-3: 20 cold query
# cold_num_in_window_2: # of cold query in current window, e.g. exp-3: 80 cold query
def create_query_in_two_window(hot_size, cold_size, hot_frequency_1, hot_num_in_window_1, cold_num_in_window_1, hot_frequency_2, cold_num_in_window_2, num_window=20):

    query_set = []

    # create hot_seed and cold_seed
    hot_query_seed, cold_query_seed = create_seed_query(hot_size, cold_size)

    hot_query_idx = 0
    cold_query_idx = 0

    current_window = 0

    # repeat num_window times
    while current_window < num_window:
        queries = []

        query_window_1 = []
        current_hot_num_in_window = 0
        # fill hot query in window_1 (hot_num_in_window_1, each frequency = hot_frequency_1)
        while current_hot_num_in_window < hot_num_in_window_1:
            hot_query = hot_query_seed[hot_query_idx]
            cnt = 0
            while cnt < hot_frequency_1:
                query_window_1.append(hot_query)
                cnt += 1
            hot_query_idx += 1
            current_hot_num_in_window += 1
        # fill cold query in window_1 (cold_num_in_window_1)
        current_cold_num_in_window = 0
        while current_cold_num_in_window < cold_num_in_window_1:
            cold_query = cold_query_seed[cold_query_idx]
            query_window_1.append(cold_query)
            cold_query_idx += 1
            current_cold_num_in_window += 1
        # shuffle window_1
        random.shuffle(query_window_1)
        queries += query_window_1

        query_window_2 = []
        # fill same hot query in window_2 (same set of hot query, each frequency = hot_frequency_2)
        hot_query_idx -= hot_num_in_window_1
        current_hot_num_in_window = 0
        while current_hot_num_in_window < hot_num_in_window_1:
            hot_query = hot_query_seed[hot_query_idx]
            cnt = 0
            while cnt < hot_frequency_2:
                query_window_2.append(hot_query)
                cnt += 1
            hot_query_idx += 1
            current_hot_num_in_window += 1
        # fill cold query in window_2 (cold_num_in_window_2)
        current_cold_num_in_window = 0
        while current_cold_num_in_window < cold_num_in_window_2:
            cold_query = cold_query_seed[cold_query_idx]
            query_window_2.append(cold_query)
            cold_query_idx += 1
            current_cold_num_in_window += 1
        # shuffle window_2
        random.shuffle(query_window_2)
        queries += query_window_2

        query_set += queries
        current_window += 1
        print("current_window = ", current_window)

    return query_set



def get_write_records():
    file_name = "../data/random_data_" + str(NUM_KEYS) + ".txt"
    f = open(file_name, 'r')
    line = f.readline()
    data_list = []
    while line != "":
        line = line.replace('\n', '')
        data_list.append(int(line))
        line = f.readline()
    f.close()
    # write last 10K records
    num_write = component_size * 5
    write_records = data_list[-num_write:]
    return write_records

def load_range_queries(query_file_name):
    queries = []
    f = open(query_file_name, 'r')
    line = f.readline()
    while line:
        tem = line.rstrip("\n").split(",")
        query = tem[1]
        queries.append(query)
        line = f.readline()
    f.close()
    return queries

def create_query_with_write(query_file_name):
    write_records = get_write_records()
    query_set = load_range_queries(query_file_name)

    content_list = []

    read_operation_list = [0] * 20000
    write_operation_list = [1] * 10000
    operation_list = read_operation_list + write_operation_list
    random.shuffle(operation_list)

    idx_query = 0
    idx_write = 0

    for i in range(len(operation_list)):
        operation = operation_list[i]
        if operation == 0:
            query = query_set[idx_query]
            idx_query += 1
            operation_content = "0:" + query
        else:
            record = write_records[idx_write]
            idx_write += 1
            operation_content = "1:" + str(record)
        content_list.append(operation_content)

    return content_list

def analysis_queries(query_set):

    query_dic = dict()

    for query in query_set:
        if query not in query_dic:
            query_dic[query] = 1
        else:
            query_dic[query] += 1

    # print(query_dic)
    for q, q_f in query_dic.items():
        # print(q, " = ", q_f)
        if q_f == 30:
            print(q, " = ", q_f)

def write_query(file_name, query_set):
    f = open(file_name, 'w')
    for q in query_set:
        f.write(q)
        f.write('\n')
    f.close()

# exp-1: 1000, hot: 1 - f500, cold: 500 - f1
# 40K, hot: 40, cold: 40*500
# hot_size: # of hot query in total, e.g. 20K query, exp-1: 20 hot query
# cold_size: # of cold query in total, e.g. 20K query, exp-1: 20*500 cold query
# hot_frequency: hot query frequency in window, e.g. exp-1, window = 100, hot frequency = 500
# hot_num_in_window: # of hot query in current window, e.g. exp-1: 1 hot query
# cold_num_in_window: # of cold query in current window, e.g. exp-1: 500 cold query

# hot_size, cold_size, hot_frequency, hot_num_in_window, cold_num_in_window, num_window=40):
# query_set_1 = create_query_in_window(20, 10000, 500, 1, 500)
# file_name_1 = "../data/query_hot_1.txt"
# query_set_1 = load_range_queries(file_name_1)
# analysis_queries(query_set_1)
# write_query(file_name_1, query_set_1)

# file_name_2 = "../data/query_hot_2.txt"
# query_set_2 = create_query_with_write(file_name_1)
# write_query(file_name_2, query_set_2)

# exp-3: 1000, hot: 30 - f30, cold: 100 - f1
# 20K, hot: 20*30, cold: 20*100
# hot_size: # of hot query in total, e.g. 20K query, exp-3: 20*30 hot query
# cold_size: # of cold query in total, e.g. 20K query, exp-3: 20*100 cold query
# hot_frequency_1: hot query frequency in window, e.g. exp-3, window = 1, hot frequency_1 = 6
# hot_frequency_2: hot query frequency in window, e.g. exp-3, window = 2, hot frequency_1 = 24
# hot_num_in_window_1: # of hot query in current window, e.g. exp-3: 30 hot query
# cold_num_in_window_1: # of cold query in current window, e.g. exp-3: 20 cold query
# cold_num_in_window_2: # of cold query in current window, e.g. exp-3: 80 cold query

# create_query_in_two_window(hot_size, cold_size, hot_frequency_1, hot_num_in_window_1, cold_num_in_window_1, hot_frequency_2, cold_num_in_window_2, num_window=20):
query_set_3 = create_query_in_two_window(600, 2000, 6, 30, 20, 24, 80)
file_name_3 = "../data/query_hot_3.txt"
write_query(file_name_3, query_set_3)
# analysis_queries(query_set_3)

file_name_4 = "../data/query_hot_4.txt"
query_set_4 = create_query_with_write(file_name_3)
write_query(file_name_4, query_set_4)




