import random

import numpy as np

MIN_KEY = 1  # Smallest key
NUM_KEYS = 1000000  # Total number of keys
max_key = MIN_KEY + NUM_KEYS - 1
component_size = 2000
# min_query_length = 3000
# max_query_length = 4000
query_length = 6000
total_num_component = NUM_KEYS / component_size
num_write_component = 5
num_write = num_write_component * component_size

def build_uniform_query(low, high, current_data_list):
    # current_data_list = sorted(current_data_list)
    min_q = int(np.random.randint(low, high, 1))
    # while (min_q + query_length - 1) > max_record:
    #     min_q = int(np.random.randint(low, high, 1))
    max_q = min_q + query_length - 1
    query = str(query_length) + "-" + str(min_q) + "-" + str(max_q)
    return query, current_data_list

def build_hot_Possion_query(lam, max_record, current_data_list):
    # if time_independent:
    #     current_data_list = sorted(current_data_list)

    # q_length = random.randint(min_query_length, max_query_length)
    min_q = np.random.poisson(lam, 1)
    min_q = int(min_q)
    max_q = min_q + query_length - 1
    while max_q > max_record:
        min_q = np.random.poisson(lam, 1)
        min_q = int(min_q)
        max_q = min_q + query_length - 1
    query = str(query_length) + "-" + str(min_q) + "-" + str(max_q)
        # print("from if", query)

    # print("current_list = ", current_data_list)
    # print("query = ", query)
    return query, current_data_list

def build_Zipf_query(parameter, max_record, current_data_list):
    # parameter is ZIPF_ALPHA, if Zipf distribution

    min_q = np.random.zipf(parameter, 1)
    min_q = int(min_q)
    while (min_q + query_length - 1) > max_record:
        min_q = np.random.zipf(parameter, 1)
        min_q = int(min_q)
    max_q = min_q + query_length - 1
    query = str(query_length) + "-" + str(min_q) + "-" + str(max_q)
    return query, current_data_list

def build_query_with_ratio73():

    # 23333
    # hot: 70 = 16333, cold, 30 = 7000
    num_hot = 16333
    num_cold = 7000

    query_set = []
    q_length = 260
    # hot part
    query_set_hot = []
    uniform_list_hot = np.random.randint(10500, 110500, 7000)
    for min_q in uniform_list_hot:
        min_q = int(min_q)
        while (min_q + q_length - 1) >= max_key:
            min_q_l = np.random.randint(10500, 110500, 1)
            min_q = int(min_q_l[0])
        max_q = min_q + q_length - 1
        query = str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_hot.append(query)
    idx = 0
    while (len(query_set) < num_hot):
        query_set.append(query_set_hot[idx])
        idx += 1
        if idx == len(query_set_hot):
            idx = 0

    # cold part
    query_set_cold = []
    # for i in range(200000, 990000):
    min_q = 200000
    while len(query_set_cold) < 3000:
        max_q = min_q + q_length - 1
        query = str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_cold.append(query)
        min_q += q_length
    idx = 0
    while (len(query_set) < 23333):
        query_set.append(query_set_cold[idx])
        idx += 1
        if idx == len(query_set_cold):
            idx = 0

    random.shuffle(query_set)
    return query_set

def build_content_list_h7c3(read_ratio, write_ratio, write_list):

    num_read = int((num_write / write_ratio) * read_ratio)
    read_operation_list = [0] * num_read
    query_set = build_query_with_ratio73()

    write_operation_list = [1] * num_write
    operation_list = read_operation_list + write_operation_list
    random.shuffle(operation_list)
    print("length of operation list = ", len(operation_list))

    current_data_list = load_list
    current_key_length = num_load
    cnt = 0
    content_list = []
    max_record = max(load_list)

    idx_query = 0

    for i in range(len(operation_list)):
        operation = operation_list[i]
        if operation == 0:
            # read operation = 0
            # uniform
            query = query_set[idx_query]
            idx_query += 1
            # query, current_data_list = build_uniform_query(15, 45, max_record, current_data_list)
            operation_content = "0:" + query
        else:
            # write operation = 1
            new_record = write_list[cnt]
            if new_record > max_record:
                max_record = new_record
            operation_content = "1:" + str(new_record)
            current_data_list.append(new_record)
            cnt += 1
            current_key_length += 1
        content_list.append(operation_content)
        if i % 100000 == 0:
            print(i)
    return content_list

def build_query_with_37():

    num_hot = 7000
    num_cold = 16333

    # if cold: 70% = 14K, hot 30% = 6K
    # hot is 2k*3 = 6K
    # 60000-20000 choose 2K, do three times, shuffle
    # 200,000 - 990,000 is cold
    # (990,000-100,000) / 7K = 127
    # every range query size = 120

    query_set = []

    q_length = 120
    # hot part
    query_set_hot = []
    uniform_list_hot = np.random.randint(20000, 60000, 2000)
    for min_q in uniform_list_hot:
        min_q = int(min_q)
        while (min_q + q_length - 1) >= max_key:
            min_q_l = np.random.randint(20000, 60000, 1)
            min_q = int(min_q_l[0])
        max_q = min_q + q_length - 1
        query = str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_hot.append(query)
    idx = 0
    while (len(query_set) < num_hot):
        query_set.append(query_set_hot[idx])
        idx += 1
        if idx == len(query_set_hot):
            idx = 0

    # cold part
    query_set_cold = []
    # for i in range(100000, 990000):
    min_q = 100000
    while len(query_set_cold) < 7000:
        max_q = min_q + q_length - 1
        query = str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_cold.append(query)
        min_q += q_length
    idx = 0
    while (len(query_set) < 23333):
        query_set.append(query_set_cold[idx])
        idx += 1
        if idx == len(query_set_cold):
            idx = 0

    random.shuffle(query_set)
    return query_set

def build_content_list_h3c7(read_ratio, write_ratio, write_list):

    num_read = int((num_write / write_ratio) * read_ratio)
    read_operation_list = [0] * num_read
    query_set = build_query_with_37()

    write_operation_list = [1] * num_write
    operation_list = read_operation_list + write_operation_list
    random.shuffle(operation_list)
    print("length of operation list = ", len(operation_list))

    current_data_list = load_list
    current_key_length = num_load
    cnt = 0
    content_list = []
    max_record = max(load_list)

    idx_query = 0

    for i in range(len(operation_list)):
        operation = operation_list[i]
        if operation == 0:
            # read operation = 0
            # uniform
            query = query_set[idx_query]
            idx_query += 1
            # query, current_data_list = build_uniform_query(15, 45, max_record, current_data_list)
            operation_content = "0:" + query
        else:
            # write operation = 1
            new_record = write_list[cnt]
            if new_record > max_record:
                max_record = new_record
            operation_content = "1:" + str(new_record)
            current_data_list.append(new_record)
            cnt += 1
            current_key_length += 1
        content_list.append(operation_content)
        if i % 100000 == 0:
            print(i)
    return content_list

def build_content_list(read_ratio, write_ratio, write_list):

    num_read = int((num_write / write_ratio) * read_ratio)
    read_operation_list = [0] * num_read

    write_operation_list = [1] * num_write
    operation_list = read_operation_list + write_operation_list
    random.shuffle(operation_list)
    print("length of operation list = ", len(operation_list))

    current_data_list = load_list
    current_key_length = num_load
    cnt = 0
    content_list = []
    # max_record = max(load_list)

    for i in range(len(operation_list)):
        operation = operation_list[i]
        if operation == 0:
            # read operation = 0
            # 1000000
            # uniform
            # update 90500, 100500
            # no update 710000, 910000
            query, current_data_list = build_uniform_query(710000, 910000, current_data_list)
            # query, current_data_list = build_uniform_query(90500, 100500, current_data_list)
            # query, current_data_list = build_uniform_query(10500, 110500, max_record, current_data_list)
            # query, current_data_list = build_uniform_query(15, 145, current_data_list)
            # poisson
            # query, current_data_list = build_hot_Possion_query(100000, max_record, current_data_list)
            # zipf
            # query, current_data_list = build_Zipf_query(2, max_record, current_data_list)
            # query, current_data_list = build_uniform_query(100, 600, max_record, current_data_list)
            operation_content = "0:" + query
        else:
            # write operation = 1
            new_record = write_list[cnt]
            # if new_record > max_record:
            #     max_record = new_record
            operation_content = "1:" + str(new_record)
            current_data_list.append(new_record)
            cnt += 1
            current_key_length += 1
        content_list.append(operation_content)
        if i % 100000 == 0:
            print(i)
    return content_list

def build_content_list_with_seed(read_ratio, write_ratio, write_list):
    num_read = int((num_write / write_ratio) * read_ratio)
    read_operation_list = [0] * num_read

    write_operation_list = [1] * num_write
    operation_list = read_operation_list + write_operation_list
    random.shuffle(operation_list)
    print("length of operation list = ", len(operation_list))

    current_data_list = load_list
    current_key_length = num_load
    cnt = 0
    content_list = []
    max_record = max(load_list)
    seed_length = 5000
    cnt_query = 0

    for i in range(len(operation_list)):
        operation = operation_list[i]
        if operation == 0:
            # read operation = 0
            # 1000000
            cnt_query += 1
            if cnt_query < seed_length:
                query, current_data_list = build_uniform_query(10000, 600000, max_record, current_data_list)
                operation_content = "0:" + query
            else:
                idx = np.random.randint(0, len(content_list) - 1, 1)
                duplicate_content = content_list[idx[0]]
                while duplicate_content.startswith("1:"):
                    idx = np.random.randint(0, len(content_list) - 1, 1)
                    duplicate_content = content_list[idx[0]]
                operation_content = duplicate_content
        else:
            # write operation = 1
            new_record = write_list[cnt]
            if new_record > max_record:
                max_record = new_record
            operation_content = "1:" + str(new_record)
            current_data_list.append(new_record)
            cnt += 1
            current_key_length += 1
        content_list.append(operation_content)
        if i % 100000 == 0:
            print(i)
    return content_list

def write_content(file_name, content_list):
    f = open(file_name, 'w')
    for c in content_list:
        f.write(c)
        f.write('\n')
    f.close()

# file_name = "../data/random_data_" + str(NUM_KEYS) + ".txt"
file_name = "../data/random_data_" + str(NUM_KEYS) + "_with_anti_no_update.txt"
f = open(file_name, 'r')
line = f.readline()
data_list = []
while line != "":
    line = line.replace('\n', '')
    # data_list.append(int(line))
    data_list.append(line)
    line = f.readline()
f.close()

# number of load = total_num_component * component size
num_load = int((total_num_component - num_write_component) * component_size)
# print(num_load)
load_list = data_list[:num_load]
write_list = data_list[-num_write:]


# 90% read - 10% write
file_name_91 = "../data/query_9_1_5_no_update.txt"
content_list_91 = build_content_list(9, 1, write_list)
write_content(file_name_91, content_list_91)
#
# 80% read - 20% write
# file_name_82 = "../data/query_8_2_500.txt"
# content_list_82 = build_content_list(8, 2, write_list)
# write_content(file_name_82, content_list_82)

# file_name_73 = "../data/query_mixrw_uniform_h7c3.txt"
# content_list_73 = build_content_list_h7c3(7, 3, write_list)
# write_content(file_name_73, content_list_73)
#
# file_name_73 = "../data/query_mixrw_uniform_h3c7.txt"
# content_list_73 = build_content_list_h3c7(7, 3, write_list)
# write_content(file_name_73, content_list_73)