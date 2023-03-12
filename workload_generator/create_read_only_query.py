import random
import numpy as np


MIN_KEY = 1  # Smallest key
# NUM_KEYS = 200  # Total number of keys
NUM_KEYS = 1000000  # Total number of keys
max_key = MIN_KEY + NUM_KEYS - 1

component_size = 2000
min_query_length = 3000
max_query_length = 4000
query_length = 6000
range_query_size = 100000
point_query_size = 10000
# 0: Zipf distribution
ZIPF_ALPHA = 2
# 1: Possion distribution
# lam = 994000
lam = 10000
# lam = 100

def build_random_query(query_size):
    query_set_1 = []
    data_list_1 = sorted(data_list)
    while len(query_set_1) < query_size:
        min_q = random.randint(1, max_key-1)
        max_q = 0
        if (min_q + query_length - 1) >= max_key:
            max_q = min_q
            min_q = max_q - query_length + 1
        else:
            max_q = min_q + query_length - 1
        query = "0" + "," +  str(query_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_1.append(query)
    return query_set_1

def build_query_with_ratio73(hot_ratio, cold_ratio):
    hot_query_size = range_query_size * hot_ratio
    cold_query_size = range_query_size * cold_ratio

    # if hot: 70% = 14K, cold 30% = 6K
    # 10500, 110500 create 7K * 2 = 14K
    # 200,000 - 990,000 is cold
    # (990,000-200,000) / 3000 = 263, 3K*2 = 6K
    # every range query size = 260

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
        query = "0" + "," + str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_hot.append(query)
    query_set += query_set_hot
    query_set += query_set_hot

    # cold part
    query_set_cold = []
    for i in range(200000, 990000):
        min_q = i
        max_q = min_q + q_length - 1
        query = "0" + "," + str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_cold.append(query)
        if len(query_set_cold) == 3000:
            break
    query_set += query_set_cold
    query_set += query_set_cold

    random.shuffle(query_set)
    return query_set

def build_query_with_ratio37(hot_ratio, cold_ratio):
    hot_query_size = range_query_size * hot_ratio
    cold_query_size = range_query_size * cold_ratio

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
        query = "0" + "," + str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_hot.append(query)
    query_set += query_set_hot
    query_set += query_set_hot
    query_set += query_set_hot

    # cold part
    query_set_cold = []
    for i in range(100000, 990000):
        min_q = i
        max_q = min_q + q_length - 1
        query = "0" + "," + str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_cold.append(query)
        if len(query_set_cold) == 7000:
            break
    query_set += query_set_cold
    query_set += query_set_cold

    random.shuffle(query_set)
    return query_set

def build_query_cold(min_key_range, max_key_range, q_length):
    query_set = []
    # cnt = (max_key_range-max_key_range) / q_length
    min_q = min_key_range
    while (min_q < max_key_range):
        max_q = min_q + q_length - 1
        query = "0" + "," + str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set.append(query)
        min_q += q_length
    random.shuffle(query_set)
    return query_set

def build_query_cold_sparse(min_key_range, max_key_range, q_length):
    query_set = []
    # cnt = (max_key_range-max_key_range) / q_length
    min_q = min_key_range
    while (min_q < max_key_range):
        max_q = min_q + q_length - 1
        query = "0" + "," + str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set.append(query)
        min_q += component_size
    random.shuffle(query_set)
    return query_set

def build_uniform_query(low, high, query_size):
    query_set_1 = []
    # uniform_list = np.random.uniform(low, high, range_query_size)
    # uniform_list = np.random.randint(low, high, range_query_size)
    uniform_list = np.random.randint(low, high, query_size)
    for min_q in uniform_list:
        min_q = int(min_q)
        while (min_q + query_length - 1) >= max_key:
            min_q_l = np.random.uniform(low, high, 1)
            min_q = int(min_q_l[0])
        max_q = min_q + query_length - 1
        query = "0" + "," + str(query_length) + "-" + str(min_q) + "-" + str(max_q)
        query_set_1.append(query)
    random.shuffle(query_set_1)
    return query_set_1

def build_Zipf_query(keys, parameter, time_independent):
    # parameter is ZIPF_ALPHA, if Zipf distribution
    # parameter is lam, if Possion distribution
    print("length of keys = ", len(keys))
    range_queries = []

    range_index_list = np.random.zipf(parameter, range_query_size)

    for idx in range_index_list:
        index = NUM_KEYS - idx
        max_q = keys[index]
        q_length = random.randint(min_query_length, max_query_length)
        min_q = max_q - q_length + 1
        if time_independent == False:
            while min_q < 0:
                index = np.random.zipf(parameter, 1)
                max_index = NUM_KEYS - index
                max_q = keys[max_index[0]]
                min_q = max_q - q_length + 1
        query = "0" + "," + str(q_length) + "-" + str(min_q) + "-" + str(max_q)
        range_queries.append(query)

    return range_queries

def build_multiple_centers_possion_query(keys, parameter, time_independent, multiple_centers):

    if time_independent:
        keys = sorted(keys)

    range_queries = []

    range_index_list = np.random.poisson(parameter, int(range_query_size/len(multiple_centers)))
    print("leng of range_index_list = ", len(range_index_list))

    for center in multiple_centers:
        for idx in range_index_list:
            # q_length = random.randint(min_query_length, max_query_length)
            q_length = query_length
            min_q = keys[idx] + center
            max_q = min_q + q_length - 1
            while max_q > NUM_KEYS:
                min_q_index = np.random.poisson(parameter, 1)
                min_q = keys[min_q_index[0]]
                max_q = min_q + q_length - 1
            query = "0" + "," + str(q_length) + "-" + str(min_q) + "-" + str(max_q)
            range_queries.append(query)
    random.shuffle(range_queries)
    return range_queries

def build_multiple_centers_possion_query(parameter, multiple_centers):

    range_queries = []

    possion_list = np.random.poisson(parameter, int(range_query_size/len(multiple_centers)))

    for center in multiple_centers:
        for min_q in possion_list:
            min_q = int(min_q)
            while (min_q + query_length - 1) >= max_key:
                min_q_l = np.random.poisson(parameter, 1)
                min_q = int(min_q_l[0])
            max_q = min_q + query_length - 1
            query = "0" + "," + str(query_length) + "-" + str(min_q) + "-" + str(max_q)
            range_queries.append(query)
    random.shuffle(range_queries)
    return range_queries


def write_query(file_name, query_set):
    f = open(file_name, 'w')
    for q in query_set:
        f.write(q)
        f.write('\n')
    f.close()

file_name = "../data/random_data_" + str(NUM_KEYS) + ".txt"
f = open(file_name, 'r')
line = f.readline()
data_list = []
while line != "":
    line = line.replace('\n', '')
    data_list.append(int(line))
    line = f.readline()
f.close()

# multiple_centers = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000]
multiple_centers = [0, 100000, 200000, 300000, 400000]
#
query_set_3 = build_uniform_query(80500, 110500, 20000)
query_set_4 = build_uniform_query(360500, 410500, 15000)
query_set_5 = build_uniform_query(760500, 810500, 15000)
query_set_6 = query_set_3 + query_set_4 + query_set_5
# query_set_3 = build_uniform_query(90500, 100500)
file_name_3 = "../data/query_read_only_motivation.txt"
write_query(file_name_3, query_set_6)
print("uniform query = ", len(query_set_6))


