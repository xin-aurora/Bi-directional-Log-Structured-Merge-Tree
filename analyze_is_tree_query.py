from lsm.istree import IntervalStatisticTree

def load_range_queries(file_name):
    queries = []
    f = open(file_name, 'r')
    line = f.readline()
    while line:
        query = line.rstrip("\n")
        queries.append(query)
        line = f.readline()
    f.close()
    return queries

def count_random_query(query_set: list, random_create_result: list, hot_query_set: set):

    # num = 0
    random_create_set = set()
    for i in range(len(random_create_result)):
        tem = random_create_result[i].rstrip("\n")
        query = query_set[i]
        if tem == "1":
            # num += 1
            random_create_set.add(query)
    # print("total nun create = ", num)
    print("total num of create query = ", len(random_create_set))

    num_create_hot = 0
    num_create_cold = 0
    for query in random_create_set:
        if query in hot_query_set:
            num_create_hot += 1
        else:
            num_create_cold += 1
    print("num_create_hot = ", num_create_hot)
    print("num_create_cold", num_create_cold)
    return num_create_hot, num_create_cold

# exp-1
query_name_1 = "./data/query_hot_1.txt"
query_set_1 = load_range_queries(query_name_1)

hot_query_set_f = "./hot_query_in_1.txt"
hot_query_set = load_range_queries(hot_query_set_f)

random_result_f = "./lazy_create_sp_info_similarity_e1.txt"
random_create_result = load_range_queries(random_result_f)

# print(len(random_create_result))

count_random_query(query_set_1, random_create_result, hot_query_set)




