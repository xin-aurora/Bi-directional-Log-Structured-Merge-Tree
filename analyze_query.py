
def load_range_queries(query_file_name):
    queries = []
    f = open(query_file_name, 'r')
    line = f.readline()
    while line:
        query = line.rstrip("\n")
        queries.append(query)
        line = f.readline()
    f.close()
    return queries

def analysis_queries(query_set):

    hot_queries = set()

    query_dic = dict()

    for query in query_set:
        if query not in query_dic:
            query_dic[query] = 1
        else:
            query_dic[query] += 1

    for q, q_f in query_dic.items():
        if q_f != 1:
            hot_queries.add(q)

    return hot_queries

def count_random_query(query_set: list, random_create_result: list, hot_query_set: set):

    # num = 0
    random_create_set = set()
    for i in range(len(random_create_result)):
        tem = random_create_result[i].rstrip("\n").split(":")
        query = query_set[i]
        if tem[1] == "1":
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

def write_query(file_name, query_set):
    f = open(file_name, 'w')
    for q in query_set:
        f.write(q)
        f.write('\n')
    f.close()

query_name_1 = "./data/query_hot_3.txt"
query_set_1 = load_range_queries(query_name_1)
print(len(query_set_1))
# hot_query_1 = analysis_queries(query_set_1)
# file_name_1 = "./hot_query_in_1.txt"
# write_query(file_name_1, hot_query_1)
#
# query_name_2 = "./data/query_hot_3.txt"
# query_set_2 = load_range_queries(query_name_2)
# hot_query_2 = analysis_queries(query_set_2)
# file_name_2 = "./hot_query_in_3.txt"
# write_query(file_name_2, hot_query_2)

# analyze random query
# read random result
random_result_f = "./e3.txt"
random_create_result = load_range_queries(random_result_f)
hot_query_set_f = "./hot_query_in_3.txt"
hot_query_set = load_range_queries(hot_query_set_f)
count_random_query(query_set_1, random_create_result, hot_query_set)
# count_random_query(query_set: list, random_create_result: list, hot_query_set: set):
# print(len(random_result))