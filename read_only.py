import gc
import os
import json
import shutil
import time

from lsm.component import SpecialDiskComponent
from lsm.recordgenerator import SequentialNumericStringGenerator
from lsm.mergepolicy import LeveledPolicy, LazyLeveledPolicy
from lsm.tree import LeveledLSMTree, LSMTree, LazyLeveledLSMTree

# tree config
key_len = 20  # bigint of 20 bytes
data_len = 1022  # \t + key + key + key + key ... (50)... + \n 1002
component_size = 2000
mem_budget = component_size * (key_len + data_len)  # hold 10 records
num_records = 1000000
size_ratio = 2
l_0 = 1
M = 20
fragment_threshold = 4
IO_unit = 100
cache_size = 20
do_load = True # if do_load = False, do_reopen
do_point_query = False

generator = SequentialNumericStringGenerator(0, 20)

root = os.path.dirname(os.path.realpath(__file__))

policies = (
    (LeveledPolicy.policy_name(), (l_0, size_ratio, size_ratio, LeveledPolicy.PICK_MIN_OVERLAP, 1, IO_unit, cache_size, True, False), LSMTree.PROP_VALUE_BINARY),
    # (LazyLeveledPolicy.policy_name(),(l_0, size_ratio, size_ratio, LazyLeveledPolicy.PICK_MIN_OVERLAP, 1, M, False, IO_unit, cache_size, False, False),LSMTree.PROP_VALUE_BINARY),
    # (LazyLeveledPolicy.policy_name(),(l_0, size_ratio, size_ratio, LazyLeveledPolicy.PICK_MIN_OVERLAP, 1, M, False, IO_unit, cache_size, True, False), LSMTree.PROP_VALUE_BINARY),
)

def load_queries(file_name):
    queries = []
    f = open(file_name, 'r')
    line = f.readline()
    while line:
        query = line.rstrip("\n")
        queries.append(query)
        line = f.readline()
    f.close()
    return queries

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

def load_tree(lsm_tree, policy):
    # load data
    # file_name = "./data/random_data_" + str(num_records) + "_with_anti_heavy" + ".txt"
    # file_name = "./data/random_data_" + str(num_records) + "_with_anti_light" + ".txt"
    file_name = "./data/random_data_" + str(num_records) + "_with_anti_medium" + ".txt"
    
    f = open(file_name, 'r')
    line = f.readline()
    while line != "":
        line = line.replace('\n', '')
        tmp = line.split(',')
        key = tmp[0]
        data = tmp[1]
        # print(key + ": " + data)
        key_b = generator.bytes_from_number(key)
        data_b_tmp = generator.bytes_from_number(data)
        data_b = b"\t"
        for j in range(50):
            data_b += data_b_tmp
        data_b += data_b_tmp
        data_b += b"\n"
        # data_b = b"\t"
        # data_b += generator.bytes_from_number(data)
        # data_b += b"\n"
        lsm_tree.add_key_data(key_b, data_b)
        line = f.readline()

    print("Save test:")
    lsm_tree.save()
    cfg_path = os.path.join(lsm_tree.location(), LSMTree.DEFAULT_CONFIG_NAME)
    with open(cfg_path, "r") as cfgf:
        cfg = json.loads(cfgf.read())
    print("Content of {0}: {1}".format(cfg_path, cfg))
    del lsm_tree
    del policy

def reopen_tree(policy, queries, cost_results):
    print("Load test:")
    lsm_tree = LSMTree.from_base_dir(base_dir)
    print(lsm_tree.location(), lsm_tree.merge_policy().policy_name(), lsm_tree.merge_policy().properties())
    print()

    # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
    cnt_r = 0
    cnt_p = 0
    # evaluation matrix
    total_num_cache_hit = 0
    total_read_cost = 0
    total_search_write_cost = 0
    total_search_io = 0
    total_num_r = 0
    total_num_op = 0
    total_num_level = 0

    for q_id in range(len(queries)):
        query = queries[q_id]
        tem = query.split(",")
        # 0: range query, 1: point query
        if tem[0] == "0":
            range_query = tem[1]
            tem_query = range_query.split("-")
            min_key = generator.bytes_from_number(int(tem_query[1]))
            max_key = generator.bytes_from_number(int(tem_query[2]))
            query_size = int(tem_query[0])
            if policy_name == LeveledPolicy.policy_name():
                # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results, 5: new_merged_components
                # 6: # of component, 7: # of level
                scanner,read_cost, r_write_cost, num_cache_hit, num_r, new_merged_components, num_op, num_level = lsm_tree.create_scanner(min_key, True, max_key, True, query_size, True)
            else:
                # scanner, read_cost, r_write_cost, num_cache_hit, num_r, new_merged_components = lsm_tree.create_scanner(min_key, True, max_key, True, query_size, True)
                # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results, 5: new_merged_components
                scanner,read_cost, r_write_cost, num_cache_hit, num_r, new_merged_components = lsm_tree.create_scanner_basePartition_save_cost(min_key, True, max_key, True, query_size, True)
            total_num_r += num_r
            # print(range_query, " results num = ", num_r)
            total_num_cache_hit += num_cache_hit
            total_read_cost += read_cost
            total_search_write_cost += r_write_cost
            total_search_io += read_cost + r_write_cost
            total_num_op += num_op
            total_num_level += num_level
            # count num of disk components
            num_disk_component = len(lsm_tree.disk_components())
            # 0: idx, 1: num_disk_component, 2: total_search_io, 3: total read, 4: total write
            str_cost = str(cnt_r) + "," + str(num_disk_component) + "," + str(total_search_io) + "," + str(total_read_cost) + "," + str(total_search_write_cost) \
                       + "," + str(num_op) + "," + str(num_level) + "," + str(read_cost*IO_unit)  + "\n"
            cost_results.write(str_cost)
            del scanner
            gc.collect()
            cnt_r += 1
        if (q_id % 10000) == 0:
            # oc_results.close()
            cost_results.close()
            print("current idx = ", q_id)
            cost_results = open(cost_results_f, 'a')

    str_cost = str(cnt_r) + "," + str(num_disk_component) + "," + str(total_search_io) + "," + str(
        total_read_cost) + "," + str(total_search_write_cost)+ "\n"
    cost_results.write(str_cost)
    finish_time = time.time()
    print("----------------")
    print("-----Finish-----")
    print("current idx = ", q_id)
    print("total_search_read_cost = ", total_read_cost, ", total_search_write_cost = ", total_search_write_cost)
    print("total_io = ", total_search_io, ", num_disk_component = ", num_disk_component)
    print("total number of cache hit = ", total_num_cache_hit)
    print("total_num_r = ", total_num_r)
    print("total_num_op = ", total_num_op, ", total_num_level = ", total_num_level)
    # if policy_name == LazyLeveledPolicy.policy_name():
    #     lsm_tree.sp_usage.close()
    lsm_tree.query_results.close()
    del lsm_tree
    del policy
    return finish_time
    # shutil.rmtree(base_dir)

# create policy
for policy_name, props, impl in policies:
    base_dir = os.path.join(root, policy_name + "-test")
    if do_load:
        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir)

    if policy_name == LeveledPolicy.policy_name():
        lsm_tree = LeveledLSMTree(base_dir, key_len, data_len, mem_budget, props[5], props[6], props[7], props[8], impl)
        policy = LeveledPolicy(lsm_tree, props[0], props[1], props[2], props[3], props[4])
    elif policy_name == LazyLeveledPolicy.policy_name():
        lsm_tree = LazyLeveledLSMTree(base_dir, key_len, data_len, mem_budget, props[7], props[8], props[9], props[10], impl)
        # props[6] is have complementary set
        policy = LazyLeveledPolicy(lsm_tree, props[0], props[1], props[2], props[3], props[4], props[0], props[5], props[6], props[10])
    lsm_tree.set_merge_policy(policy)

    print("Dir: {0}, policy: {1}, properties: {2}"
          .format(lsm_tree.location(), policy.policy_name(), policy.properties()))


    if do_load:
        load_tree(lsm_tree, policy)
    else:
        start_time = time.time()
        query_set_name = "./data/query_uniform_10_long.txt"
        # query_set_name = "./data/query_hot_uniform.txt"
        query_set = load_range_queries(query_set_name)
        # oc_results_f = "./result/" + str(policy.policy_name()) + "_oc_results.txt"
        # oc_results = open(oc_results_f, 'w')
        cost_results_f = "./result/" + str(policy.policy_name()) + "_cost_results.txt"
        cost_results = open(cost_results_f, 'w')
        finish_time = reopen_tree(policy, query_set, cost_results)
        # oc_results.close()
        cost_results = open(cost_results_f, 'a')
        str_r = str(finish_time - start_time) + "\n"
        cost_results.write(str_r)
        cost_results.close()
        print("--- %s seconds ---" % (finish_time - start_time))


