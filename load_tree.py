import gc
import os
import json
import shutil

from lsm.recordgenerator import SequentialNumericStringGenerator
from lsm.mergepolicy import LeveledPolicy, LazyLeveledPolicy
from lsm.tree import LeveledLSMTree, LSMTree, LazyLeveledLSMTree

# tree config
key_len = 20  # bigint of 20 bytes
data_len = 1022  # \t + key + key + key + key ... (50)... + \n 1002
mem_budget = 2000 * (key_len + data_len)  # hold 10 records
num_records = 1000000 # 500 flushes
# num_records = 140
size_ratio = 2
l_0 = 1
M = 10
fragment_threshold = 10
IO_unit = 2000
do_load = False # if do_load = False, do_reopen

generator = SequentialNumericStringGenerator(0, 20)

root = os.path.dirname(os.path.realpath(__file__))

policies = (
    # (LeveledPolicy.policy_name(), (l_0, size_ratio, size_ratio, LeveledPolicy.PICK_MIN_OVERLAP, 1, IO_unit), LSMTree.PROP_VALUE_BINARY),
    (LazyLeveledPolicy.policy_name(),(l_0, size_ratio, size_ratio, LazyLeveledPolicy.PICK_MIN_OVERLAP, 1, 0.2, fragment_threshold, M, M, False, True, False, IO_unit),
     LSMTree.PROP_VALUE_BINARY),
    # (LazyLeveledPolicy.policy_name(), (l_0, size_ratio, size_ratio, LazyLeveledPolicy.PICK_MIN_OVERLAP, 1, 0.2, fragment_threshold, M, M, False, False, False, IO_unit),
    #  LSMTree.PROP_VALUE_BINARY),
)

def scan_test(lsm_tree, queryies, oc_results, cost_results):
    print("Range query test:")
    # 0: (0:q_idx, 1: OC_size, 2: disjoint_OC, 3: disjoint_OC_rate, 4: overlap_OC, 5: overlap_OC_rate)
    # 1: (0:q_idx, 1: read_cost, 2: accu_read_cost, 3: write_cost, 4: accu_write_cost, 5: search_IO, 6: accu_search_IO)
    total_write_cost = 0
    total_num_r = 0
    total_disjoint_OC  = 0
    total_overlap_OC  = 0
    base_operational_size  = 0
    total_read_cost  = 0
    total_search_io = 0
    total_io_records = 0
    idx = 0
    for query in queryies:
        tem = query.split("-")
        min_key = generator.bytes_from_number(int(tem[1]))
        max_key = generator.bytes_from_number(int(tem[2]))
        query_size = int(tem[0])
        scanner, operational_size, disjoint_OC, overlap_OC, read_cost, write_cost, search_io,num_r, avg_ranges = lsm_tree.create_scanner(
            min_key, True, max_key, True, query_size, True)
        # print("num of results = ", num_r)
        total_write_cost += write_cost
        total_num_r += num_r
        total_disjoint_OC += disjoint_OC
        total_overlap_OC += overlap_OC
        base_operational_size += operational_size
        total_read_cost += read_cost
        # search_io = read_cost + write_cost
        total_search_io += search_io
        io_records = search_io / num_r
        total_io_records += io_records
        # 0: (0:q_idx, 1: OC_size, 2: disjoint_OC, 3: disjoint_OC_rate, 4: overlap_OC, 5: overlap_OC_rate), 6: average_num_of_ranges, 7: num_rf
        str_oc = str(idx) + "," + str(operational_size) + "," + str(disjoint_OC) + "," + str(disjoint_OC/operational_size) + \
            "," + str(overlap_OC) + "," + str(overlap_OC/operational_size) + "," + str(avg_ranges) + "," + str(lsm_tree.num_rf) +"\n"
        oc_results.write(str_oc)
        # 1: (0:q_idx, 1: read_cost, 2: accu_read_cost, 3: write_cost, 4: accu_write_cost, 5: search_IO, 6: accu_search_IO),
        # 7: search_IO/num_records, 8: accu_search_IO/num_records
        str_cost = str(idx) + "," + str(read_cost) + "," + str(total_read_cost) + "," + str(write_cost) + "," + \
            str(total_write_cost) + "," + str(search_io) + "," + str(total_search_io) + "," + str(io_records) + \
        "," + str(total_io_records) + "\n"
        cost_results.write(str_cost)
        lsm_tree.query_results.write(str(idx) + ': ' + str(num_r)+'\n')
        idx += 1
        del scanner
        gc.collect()
        if (idx % 100) == 0:
            oc_results.close()
            cost_results.close()
            print("current idx = ", idx)
            # oc_results_f = "./result/" + str(policy.policy_name()) + "_oc_results.txt"
            oc_results = open(oc_results_f, 'a')
            # cost_results_f = "./result/" + str(policy.policy_name()) + "_cost_results.txt"
            cost_results = open(cost_results_f, 'a')
            # count fragment
            f_results_f = "./result/" + str(policy.policy_name()) + "_f_results.txt"
            f_results = open(f_results_f, 'a+')
            disk_components = lsm_tree._disk_components()
            component_map = {}
            for c in disk_components:
                num_ranges = len(c.key_ranges())
                # print(c.name(), " ranges = ", num_ranges)
                if num_ranges in component_map:
                    component_map[num_ranges] += 1
                else:
                    component_map[num_ranges] = 1
            f_results.write(str(idx) + ': fragment info = ' +str(component_map) + '\n')
            f_results.close()

    print("base_operational_size =", base_operational_size, ", total disjoint OC = ", total_disjoint_OC,
          ", total overlap OC = ", total_overlap_OC, ", total read pages = ", total_read_cost)
    print("total_num_r = ", total_num_r)

    lsm_tree.query_results.write('total num_results = ' + str(total_num_r) + '\n')

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
    file_name = "./data/random_data_" + str(num_records) + ".txt"
    f = open(file_name, 'r')
    line = f.readline()
    data_set = []
    while line != "":
        line = line.replace('\n', '')
        data_set.append(int(line))
        line = f.readline()
    f.close()
    # load tree
    for i in range(num_records):
        key = generator.bytes_from_number(data_set[i])
        # data = b"\t"
        # data += key
        # data += b"\n"
        data = b"\t"
        for j in range(50):
            data += key
        data += key
        data += b"\n"
        lsm_tree.add_key_data(key, data)
    lsm_tree.query_results.write("info of tree\n")
    leveled_disk_components = lsm_tree.leveled_disk_components()
    for lv in range(len(leveled_disk_components)):
        level_info = "level " + str(lv) + " contains " + str(len(leveled_disk_components[lv])) + " components."
        lsm_tree.query_results.write(level_info)
        lsm_tree.query_results.write("\n")
        for c in leveled_disk_components[lv]:
            lsm_tree.query_results.write(c.name())
            lsm_tree.query_results.write(", and its key range: ")
            for c_min, c_max in c.key_ranges():
                key_info = str(c_min) + "-" + str(c_max)
                lsm_tree.query_results.write(key_info)
                lsm_tree.query_results.write("\n")
    lsm_tree.query_results.close()

    print("Save test:")
    lsm_tree.save()
    cfg_path = os.path.join(lsm_tree.location(), LSMTree.DEFAULT_CONFIG_NAME)
    with open(cfg_path, "r") as cfgf:
        cfg = json.loads(cfgf.read())
    print("Content of {0}: {1}".format(cfg_path, cfg))
    del lsm_tree
    del policy

def reopen_tree(lsm_tree, policy, queryies, oc_results, cost_results):
    print("Load test:")
    lsm_tree = LSMTree.from_base_dir(base_dir)
    print(lsm_tree.location(), lsm_tree.merge_policy().policy_name(), lsm_tree.merge_policy().properties())
    print()
    scan_test(lsm_tree, queryies, oc_results, cost_results)
    del lsm_tree
    del policy
    # shutil.rmtree(base_dir)

# create policy
for policy_name, props, impl in policies:
    base_dir = os.path.join(root, policy_name + "-test")
    if do_load:
        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir)

    if policy_name == LeveledPolicy.policy_name():
        # props[5] is I/O unit
        lsm_tree = LeveledLSMTree(base_dir, key_len, data_len, mem_budget, props[5], impl)
        policy = LeveledPolicy(lsm_tree, props[0], props[1], props[2], props[3], props[4])
    elif policy_name == LazyLeveledPolicy.policy_name():
        # (l_0, size_ratio, size_ratio, LazyLeveledPolicy.PICK_MIN_OVERLAP, 1, 0.2, fragment_threshold, M, M, False, True,
        # False, IO_unit)
        # 0: l_0, 1: l_1, 2: size_ratio, 3: pick strategy, 4: overflow, 5: length_t, 6: frag_t, 7: ssc_threshold, 8: usc_threshold
        # 9: with_unsafe, 10: do_remove_fragment, 11: partial_merge, 12: I/O unit
        # def __init__(self, base_dir: str, key_len: int, data_len: int, mem_budget: int, do_remove_fragment: bool,
        #              do_partial_merge: bool, IO_unit: int,
        #              impl: str = LSMTree.PROP_VALUE_BINARY):
        lsm_tree = LazyLeveledLSMTree(base_dir, key_len, data_len, mem_budget, props[10], props[11], props[12],
                                      impl)
        # def __init__(self, tree, components_0: int, components_1: int, ratio: float, pick: str, overflow: int,
        #              length_t: float, size_t: int, nc_threshold: int, ssc_threshold: int, usc_threshold: int,
        #              with_unsafe: bool, do_partial_merge: bool):
        policy = LazyLeveledPolicy(lsm_tree, props[0], props[1], props[2], props[3], props[4], props[5], props[6],
                                   props[0], props[7], props[8], props[9], props[11])
    lsm_tree.set_merge_policy(policy)

    print("Dir: {0}, policy: {1}, properties: {2}"
          .format(lsm_tree.location(), policy.policy_name(), policy.properties()))

    if policy_name == LazyLeveledPolicy.policy_name():
        print()
        print("with unsafe sp = ", policy.with_unsafe, ", remove fragment = ", lsm_tree.remove_frag,
              ", do partial merge = ", lsm_tree.do_partial_merge)
        print()

    if do_load:
        load_tree(lsm_tree, policy)
    else:
        query_set_name = "./data/query_1_1.txt"
        query_set = load_range_queries(query_set_name)
        oc_results_f = "./result/" + str(policy.policy_name()) + "_oc_results.txt"
        oc_results = open(oc_results_f, 'w')
        cost_results_f = "./result/" + str(policy.policy_name()) + "_cost_results.txt"
        cost_results = open(cost_results_f, 'w')
        reopen_tree(lsm_tree, policy, query_set, oc_results, cost_results)
        oc_results.close()
        cost_results.close()


