import operator
import sys

""" Base class of partition function. """

class Partition():
    def __init__(self, partition_strategy: str):
        self.partition_strategy = partition_strategy

class Uniform_partition(Partition):
    def __init__(self, partition_strategy: str, min_key: int, max_key: int, \
                 partition_unit: int, disk_IO: int, k):
        super().__init__(partition_strategy)
        # self.partition_function
        self.min_key = min_key # min_key in overall key range
        self.max_key = max_key # max_key in overall key range
        self.partition_unit = partition_unit # later may change to partition function
        self.num_counters = int ((self.max_key - self.min_key) / partition_unit)
        self.setup_counters()
        self.disk_IO = disk_IO
        self.theta = 1
        self.setup_threshold()
        self.top_k_size = k # realted to top-k
        self.sentinel_ranges = dict() # key: range idx, value: score
        self.frozen_threshold = (partition_unit / 2)
        self.frozen = True
        self.num_operations = 0
        # self.read_write_ratio = 2.3  # later can be compute by read / write operations
        self.total_read = 0 # operation
        self.total_write = 0 # operation
        self.mem_component_size = int ((self.max_key - self.min_key) / partition_unit)

    def get_sentinel_ranges(self):
        return self.sentinel_ranges

    def setup_counters(self):
        self.read_counters: list = [0] * self.num_counters
        self.write_counters: list = [0] * self.num_counters

    def setup_threshold(self):
        self.create_threshold = self.partition_unit / self.disk_IO
        self.create_threshold = self.create_threshold * self.theta
        self.total_save = 0

    # write a key into mem table
    # update write counter
    def insert_key(self, key):
        counter_idx = self.get_range_by_key(key)
        self.write_counters[counter_idx] += 1

    # crease sentinel component
    # update write counter
    def insert_range_write(self, min_key, max_key):
        # increase operations
        self.num_operations += 1
        self.total_write += 1

        range = (min_key, max_key)
        counter_idxs = self.get_ranges_by_range(range)
        for idx in counter_idxs:
            self.read_counters[idx] += self.partition_unit

    # range query
    # update read counter
    def insert_range_read(self, q_min_key, q_max_key, frequency: int):

        # increase operations
        self.num_operations += 1

        sentienal_ranges = []
        query = (q_min_key,q_max_key)
        counter_idxs = self.get_ranges_by_range(query)
        for idx in counter_idxs:
            self.read_counters[idx] += frequency
            # compute score by read counter and write counter
            socre = self.read_counters[idx] - self.write_counters[idx]
            if len(self.sentinel_ranges) < self.top_k_size:
                self.sentinel_ranges[idx] = socre
                sentienal_ranges.append(idx)
            else:
                sorted_dict = sorted(self.sentinel_ranges.items(), key=operator.itemgetter(1))
                for cur_min_in_queue in sorted_dict:
                    if socre >= cur_min_in_queue[1]:
                        del self.sentinel_ranges[cur_min_in_queue[0]]
                        # update top-k list
                        self.sentinel_ranges[idx] = socre
                        sentienal_ranges.append(idx)
                        break
                    else:
                        break

        if self.num_operations > self.frozen_threshold:
            self.frozen = False
            return sentienal_ranges
        else:
            return []


    def insert_range_read_cost(self, map_range_level: dict, cur_mem_comp_size: int):

        # increase operations
        self.total_read += 1

        # rest read operation = stay in L0, computed by read write ratio
        # rest write = (memory table size - current table size)
        # rest read : rest write = read write ratio
        # rest read = rest write * read&write r
        if self.total_write != 0:
            read_write_ratio = self.total_read / self.total_write
            rest_read_operations = (self.mem_component_size - cur_mem_comp_size) * read_write_ratio
        else:
            # read only workload
            # current component safe forever
            rest_read_operations = -1

        sentienal_ranges = []
        for key, num_level in map_range_level.items():
            # key = (r_min_key, r_max_key, idx)
            idx = key[2]
            save = len(num_level) - 1
            self.read_counters[idx] += save
            if rest_read_operations == -1:
                # read only
                estimate_save = self.read_counters[idx]
            else:
                # avg_save = current save / current Hit (read frequency)
                # estimate frequency = current Hit / total Hit * rest Hit
                # estimate_save = current save * (rest Hit / total Hit)
                estimate_save = self.read_counters[idx] / self.total_read * rest_read_operations

            if estimate_save > self.create_threshold:
                sentienal_ranges.append(idx)
                self.sentinel_ranges[idx] = estimate_save

        # if get read write ratio by real read write ratio
        # need to setup cold phase
        if self.frozen:
            # buffer enough operations to counter
            # then decided create sentinel component
            if (self.total_read + self.total_write) > self.frozen_threshold:
                self.frozen = False
                return sentienal_ranges
            else:
                return []
        else:
            # frozen = false, can create sentinel component
            return sentienal_ranges

    def get_range_by_key(self, key) -> int:
        # uniform distribution
        # get range-idx by key / partition unit
        return int ((int(key) - self.min_key)/self.partition_unit)

    def get_ranges_by_range(self, range: tuple()) -> list:
        # uniform distribution
        # get range-idx by key / partition unit
        match_list: list = []
        start_range = int ( (int(range[0]) - self.min_key)/self.partition_unit)
        end_range = int ( ( int(range[1]) - self.min_key)/self.partition_unit)
        # if start_range == end_range, only match to one range
        # else, put all overlapping ranges into match_list
        if (end_range > start_range):
            while (start_range <= end_range):
                match_list.append(start_range)
                start_range+=1
        else:
            match_list.append(start_range)

        return match_list

    def reset_read_counter_by_range(self, range: tuple()):
        match_list = self.get_ranges_by_range(range)
        for idx in match_list:
            self.read_counters[idx] = 0
            self.write_counters[idx] = 0
            del self.sentinel_ranges[idx]

# if __name__ == "__main__":
#     min_key = 1
#     max_key = 101
#     unit = 10
#     par = Uniform_partition("Uniform", min_key, max_key, unit, 3)
#
#     key_1 = 19
#     print(par.get_range_by_key(key_1))
#
#     range_1 = (19,36)
#     # print(par.get_ranges_by_range(range_1))
#
#     # print(par.read_counters)
#     par.insert_range_read(19, 36, 1)
#     # print(par.read_counters)
#     # print(par.write_counters)
#     par.insert_key(19)
#     # print(par.write_counters)
#
#     # par.insert_range_read(22, 48, 1)
#     # par.insert_range_read(13, 30, 1)
#     # par.insert_range_read(19, 42, 1)
#
#     print(par.insert_range_read(22, 48, 1))
#     print(par.insert_range_read(13, 30, 1))
#     print(par.insert_range_read(19, 42, 1))
#     print(par.insert_range_read(2, 14, 1))
#
#     top_k = par.get_top_k()
#     print(top_k)

