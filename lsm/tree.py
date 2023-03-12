import gc
import math
import os
import sys
import time
import glob
import json
from threading import Lock
from typing import Optional, List, Union
import collections

from .component import Component, AbstractDiskComponent, AbstractMemoryComponent, MemoryComponent, MemoryDBComponent, \
    DiskComponent, DiskDBComponent, SpecialDiskComponent, SpecialMemoryComponent
from .mergepolicy import LazyLeveledPolicy
from .partition import Uniform_partition
from .recordgenerator import SequentialNumericStringGenerator, SHA256Generator
from .scanner import DiskScanner, MemoryScanner, DiskDBScanner, MemoryDBScanner, LSMScanner, Scanner
from .istree import IntervalStatisticTree, RandomTree, IntervalNode


class LSMTree:
    """ Base class of LSM-tree. """

    DEFAULT_CONFIG_NAME = "config.json"
    PROP_KEY_LENGTH = "key-len"
    PROP_DATA_LENGTH = "data-len"
    PROP_MEMORY_BUDGET = "mem-budget"
    PROP_MERGE_POLICY = "policy"
    PROP_MERGE_POLICY_PROPERTIES = "policy-props"
    PROP_IMPLEMENTATION = "impl"
    PROP_VALUE_BINARY = "binary"
    PROP_VALUE_SQLITE3 = "sqlite3"
    PROP_IOUNIT = "io-unit"
    PROP_CACHE = "cache-size"
    PROP_USEISTREE = "use-is-tree"
    PROP_REMOVE_FRAGMENT = "do-remove-fragment"
    PROP_PARTIAL_MERGE = "do-partial-merge"
    PROP_COST_COUNTER = "have-cost-counter"
    PROP_REBUILDING = "do-rebuilding"

    @classmethod
    def from_config(cls, base_dir: str, config: Union[dict, str]):
        """ Create a LSMTree from a configuration object as string or dict. """
        if isinstance(config, dict):
            tree_info = config
        elif isinstance(config, str):
            try:
                tree_info = json.loads(config)
            except json.JSONDecodeError as e:
                raise ValueError("Error parsing {0}: {1}".format(config, e))
            if type(tree_info) != dict:
                raise TypeError("{0} cannot be converted to dict".format(config))
        else:
            raise TypeError("{0} {1} is not supported".format(config, type(config)))
        try:
            key_len = int(tree_info[LSMTree.PROP_KEY_LENGTH])
            data_len = int(tree_info[LSMTree.PROP_DATA_LENGTH])
            mem_budget = int(tree_info[LSMTree.PROP_MEMORY_BUDGET])
            impl = str(tree_info[LSMTree.PROP_IMPLEMENTATION]).strip().lower()
            IO_unit = int(tree_info[LSMTree.PROP_IOUNIT])
            cache_size = int(tree_info[LSMTree.PROP_CACHE])
            use_counter = bool(tree_info[LSMTree.PROP_COST_COUNTER])
            do_rebuilding = bool(tree_info[LSMTree.PROP_REBUILDING])
        except (KeyError, ValueError) as e:
            raise KeyError("Missing parameter or invalid parameter type {0}: {1}".format(tree_info, e))
        if key_len < 1 or data_len < 0:
            raise ValueError("Invalid parameter: {0} = {1}, {2} = {3}"
                             .format(LSMTree.PROP_KEY_LENGTH, key_len, LSMTree.PROP_DATA_LENGTH, data_len))
        if key_len + data_len > mem_budget:
            raise ValueError("Invalid parameter: {0} = {1}, {2} + {3}= {4}"
                             .format(LSMTree.PROP_MEMORY_BUDGET, mem_budget,
                                     LSMTree.PROP_KEY_LENGTH, LSMTree.PROP_DATA_LENGTH, key_len + data_len))
        if impl != LSMTree.PROP_VALUE_BINARY and impl != LSMTree.PROP_VALUE_SQLITE3:
            raise ValueError("Invalid parameter: {0} = \"{1}\"".format(LSMTree.PROP_impl, impl))
        pname = tree_info.get(LSMTree.PROP_MERGE_POLICY, "")
        from .mergepolicy import LeveledPolicies
        if pname in LeveledPolicies:
            if pname == LazyLeveledPolicy.policy_name():
                tree = LazyLeveledLSMTree(base_dir, key_len, data_len, mem_budget, IO_unit,cache_size, use_counter, do_rebuilding, impl)
            else:
                tree = LeveledLSMTree(base_dir, key_len, data_len, mem_budget, IO_unit, cache_size, use_counter, do_rebuilding, impl)
        else:
            tree = StackLSMTree(base_dir, key_len, data_len, mem_budget, impl)
        from .mergepolicy import MergePolicy
        policy = MergePolicy.from_config(tree_info, tree)
        if policy is None:
            del tree
            raise ValueError("Error creating merge policy from configuration: {0}".format(tree_info))
        tree.set_merge_policy(policy)
        return tree

    @classmethod
    def from_config_file(cls, base_dir: str, config_path: str):
        if not os.path.isfile(config_path):
            raise FileNotFoundError("{0} does not exist".format(os.path.abspath(config_path)))
        with open(config_path, "r") as cfgf:
            return LSMTree.from_config(base_dir, cfgf.read())

    @classmethod
    def from_base_dir(cls, base_dir: str):
        if not os.path.isdir(base_dir):
            raise FileNotFoundError("{0} does not exist".format(os.path.abspath(base_dir)))
        return LSMTree.from_config_file(base_dir, os.path.join(base_dir, LSMTree.DEFAULT_CONFIG_NAME))

    def __init__(self, base_dir: str, key_len: int, data_len: int, mem_budget: int, IO_unit: int, cache_size: int, use_counter: bool, do_rebuilding: bool, impl: str = PROP_VALUE_BINARY):
        self._base_dir = os.path.abspath(base_dir)
        self._key_len = key_len
        if self._key_len < 1:
            raise ValueError("key_len must be at least 1")
        self._data_len = data_len
        if self._data_len < 0:
            raise ValueError("data_len must be at least 0")
        self._budget = mem_budget
        if self._budget < self._key_len + self._data_len:
            raise ValueError("Memory budget {0} is smaller than record size {1}".format(mem_budget, key_len + data_len))
        if impl.lower() == LSMTree.PROP_VALUE_SQLITE3:
            self._impl = 1  # SQLite3
        else:
            self._impl = 0  # Binary
        self._mem: Optional[AbstractMemoryComponent] = None
        self._disks = []
        self._io_lock = Lock()
        if isinstance(self, StackLSMTree):
            from .mergepolicy import StackPolicy
            self._merge_policy: Optional[StackPolicy] = None
        else:
            from .mergepolicy import LeveledPolicy
            self._merge_policy: Optional[LeveledPolicy] = None
        self._loaded = False
        self._init_min_id, self._init_max_id = self._next_memory_component_ids()
        if not os.path.isdir(self._base_dir):
            # os.mkdir(self._base_dir, 755)  # New database
            os.mkdir(self._base_dir)  # New database
        else:
            self._load()
        self.__allocate_memory_component()
        self.IO_unit = IO_unit
        self.cache_size = cache_size
        self.cache_pid_range = collections.OrderedDict()
        self.cache_cname_pid = {}
        self.min_pid = 0
        self.cur_pid = -1
        self.mem_name = None
        self.use_counter = use_counter
        self.do_rebuilding = do_rebuilding

    def __del__(self):
        if self._mem is not None:
            del self._mem
        for d in self.disk_components():
            del d

    def is_primary(self):
        return self._data_len > 0

    # Override in sub-class
    def _disk_components(self) -> List[AbstractDiskComponent]:
        return self._disks

    # Override in sub-class
    def _load(self) -> bool:
        pass

    # Override in sub-class
    def _flush(self) -> Optional[AbstractDiskComponent]:
        return self._mem.flush(self._mem.min_id(), self._mem.max_id(), self._base_dir)

    # Override in sub-class
    def _merge(self, components: List[AbstractDiskComponent], max_records: int) -> List[AbstractDiskComponent]:
        # No merge
        return []

    # Override in sub-class
    def _next_memory_component_ids(self) -> (int, int):
        return -1, -1

    # Override in sub-class
    def _add_components(self, new_components: List[AbstractDiskComponent]) -> None:
        pass

    # Override in sub-class
    def _remove_components(self, old_components: List[AbstractDiskComponent]) -> None:
        pass

    # Override in sub-class
    def _components_info(self) -> []:
        return []

    def __allocate_memory_component(self) -> None:
        self._mem = self._new_memory_component(self._init_min_id, self._init_max_id)

    def location(self) -> str:
        return self._base_dir

    def components_info(self, oneline=True) -> str:
        if oneline:
            return json.dumps(self._components_info())
        else:
            return json.dumps(self._components_info(), indent=4)

    def set_merge_policy(self, policy) -> bool:
        from .mergepolicy import MergePolicy
        if policy is None:
            raise ValueError("policy must not be None")
        if not isinstance(policy, MergePolicy):
            raise TypeError("policy {0} is not an instance of {1}".format(type(policy), MergePolicy))
        self._merge_policy: MergePolicy = policy
        self._disks = self._merge_policy.sort_components()
        return True

    def save_to(self, config_path: str) -> bool:
        """ Save the current status. """
        with self._io_lock:
            if self._mem.num_records()[0] > 0:
            # if self._mem.num_records() > 0:
                print(self._mem.num_records()[0])
                if not self._flush(False):
                    return False
            try:
                with open(config_path, "w") as cfgf:
                    if self._impl == 1:
                        impl_value = LSMTree.PROP_VALUE_SQLITE3
                    else:
                        impl_value = LSMTree.PROP_VALUE_BINARY
                    config = {
                        LSMTree.PROP_KEY_LENGTH: self._key_len,
                        LSMTree.PROP_DATA_LENGTH: self._data_len,
                        LSMTree.PROP_MEMORY_BUDGET: self._budget,
                        LSMTree.PROP_MERGE_POLICY: self._merge_policy.policy_name(),
                        LSMTree.PROP_MERGE_POLICY_PROPERTIES: self._merge_policy.properties(),
                        LSMTree.PROP_IMPLEMENTATION: impl_value,
                        LSMTree.PROP_IOUNIT: self.IO_unit,
                        LSMTree.PROP_CACHE: self.cache_size,
                        LSMTree.PROP_COST_COUNTER: self.use_counter,
                        LSMTree.PROP_REBUILDING: self.do_rebuilding,
                    }
                    cfgf.write(json.dumps(config, indent=4, sort_keys=True))
                cfgf.close()
            except IOError as e:
                raise IOError("Failed to save: {0}".format(e))
        return True

    def save(self):
        return self.save_to(os.path.join(self._base_dir, self.DEFAULT_CONFIG_NAME))

    def merge_policy(self):
        return self._merge_policy

    def memory_component(self) -> Optional[AbstractMemoryComponent]:
        return self._mem

    def disk_components(self) -> List[AbstractDiskComponent]:
        return self._disk_components()

    def _new_memory_component(self, min_id: int, max_id: int) -> AbstractMemoryComponent:
        if self._impl == 1:
            return MemoryDBComponent(min_id, max_id, self._key_len, self._data_len, self._budget)
        else:
            return MemoryComponent(min_id, max_id, self._key_len, self._data_len, self._budget)

    def _new_disk_component(self, min_id: int, max_id: int) -> AbstractDiskComponent:
        if self._impl == 1:
            return DiskDBComponent(min_id, max_id, self._key_len, self._data_len, self._base_dir)
        else:
            return DiskComponent(min_id, max_id, self._key_len, self._data_len, self._base_dir)

    def _disk_component_ext(self):
        if self._impl == 1:
            return Component.DISK_DB_EXT
        else:
            return Component.DISK_BIN_EXT

    def add_key_data(self, key: bytes, data: Optional[bytes], upsert: bool = True, load: bool = True) -> (bool, int):
        if len(key) != self._key_len:
            raise ValueError("Invalid key of length {0}, while {1} is required".format(len(key), self._key_len))
        if data is None or len(data) == 0:
            if self.is_primary():
                raise ValueError("Data of length {0} is required".format(self._data_len))
        else:
            if len(data) != self._data_len:
                raise ValueError("Invalid data of length {0}, while {1} is required".format(len(data), self._data_len))
        with self._io_lock:
            if not upsert:
                idx, rkey, rdata = self.search(key)
                if rkey is not None:
                    if idx == -1:
                        print("Duplicate key found in memory: {0}".format(rkey.hex()))
                    else:
                        print("Duplicate key found at disk component {0}: {1}".format(idx, rkey.hex()))
                    return False, 0, 0
            self._mem.write_key_data(key, data)
            total_write_cost = 0
            if self._mem.is_full():
                f_d = self._flush()
                for n_records in f_d.num_records():
                    total_write_cost += (n_records / self.IO_unit)
                if f_d is None:
                    raise IOError("Failed to flush the memory component")
                self._add_components([f_d, ])
                next_min, next_max = self._next_memory_component_ids()
                del self._mem
                self._mem = self._new_memory_component(next_min, next_max)
                if self._merge_policy is not None:
                    is_flush = True
                    while True:
                        mergable_components, max_records = self._merge_policy.get_mergable_components(is_flush)
                        is_flush = False
                        if len(mergable_components) > 0:
                            new_ds, num_write, write_cost = self._merge(mergable_components, max_records)
                            total_write_cost += write_cost
                            if len(new_ds) > 0:
                                self._remove_components(mergable_components)
                                self._add_components(new_ds)
                                self._disks = self._merge_policy.sort_components()
                                mergable_components = set(mergable_components)
                                mergable_components = list(mergable_components)
                                if len(mergable_components) > 1:
                                    for i in range(len(mergable_components)):
                                        md: AbstractDiskComponent = mergable_components[i]
                                        md.remove_files()
                                        del md
                        else:
                            break
            return True, total_write_cost

    def add_record(self, record: bytes, upsert: bool = True) -> bool:
        if len(record) != self._key_len + self._data_len:
            return False
        key = record[:self._key_len]
        data = record[self._key_len:] if self.is_primary() else None
        return self.add_key_data(key, data, upsert)

    def search(self, key: bytes) -> (int, Optional[bytes], Optional[bytes]):
        if self._mem.num_records()[0] > 0 and self._mem.min_key() <= key <= self._mem.max_key():
            rkey, rdata = self._mem.get_record(key)
            if rkey is not None:
                return -1, rkey, rdata, 1
        disks = self._disk_components().copy()
        read_amplification = 1
        for idx in range(0, len(disks)):
            d = disks[idx]
            if d.is_in_range(key):
                rkey, rdata = d.get_record(key)
                if rkey is None:
                    read_amplification += 1
                else:
                    return idx, rkey, rdata, read_amplification
                # if rkey is not None:
                #     return idx, rkey, rdata
        return -2, None, None, read_amplification

    @staticmethod
    def overlapping(search_min: Optional[bytes], include_min: bool,
                    search_max: Optional[bytes], include_max: bool,
                    min_key: bytes, max_key: bytes) -> bool:
        if search_min is None and search_max is None:
            return True
        if search_min is not None:
            if search_min > max_key or (search_min == max_key and not include_min):
                return False
        if search_max is not None:
            if search_max < min_key or (search_max == min_key and not include_max):
                return False
        return True

    def create_scanner(self, min_key: bytes, include_min: bool, max_key: bytes, include_max: bool, length: int, update) \
            -> Optional[LSMScanner]:
        scanners = []
        if self._mem.num_records()[0] > 0 and \
                self.overlapping(min_key, include_min, max_key, include_max, self._mem.min_key(), self._mem.max_key()):
            if self._impl == 1:
                scanner = MemoryDBScanner(self._mem)
            else:
                scanner = MemoryScanner(self._mem)
            if min_key is not None:
                scanner.set_min_key(min_key, include_min)
            if max_key is not None:
                scanner.set_max_key(max_key, include_max)
            scanners.append(scanner)
        disks = self.disk_components().copy()
        if self._impl == 1:
            for d in disks:
                for ridx in d.overlapping_range_indexes(min_key, include_min, max_key, include_max):
                    scanner = DiskDBScanner(d, ridx)
                    if min_key is not None:
                        scanner.set_min_key(min_key, include_min)
                    if max_key is not None:
                        scanner.set_max_key(max_key, include_max)
                    scanners.append(scanner)
        else:
            for d in disks:
                for ridx in d.overlapping_range_indexes(min_key, include_min, max_key, include_max):
                    scanner = DiskScanner(d, ridx, self.IO_unit)
                    if min_key is not None:
                        scanner.set_min_key(min_key, include_min)
                    if max_key is not None:
                        scanner.set_max_key(max_key, include_max)
                    scanners.append(scanner)
                    # self.query_results.write(d.name())
                    # self.query_results.write('\n')
        num_r = 0
        operational_components = set()
        if len(scanners) > 0:
            lsm_scanners = LSMScanner(scanners, length, self.IO_unit)
            lsm_scanners.open()
            while True:
                d_name, ridx, key, data = lsm_scanners.next()
                if key is None:
                    break
                else:
                    num_r += 1
                    key_cname = str(int(key)) + " from " + str(d_name)
                    # self.query_results.write(key_cname)
                    # self.query_results.write('\n')
                    operational_components.add(d_name)
            total_read_cost = lsm_scanners.close()

            leveled_disk_components = self.leveled_disk_components()
            group_1 = []
            group_2 = []
            for o_c in operational_components:
                # self.query_results.write(o_c)
                # self.query_results.write('\n')
                tem = o_c.split("-")
                d_min_id = tem[0]
                tem_1 = tem[1].split(".")
                d_max_id = tem_1[0]
                for d_c in leveled_disk_components[int(d_min_id)]:
                    if d_c.max_id() == int(d_max_id):
                        # print("d_c max_id = ", d_c.max_id())
                        # print("target max_id = ", d_max_id)
                        # print()
                        # if d_c has overlap with components in group1
                        if len(group_1) > 0:
                            for d_1 in group_1:
                                if self._merge_policy.is_overlapping(d_1, d_c):
                                # if self.is_unsafe(d_1, d_c):
                                    group_2.append(d_1)
                                    group_1.remove(d_1)
                                    if d_c not in group_2:
                                        group_2.append(d_c)
                        # if d_c has overlap with components in group2
                        elif len(group_2) > 0:
                            for d_2 in group_2:
                                if d_c not in group_2:
                                    if self._merge_policy.is_overlapping(d_2, d_c):
                                    # if self.is_unsafe(d_2, d_c):
                                        group_2.append(d_c)
                        if d_c not in group_2:
                            # print("group 1 = ", len(group_1))
                            group_1.append(d_c)

            # scanner, operational_size, disjoint_OC, overlap_OC, read_cost, write_cost, num_r
            return lsm_scanners, len(lsm_scanners.scanners()), len(group_1), len(group_2), total_read_cost, 0, total_read_cost, num_r, 1
        else:
            return None, None, None, None, None, None, None

    def get_new_pid (self) -> int:
        self.cur_pid += 1
        return self.cur_pid

    def get_pages_from_position(self, start_pos: int, num_read_records: int) -> List[tuple]:
        record_length = self._mem.record_length()
        if num_read_records == 0:
            end_pos = start_pos + self.IO_unit * self._mem.record_length()
        else:
            end_pos = start_pos + num_read_records * self._mem.record_length()
        # count how many page read
        num_page = num_read_records / self.IO_unit
        if (num_read_records % self.IO_unit) != 0:
            num_page = int(num_page) + 1
        else:
            num_page = int(num_page)
        # the pages used to store in cache
        list_scan_read_pages = []
        for i in range(num_page):
            start_pos = start_pos + self.IO_unit * i * record_length
            end_pos = start_pos + self.IO_unit * record_length
            # t = (start_pos, end_pos)
            t = (start_pos, end_pos, 1)
            list_scan_read_pages.append(t)
        return list_scan_read_pages

    def update_cache(self, lsm_scanners: LSMScanner) -> (int, int):
        total_num_cache_hit = 0
        total_num_read_page = 0
        total_num_read_page_memory = 0
        for scanner in lsm_scanners.scanners():
            if not scanner.is_memory:
                scanner: DiskScanner
                num_read_page, num_cache_hit = self.update_disk_scanner_cache(scanner)
                total_num_cache_hit += num_cache_hit
                total_num_read_page += num_read_page
            else:
                # memory scanner
                scanner: MemoryScanner
                num_read_records = scanner.num_read_records
                total_num_read_page_memory += num_read_records
        return total_num_cache_hit, total_num_read_page, total_num_read_page_memory

    def update_disk_scanner_cache(self, scanner: DiskScanner) -> (int, int):
        num_read_page = 0
        num_cache_hit = 0
        c_name = scanner.component().name()
        num_read_records = scanner.num_read_records
        scanner_start_pos = scanner.start_pos
        list_scan_read_pages = self.get_pages_from_position(scanner_start_pos, num_read_records)
        if c_name in self.cache_cname_pid:
            # page in cache
            page_list: list = self.cache_cname_pid[c_name]
            for new_page in list_scan_read_pages:
                tem_new_page_start = new_page[0]
                tem_new_page_end = new_page[1]
                bl = False
                for p_id in page_list:
                    if p_id not in self.cache_pid_range:
                        # if p_id is invalid
                        page_list.remove(p_id)
                    elif p_id < self.min_pid:
                        page_list.remove(p_id)
                    else:
                        cache_page: tuple = self.cache_pid_range[p_id]
                        # print("current_page_range = ", cur_page_range)
                        # compare
                        if cache_page[0] <= tem_new_page_start < tem_new_page_end <= cache_page[1]:
                            # cache hit
                            num_cache_hit += 1
                            f = cache_page[2]
                            f += 1
                            new_cache_page = (cache_page[0], cache_page[1], f)
                            bl = True
                            # page replacement
                            # LRU: Least Recently Used
                            new_pid = self.get_new_pid()
                            page_list.remove(p_id)
                            page_list.append(new_pid)
                            del self.cache_pid_range[p_id]
                            self.cache_pid_range[new_pid] = new_cache_page
                            break
                        elif cache_page[0] <= tem_new_page_start < cache_page[1] < tem_new_page_end:
                            # update tem_new_page_start
                            tem_new_page_start = cache_page[1]
                        elif tem_new_page_start < cache_page[0] < tem_new_page_end <= cache_page[1]:
                            # update tem_new_page_end
                            tem_new_page_end = cache_page[0]
                if not bl:
                    p_id = self.get_new_pid()
                    self.cache_pid_range[p_id] = new_page
                    page_list.append(p_id)
                    # num_page = int (math.ceil(scanner.num_read_records / self.IO_unit))
                    # num_read_page += num_page
                    num_read_page += 1
                    if len(self.cache_pid_range) > self.cache_size:
                        for fist_p, first_range in self.cache_pid_range.items():
                            del self.cache_pid_range[fist_p]
                            self.min_pid = fist_p
                            break
                # update page list
                self.cache_cname_pid[c_name] = page_list
        else:
            num_read_page += self.insert_pages_into_cache(c_name, list_scan_read_pages)
        return num_read_page, num_cache_hit

    def insert_component_into_cache(self, components: List[SpecialDiskComponent]):
        for component in components:
            c_name = component.name()
            start_pos = 0
            record_length = component.record_length()
            list_scan_read_pages = self.get_pages_from_position(start_pos, record_length)
            if c_name in self.cache_cname_pid:
                # replace with new pages
                self.delete_page_by_component(component)
                for new_page in list_scan_read_pages:
                    p_id = self.get_new_pid()
                    self.cache_pid_range[p_id] = new_page
                    if c_name in self.cache_pid_range:
                        page_list: list = self.cache_cname_pid[c_name]
                    else:
                        page_list = []
                    page_list.append(p_id)
                    if len(self.cache_pid_range) > self.cache_size:
                        for fist_p, first_range in self.cache_pid_range.items():
                            del self.cache_pid_range[fist_p]
                            self.min_pid = fist_p
                            break
                # update page list
                self.cache_cname_pid[c_name] = page_list
            else:
                self.insert_pages_into_cache(c_name, list_scan_read_pages)

    def insert_pages_into_cache(self, c_name: str, list_read_pages: List[tuple]) -> int:
        num_read_page = 0
        page_list = []
        for new_page in list_read_pages:
            p_id = self.get_new_pid()
            self.cache_pid_range[p_id] = new_page
            page_list.append(p_id)
            num_read_page += 1
            if len(self.cache_pid_range) > self.cache_size:
                for fist_p, first_range in self.cache_pid_range.items():
                    del self.cache_pid_range[fist_p]
                    self.min_pid = fist_p
                    break
        self.cache_cname_pid[c_name] = page_list
        return num_read_page

    def delete_page_by_component(self, component):
        c_name = component.name()
        if c_name in self.cache_cname_pid:
            old_list = self.cache_cname_pid[c_name]
            del self.cache_cname_pid[c_name]
            for old_p_id in old_list:
                if old_p_id > self.min_pid:
                    del self.cache_pid_range[old_p_id]

class StackLSMTree(LSMTree):
    """ LSM-tree using any stack-based merge policy. """

    def __init__(self, base_dir: str, key_len: int, data_len: int, mem_budget: int,
                 impl: str = LSMTree.PROP_VALUE_BINARY):
        super().__init__(base_dir, key_len, data_len, mem_budget, impl)

    def set_merge_policy(self, policy) -> bool:
        from .mergepolicy import StackPolicy
        if policy is None:
            raise ValueError("policy must not be None")
        if not isinstance(policy, StackPolicy):
            raise TypeError("policy {0} is not an instance of {1}".format(type(policy), StackPolicy))
        self._merge_policy: StackPolicy = policy
        self._disks = self._merge_policy.sort_components()
        return True

    def _next_memory_component_ids(self) -> (int, int):
        if self._mem is None:
            return 1, 1
        else:
            return self._mem.min_id() + 1, self._mem.max_id() + 1

    def _add_components(self, new_components: List[AbstractDiskComponent]) -> None:
        self._disks = new_components + self._disks

    def _remove_components(self, old_components: List[AbstractDiskComponent]) -> None:
        for old_d in old_components.copy():
            self._disks.remove(old_d)

    def _load(self) -> bool:
        all_max_id = 0
        for file in glob.glob(os.path.join(self._base_dir, "*." + self._disk_component_ext())):
            basename = os.path.basename(file)[:-len(self._disk_component_ext()) - 1]
            min_id, max_id = [int(n) for n in basename.split("-")]
            all_max_id = max(all_max_id, max_id)
            self._disks.append(self._new_disk_component(min_id, max_id))
        self._init_min_id = all_max_id + 1
        self._init_max_id = all_max_id + 1
        return True

    def _merge(self, components: List[AbstractDiskComponent], max_records: int) -> List[AbstractDiskComponent]:
        # Merging component a-b, c-d and e-f, where a < b < c < d < e < f, new component ID will be a-f
        min_id = float("inf")
        max_id = -float("inf")
        scanners = []
        if self._impl == 1:
            for d in components:
                if d.min_id() < min_id:
                    min_id = d.min_id()
                if d.max_id() > max_id:
                    max_id = d.max_id()
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskDBScanner(d, ridx))
        else:
            for d in components:
                if d.min_id() < min_id:
                    min_id = d.min_id()
                if d.max_id() > max_id:
                    max_id = d.max_id()
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskScanner(d, ridx))
        new_d = self._new_disk_component(min_id, max_id)
        new_d.open()
        lsm_scanner = LSMScanner(scanners, -1)
        lsm_scanner.open()
        while True:
            d_name, ridx, key, data = lsm_scanner.next()
            if key is None:
                break
            if not new_d.write_key_data(key, data):
                new_name = new_d.name()
                new_d.close()
                if os.path.isfile(new_d.get_binary_path()):
                    os.remove(new_d.get_binary_path())
                lsm_scanner.close()
                del lsm_scanner
                del new_d
                raise IOError("Error bulk-loading to merged component {0}".format(new_name))
        lsm_scanner.close()
        new_d.close()
        del lsm_scanner
        return [new_d, ]

    def _components_info(self) -> []:
        info = []
        if self._mem is not None:
            info.append({
                "name": self._mem.name(),
                "size": self._mem.component_size(),
                "records": self._mem.num_records(),
                "min-key": "" if self._mem.min_key() is None else self._mem.min_key().hex(),
                "max-key": "" if self._mem.max_key() is None else self._mem.max_key().hex(),
            })
        for d in self._disks:
            info.append({
                "name":    d.name(),
                "size":    d.actual_component_size(),
                "records": d.actual_num_records(),
                "min-key": d.min_key().hex(),
                "max-key": d.max_key().hex(),
                "virtual": [{
                    "size": d.component_sizes()[i],
                    "records": d.num_records()[i],
                    "min-key": d.key_ranges()[i][0],
                    "max-key": d.key_ranges()[i][1],
                } for i in range(len(d.key_ranges()))]
            })
        return info


class LeveledLSMTree(LSMTree):
    """ LSM-tree using any Leveled merge policy. """

    def __init__(self, base_dir: str, key_len: int, data_len: int, mem_budget: int, IO_unit: int, cache_size: int, use_counter, do_rebuilding,
                 impl: str = LSMTree.PROP_VALUE_BINARY):
        self._id_lock = Lock()
        self._level_ids: List[int] = []
        self.IO_unit = IO_unit
        self.num_rf = 0
        super().__init__(base_dir, key_len, data_len, mem_budget, IO_unit, cache_size, use_counter, do_rebuilding, impl)
        # For Leveled LSM tree, each item in self._disks is a list of components in a level

    def set_merge_policy(self, policy) -> bool:
        from .mergepolicy import LeveledPolicy
        if policy is None:
            raise ValueError("policy must not be None")
        if not isinstance(policy, LeveledPolicy):
            raise TypeError("policy {0} is not an instance of {1}".format(type(policy), LeveledPolicy))
        self._merge_policy: LeveledPolicy = policy
        self._disks = self._merge_policy.sort_components()
        query_results_f = "./result/" + str(policy.policy_name()) + "_query_results.txt"
        self.query_results = open(query_results_f, 'w')
        return True

    def _load(self) -> bool:
        self._disks: List[List[AbstractDiskComponent]] = []
        max_lv0_id = 0
        map_disk = {}
        self._level_ids = []
        for file in glob.glob(os.path.join(self._base_dir, "*." + self._disk_component_ext())):
            basename = os.path.basename(file)[:-len(self._disk_component_ext()) - 1]
            min_id, max_id = [int(n) for n in basename.split("-")]
            # print("min_id = ", min_id, ", max_id = ", max_id)
            if min_id == 0:
                max_lv0_id = max(max_lv0_id, max_id)
            d = self._new_disk_component(min_id, max_id)
            if min_id in map_disk:
                map_disk[min_id].append(d)
            else:
                map_disk[min_id] = [d, ]
            # set _level_ids
            # if min_id in map_max_id:
            #     pre_max = map_max_id[min_id]
            #     if max_id > pre_max:
            #         map_max_id[min_id] = max_id
            # else:
            #     map_max_id[min_id] = max_id
        list_idx = sorted(map_disk)
        for i in list_idx:
            self._disks.append(map_disk[i])
            # self._level_ids.append(map_max_id[i])
        for level in self._disks:
            self._level_ids.append(max([d.max_id() for d in level]))
        self._init_max_id = max_lv0_id + 1
        self._level_ids[0] = self._init_max_id
        return True

    def leveled_disk_components(self) -> List[List[AbstractDiskComponent]]:
        return self._disks

    def _next_memory_component_ids(self) -> (int, int):
        return 0, self._next_level_id(0)

    def _disk_components(self) -> List[AbstractDiskComponent]:
        components = []
        for level in self._disks:
            components += level
        return components

    def _next_level_id(self, lv: int) -> int:
        with self._id_lock:
            if lv >= len(self._level_ids):
                self._level_ids.append(1)
                return 1
            else:
                id = self._level_ids[lv] + 1
                self._level_ids[lv] = id
                return id

    def _merge(self, level_from, components: List[AbstractDiskComponent], max_records: int, do_rename: bool) -> (List[AbstractDiskComponent], int, int):
        level_to = level_from + 1
        if do_rename:
            new_components = []
            for src in components:
                new_id = self._next_level_id(level_to)
                while src.is_reading():
                    time.sleep(0.5)
                src.rename(level_to, new_id)
                new_components.append(src)
            return new_components, 0, len(new_components)
        scanners = []
        if self._impl == 1:
            for d in components:
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskDBScanner(d, ridx))
        else:
            for d in components:
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskScanner(d, ridx, self.IO_unit))
        lsm_scanner = LSMScanner(scanners, -1, self.IO_unit)
        new_components = []
        new_d = None
        lsm_scanner.open()
        while True:
            d_name, ridx, key, data = lsm_scanner.next()
            if key is None:
                break
            if new_d is None:
                new_d = self._new_disk_component(level_to, self._next_level_id(level_to))
                new_d.open()
            if not new_d.write_key_data(key, data):
                new_name = new_d.name()
                new_d.close()
                if os.path.isfile(new_d.get_binary_path()):
                    os.remove(new_d.get_binary_path())
                del new_d
                for new_d in new_components:
                    if os.path.isfile(new_d.get_binary_path()):
                        os.remove(new_d.get_binary_path())
                    del new_d
                lsm_scanner.close()
                del lsm_scanner
                raise IOError("Error bulk-loading to merged component {0}".format(new_name))
            # if new_d.num_records() == max_records:
            if new_d.actual_num_records() == max_records:
                new_d.close()
                new_components.append(new_d)
                new_d = None
        if new_d is not None:
            new_d.close()
            new_components.append(new_d)
        lsm_scanner.close()
        del lsm_scanner
        total_num_write = 0
        total_write_cost = 0
        for n_d in new_components:
            num_write = 0
            for n_records in n_d.num_records():
                num_write += n_records
            write_cost = math.ceil(num_write / self.IO_unit)
            total_write_cost += write_cost
        total_write_cost += len(new_components)
        return new_components, total_num_write, total_write_cost

    def add_key_data(self, key: bytes, data: Optional[bytes], upsert: bool = True, load: bool = True) -> (bool, int, int):
        if len(key) != self._key_len:
            raise ValueError("Invalid key of length {0}, while {1} is required".format(len(key), self._key_len))
        if data is None or len(data) == 0:
            if self.is_primary():
                raise ValueError("Data of length {0} is required".format(self._data_len))
        else:
            if len(data) != self._data_len:
                raise ValueError("Invalid data of length {0}, while {1} is required".format(len(data), self._data_len))
        with self._io_lock:
            if not upsert:
                idx, rkey, rdata = self.search(key)
                if rkey is not None:
                    if idx == -1:
                        print("Duplicate key found in memory: {0}".format(rkey.hex()))
                    else:
                        print("Duplicate key found at disk component {0}: {1}".format(idx, rkey.hex()))
                    return False, 0, 0
            self._mem.write_key_data(key, data)
            # total_num_write = 0
            total_write_cost = 0
            if self._mem.is_full():
                self.mem_name = self._mem.name()
                f_d = self._flush()
                for n_records in f_d.num_records():
                    total_write_cost += (n_records / self.IO_unit)
                if f_d is None:
                    raise IOError("Failed to flush the memory component")
                self._add_components([f_d, ])
                next_min, next_max = self._next_memory_component_ids()
                del self._mem
                self._mem = self._new_memory_component(next_min, next_max)
                if self._merge_policy is not None:
                    is_flush = True
                    do_rename = False
                    while True:
                        picked, mergable_components, max_records = self._merge_policy.get_mergable_components(is_flush)
                        is_flush = False
                        if len(picked) > 0:
                            level_from = picked[0].min_id()
                            if len(mergable_components) == 0:
                                do_rename = True
                                new_ds, num_write, write_cost = self._merge(level_from, picked, max_records, do_rename)
                                mergable_components = picked
                            else:
                                mergable_components += picked
                                new_ds, num_write, write_cost = self._merge(level_from, mergable_components, max_records, do_rename)
                            total_write_cost += write_cost
                            self.insert_component_into_cache(new_ds)
                            if len(new_ds) > 0:
                                self._remove_components(mergable_components)
                                self._add_components(new_ds)
                                self._disks = self._merge_policy.sort_components()
                                mergable_components = set(mergable_components)
                                mergable_components = list(mergable_components)
                                if len(mergable_components) > 1:
                                    for i in range(len(mergable_components)):
                                        md: AbstractDiskComponent = mergable_components[i]
                                        if md not in new_ds:
                                            md.remove_files()
                                            del md
                        else:
                            break

            return True, total_write_cost

    def _add_components(self, new_components: List[AbstractDiskComponent]) -> None:
        for new_d in new_components:
            lv = new_d.min_id()
            if lv >= len(self._disks):
                self._disks.append([new_d, ])
            else:
                self._disks[lv] = [new_d, ] + self._disks[lv]

    def _remove_components(self, old_components: List[AbstractDiskComponent]) -> None:
        for old_d in old_components.copy():
            lv = old_d.min_id()
            removed = False
            try:
                self._disks[lv].remove(old_d)
                removed = True
            except (ValueError, IndexError) as e:
                # This may happen for a renamed component. Its min_id had incremented.
                pass
            if not removed and lv > 0:
                try:
                    # This may happen for a renamed component. Its min_id had incremented.
                    self._disks[lv - 1].remove(old_d)
                except (ValueError, IndexError):
                    pass

    def create_scanner(self, min_key: bytes, include_min: bool, max_key: bytes, include_max: bool, length: int, update) \
            -> Optional[LSMScanner]:
        scanners = []
        if self._mem.num_records()[0] > 0 and \
                self.overlapping(min_key, include_min, max_key, include_max, self._mem.min_key(), self._mem.max_key()):
            if self._impl == 1:
                scanner = MemoryDBScanner(self._mem)
            else:
                scanner = MemoryScanner(self._mem)
            if min_key is not None:
                scanner.set_min_key(min_key, include_min)
            if max_key is not None:
                scanner.set_max_key(max_key, include_max)
            scanners.append(scanner)
        disks = self.disk_components().copy()
        if self._impl == 1:
            for d in disks:
                for ridx in d.overlapping_range_indexes(min_key, include_min, max_key, include_max):
                    scanner = DiskDBScanner(d, ridx)
                    if min_key is not None:
                        scanner.set_min_key(min_key, include_min)
                    if max_key is not None:
                        scanner.set_max_key(max_key, include_max)
                    scanners.append(scanner)
        else:
            for d in disks:
                for ridx in d.overlapping_range_indexes(min_key, include_min, max_key, include_max):
                    scanner = DiskScanner(d, ridx, self.IO_unit)
                    if min_key is not None:
                        scanner.set_min_key(min_key, include_min)
                    if max_key is not None:
                        scanner.set_max_key(max_key, include_max)
                    scanners.append(scanner)
        num_r = 0
        operational_components = set()
        # number of level
        levels = set()
        if len(scanners) > 0:
            lsm_scanners = LSMScanner(scanners, length, self.IO_unit)
            lsm_scanners.open()
            while True:
                d_name, ridx, key, data = lsm_scanners.next()
                # print("    {0} from {1}".format(key, d_name))
                if key is None:
                    break
                else:
                    # print("    {0} from {1}".format(key, d_name))
                    num_r += 1
                    # key_cname = str(int(key)) + " from " + str(d_name)
                    # self.query_results.write(key_cname)
                    # self.query_results.write('\n')
                    operational_components.add(d_name)
                    tem = d_name.split("-")
                    levels.add(tem[0])

            # num_cache_hit = 0
            num_cache_hit, num_read_page, num_read_page_memory = self.update_cache(lsm_scanners)
            lsm_scanners.close()
            total_read_cost = num_read_page

            # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
            # 6: # of component, 7: # of level
            return lsm_scanners, total_read_cost, 0, num_cache_hit, num_r, [], len(operational_components), len(levels)
        else:
            return None, None, None, None, None, None, None, None

    def _components_info(self) -> []:
        info = []
        if self._mem is not None:
            info.append({
                "name": self._mem.name(),
                "size": self._mem.component_size(),
                "records": self._mem.num_records(),
                "min-key": "" if self._mem.min_key() is None else self._mem.min_key().hex(),
                "max-key": "" if self._mem.max_key() is None else self._mem.max_key().hex(),
            })
        for level in self._disks:
            level_info = []
            for d in level:
                level_info.append({
                    "name": d.name(),
                    "size": d.component_size(),
                    "records": d.num_records(),
                    "min-key": d.min_key().hex(),
                    "max-key": d.max_key().hex(),
                    "virtual": [{
                        "size": d.component_sizes()[i],
                        "records": d.num_records()[i],
                        "min-key": d.key_ranges()[i][0],
                        "max-key": d.key_ranges()[i][1],
                    } for i in range(len(d.key_ranges()))]
                })
            info.append(level_info)
        return info


class LazyLeveledLSMTree(LSMTree):
    """ LSM-tree using any Leveled merge policy. """

    def __init__(self, base_dir: str, key_len: int, data_len: int, mem_budget: int, IO_unit: int, cache_size: int, use_counter: bool, do_rebuilding: bool,
                 impl: str = LSMTree.PROP_VALUE_BINARY):
        self._id_lock = Lock()
        self._level_ids: List[int] = []
        self.safe_special_components = collections.OrderedDict()
        self.safe_special_components_names = set()
        # self.unsafe_special_components = []
        # key: normal component
        self.normal_components = []
        self._invalid_components = []
        self.IO_unit = IO_unit
        self.cache_size = cache_size
        self.cache_pid_range = collections.OrderedDict()
        self.cache_cname_pid = {}
        self.min_pid = 0
        self.cur_pid = -1
        self.num_rf = 0
        self._mem_special = None
        self.use_counter = use_counter
        if self.use_counter:
            # use cost function with threshold
            # 10 is k
            self.partition = Uniform_partition("Uniform", 1, 1000001, 2000, self.IO_unit, 10)
            # self.partition = Uniform_partition("Uniform", 1, 1001, 20, self.IO_unit, 10)
            self.do_update_counter = True
            self.num_query = 0
            self.do_update_counter_unit = 10
        self.generator = SequentialNumericStringGenerator(0, 20)
        self.complementary_set = None
        self.complementary_sets = []
        self.do_rebuilding = do_rebuilding
        super().__init__(base_dir, key_len, data_len, mem_budget, IO_unit, cache_size, use_counter, do_rebuilding, impl)

    def set_merge_policy(self, policy) -> bool:
        from .mergepolicy import LazyLeveledPolicy
        if policy is None:
            print("policy must not be None", file=sys.stderr)
            return False
        if not isinstance(policy, LazyLeveledPolicy):
            print("policy <{0}> is not an instance of LeveledPolicy".format(type(policy)), file=sys.stderr)
            return False
        self._merge_policy: LazyLeveledPolicy = policy
        self._disks = self._merge_policy.sort_components()
        query_result_f = "./result/" + str(policy.policy_name()) + "_query_result.txt"
        self.query_results = open(query_result_f, 'w')
        # sp_usage_f = "./result/" + str(policy.policy_name()) + "_sp_usage.txt"
        # self.sp_usage = open(sp_usage_f, 'w')
        # self._length_threshold = (int) (self._mem.max_records() * policy._length_t)
        # self.size_t = policy.size_ts
        self.have_componentary_set = policy.have_componentary
        # print("self.have_componentary_set = ", self.have_componentary_set)
        return True

    def _load(self) -> bool:
        self._disks: List[List[AbstractDiskComponent]] = []
        max_lv0_id = 0
        map_disk = {}
        self._level_ids = []
        for file in glob.glob(os.path.join(self._base_dir, "*." + self._disk_component_ext())):
            basename = os.path.basename(file)[:-len(self._disk_component_ext()) - 1]
            min_id, max_id = [int(n) for n in basename.split("-")]
            # print("min_id = ", min_id, ", max_id = ", max_id)
            if min_id == 0:
                max_lv0_id = max(max_lv0_id, max_id)
            d = self._new_disk_component(min_id, max_id, False)
            if min_id == 0:
                self.normal_components.append(d)
            if min_id in map_disk:
                map_disk[min_id].append(d)
            else:
                map_disk[min_id] = [d, ]
        list_idx = sorted(map_disk)
        for i in list_idx:
            self._disks.append(map_disk[i])
        for level in self._disks:
            self._level_ids.append(max([d.max_id() for d in level]))
        self._init_max_id = max_lv0_id + 1
        self._level_ids[0] = self._init_max_id
        self.create_sp_threshold = 0
        return True

    def add_key_data(self, key: bytes, data: bytes, upsert: bool = True, load: bool = True) -> (bool, int):
        if len(key) != self._key_len or len(data) != self._data_len:
            return False, 0
        with self._io_lock:
            if not upsert:
                idx, data = self.search(key)
                if data is not None:
                    if idx == -1:
                        print("Duplicate key found in memory: {0}".format(data.hex()))
                    else:
                        print("Duplicate key found at disk component {0}: {1}".format(idx, data.hex()))
                    return False, 0
            self._mem.write_key_data(key, data)
            # update ist
            # if not load:
            #     self.ist.insert_new_key(key)
            # total_num_write = 0
            total_write_cost = 0
            if self._mem.is_full():
                if self.have_componentary_set:
                    top_idxs = self.partition.get_sentinel_ranges()
                    complementary_ranges = self.get_ranges_by_partition_idx(top_idxs)
                    f_d, pathch_write_cost = self._flush(False, complementary_ranges)
                    # print("pathch_write_cost", pathch_write_cost)
                else:
                    f_d, pathch_write_cost = self._flush(False, [])
                total_write_cost += pathch_write_cost
                if f_d is not None:
                    for n_records in f_d.num_records():
                        # print("new regular component size = ", n_records)
                        total_write_cost += (n_records / self.IO_unit)
                    if f_d is None:
                        raise IOError("Failed to flush the memory component")
                    self._add_components([f_d, ])
                    next_min, next_max = self._next_memory_component_ids()
                    del self._mem
                    self._mem = self._new_memory_component(next_min, next_max)

                    # add newly unsafe special components
                    self.normal_components.append(f_d)

                    # new version
                    if self._merge_policy is not None:
                        is_flush = True
                        is_query = False
                        do_rename = False
                        lv = 0
                        while True:
                            picked, mergable_components, max_records, lv, keep_sp = self._merge_policy.get_mergable_components(
                                is_flush, is_query, False, lv)
                            is_flush = False
                            if len(picked) > 0:
                                level_from = picked[0].min_id()
                                # if there is not mergable component in lower level, do rename
                                if len(mergable_components) == 0:
                                    do_rename = True
                                    if keep_sp:
                                        # flush part of normal component
                                        # leave the part overlap with special component range (get tree root subtree_range) in L0
                                        # flush rest of normal component to lower level
                                        new_ds, num_write, write_cost = self.merge_complementary_memory(level_from,picked, max_records, do_rename)
                                    else:
                                        new_ds, num_write, write_cost = self._merge(level_from, picked, max_records, do_rename)
                                    mergable_components = picked
                                else:
                                    mergable_components = picked + mergable_components
                                    if keep_sp:
                                        new_ds, num_write, write_cost = self.merge_complementary_memory(level_from, picked, max_records,do_rename)
                                    else:
                                        new_ds, num_write, write_cost = self._merge(level_from, mergable_components, max_records, do_rename)
                                # total_num_write += num_write
                                total_write_cost += write_cost
                                self.insert_component_into_cache(new_ds)
                                if len(new_ds) > 0:
                                    self._remove_components(mergable_components)
                                    self._add_components(new_ds)
                                    self._disks = self._merge_policy.sort_components()
                                    mergable_components = set(mergable_components)
                                    mergable_components = list(mergable_components)
                                    if len(mergable_components) > 1:
                                        for i in range(len(mergable_components)):
                                            md: AbstractDiskComponent = mergable_components[i]
                                            # md.remove_files()
                                            # del md
                                            if md not in new_ds:
                                                md.remove_files()
                                                del md
                            else:
                                break
                            if lv == -1:
                                break
            return True, total_write_cost

    def _new_disk_component(self, min_id: int, max_id: int, is_merge) -> DiskComponent:
        if self._impl == 1:
            return DiskDBComponent(min_id, max_id, self._key_len, self._data_len, self._base_dir)
        else:
            return SpecialDiskComponent(min_id, max_id, self._key_len, self._data_len, self._base_dir, is_merge)

    def _new_memory_component(self, min_id: int, max_id: int) -> AbstractMemoryComponent:
        if self._impl == 1:
            return MemoryDBComponent(min_id, max_id, self._key_len, self._data_len, self._budget)
        else:
            return SpecialMemoryComponent(min_id, max_id, self._key_len, self._data_len, self._budget)

    def leveled_disk_components(self) -> List[List[SpecialDiskComponent]]:
        return self._disks

    def _next_memory_component_ids(self) -> (int, int):
        return 0, self._next_level_id(0)

    def _flush(self, is_special, complementary_ranges) -> Optional[SpecialDiskComponent]:
        if self.have_componentary_set:
            total_write_cost = 0
            for key in sorted(self._mem.records()):
                if self.key_move_to_complementary(key, complementary_ranges):
                    if self.complementary_set is None:
                        next_min, next_max = self._next_memory_component_ids()
                        # print("patch: next_min = ", next_min, ", next_max = ", next_max)
                        self.complementary_set = self._new_memory_component(next_min, next_max)
                    self.complementary_set.write_key_data(key, self._mem.records()[key])
                    self._mem.delete_key_data(key)
                    if self.complementary_set.is_full():
                        if not self.do_rebuilding:
                            # do rebuilding
                            self.complementary_set = None
                        else:
                            # # make a new complementary set
                            # self.complementary_sets.append(self.complementary_set)
                            # self.complementary_set = None
                            # make a patch
                            patch = self._flush_mem_complementary(False)
                            pathc_list = [patch, ]
                            # for n_d in pathc_list:
                                # total_write_cost += (n_d.num_records()[0] / self.IO_unit)
                            total_write_cost += (self._mem.max_records() / self.IO_unit)
                            # print("create patch set, patch cost = ", total_write_cost)
                            self._add_components(pathc_list)
                            self.insert_component_into_cache(pathc_list)
                            self.complementary_set = None
            if self._mem.num_records()[0] > 0:
                return self._mem.flush(self._mem.min_id(), self._mem.max_id(), self._base_dir, is_special), total_write_cost
            else:
                return None, total_write_cost
        else:
            return self._mem.flush(self._mem.min_id(), self._mem.max_id(), self._base_dir, is_special), 0

    def _flush_mem_special(self, is_special) -> Optional[SpecialDiskComponent]:
        return self._mem_special.flush(self._mem_special.min_id(), self._mem_special.max_id(), self._base_dir, is_special)

    def _flush_mem_complementary(self, is_special) -> Optional[SpecialDiskComponent]:
        return self.complementary_set.flush(self.complementary_set.min_id(), self.complementary_set.max_id(), self._base_dir, is_special)

    # def rebuilding_sentinel_components(self) -> list[SpecialDiskComponent]:
    #     # rebuilding sentinel components based on complementary_set
    #     sentinel_components = list(self.safe_special_components.values())
    #     rebuilding_map = {} # sc, keys
    #     for key in sorted(self.complementary_set__records.keys()):
    #         for sc in sentinel_components:
    #             sc: SpecialDiskComponent
    #             if self.key_move_to_complementary(key, [(sc.min_key(), sc.max_key())]):
    #                 if sc in rebuilding_map:
    #                     rebuilding_map[sc].append(key)
    #                 else:
    #                     rebuilding_map[sc] = [key,]
    #     sentinel_range_idxs = self.partition.get_sentinel_ranges()
        # for sc in rebuilding_map:
        #     # create special memory buffer



    def _disk_components(self) -> List[SpecialDiskComponent]:
        components = []
        for level in self._disks:
            components += level
        return components

    def invalid_component(self) -> List[SpecialDiskComponent]:
        return self._invalid_components

    def update_safety_ranges(self, new_min_key: bytes, new_max_key: bytes, new_merged_componentes: List[AbstractDiskComponent]):
        safety_range = str(int(new_min_key)) + "-" + str(int(new_max_key))
        self._safety_ranges[safety_range] = new_merged_componentes

    def _next_level_id(self, lv: int) -> int:
        with self._id_lock:
            if lv >= len(self._level_ids):
                self._level_ids.append(1)
                return 1
            else:
                id = self._level_ids[lv] + 1
                self._level_ids[lv] = id
                return id

    def _merge(self, level_from, components: List[SpecialDiskComponent], max_records: int, do_rename) -> (List[SpecialDiskComponent], int, int):
        # level_from = components[0].min_id()
        level_to = level_from + 1
        if do_rename:
            new_components = []
            for src in components:
                new_id = self._next_level_id(level_to)
                while src.is_reading():
                    time.sleep(0.5)
                src.rename(level_to, new_id)
                new_components.append(src)
            return new_components, 0, len(new_components)
        scanners = []
        if self._impl == 1:
            for d in components:
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskDBScanner(d, ridx))
        else:
            for d in components:
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskScanner(d, ridx, self.IO_unit))
        lsm_scanner = LSMScanner(scanners, -1)
        new_components = []
        new_d = None
        lsm_scanner.open()
        while True:
            d_name, ridx, key, data = lsm_scanner.next()
            if key is None:
                break
            if new_d is None:
                # TODO: is_all_merge is defined by mergable components
                is_all_merge = False
                new_d = self._new_disk_component(level_to, self._next_level_id(level_to), is_all_merge)
                new_d.open()
            if not new_d.write_key_data(key, data):
                new_name = new_d.name()
                new_d.close()
                if os.path.isfile(new_d.get_binary_path()):
                    os.remove(new_d.get_binary_path())
                del new_d
                for new_d in new_components:
                    if os.path.isfile(new_d.get_binary_path()):
                        os.remove(new_d.get_binary_path())
                    del new_d
                lsm_scanner.close()
                del lsm_scanner
                raise IOError("Error bulk-loading to merged component {0}".format(new_name))
            # if new_d.num_records() == max_records:
            if new_d.actual_num_records() == max_records:
                new_d.close()
                new_components.append(new_d)
                new_d = None

        if new_d is not None:
            new_d.close()
            new_components.append(new_d)
        lsm_scanner.close()
        del lsm_scanner
        total_num_write = 0
        total_write_cost = 0
        for n_d in new_components:
            num_write = 0
            for n_records in n_d.num_records():
                num_write += n_records
            write_cost = math.ceil(num_write / self.IO_unit)
            total_write_cost += write_cost
        total_write_cost += len(new_components)
        return new_components, total_num_write, total_write_cost

    def merge_complementary_memory(self, level_from, components: List[SpecialDiskComponent], max_records: int, do_rename):
        # print("keep sentinel component!")
        level_to = level_from + 1
        if do_rename:
            new_components = []
            for src in components:
                new_id = self._next_level_id(level_to)
                while src.is_reading():
                    time.sleep(0.5)
                src.rename(level_to, new_id)
                new_components.append(src)
            return new_components, 0, len(new_components)

        # flush part of normal component
        # leave the part overlap with special component range (get tree root subtree_range) in memory
        # flush rest of normal component to lower level

        top_idxs = self.partition.get_sentinel_ranges()
        complementary_ranges = self.get_ranges_by_partition_idx(top_idxs)

        total_write_cost = 0
        scanners = []
        if self._impl == 1:
            for d in components:
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskDBScanner(d, ridx))
        else:
            for d in components:
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskScanner(d, ridx, self.IO_unit))
        lsm_scanner = LSMScanner(scanners, -1)
        new_components = []
        new_d = None
        lsm_scanner.open()
        while True:
            d_name, ridx, key, data = lsm_scanner.next()
            if key is None:
                break
            if new_d is None:
                # TODO: is_all_merge is defined by mergable components
                is_all_merge = False
                new_d = self._new_disk_component(level_to, self._next_level_id(level_to), is_all_merge)
                new_d.open()
            # write besides istree root range

            if self.key_move_to_complementary(key, complementary_ranges):
                # print(key, " in complementary_set")
                if self.complementary_set is None:
                    next_min, next_max = self._next_memory_component_ids()
                    self.complementary_set = self._new_memory_component(next_min, next_max)
                self.complementary_set.write_key_data(key, data)
                if self.complementary_set.is_full():
                    patch = self._flush_mem_complementary(False)
                    pathc_list = [patch, ]
                    # for n_d in pathc_list:
                    # total_write_cost += (n_d.num_records()[0] / self.IO_unit)
                    total_write_cost += (self._budget / self.IO_unit)
                    # print("create patch set, cost = ", total_write_cost)
                    self._add_components(pathc_list)
                    self.insert_component_into_cache(pathc_list)
                    self.complementary_set = None
            else:
                new_d.write_key_data(key, data)

            # if new_d.num_records() == max_records:
            if new_d is not None and new_d.actual_num_records() == max_records:
                new_d.close()
                new_components.append(new_d)
                new_d = None
        if new_d.actual_num_records() > 0:
            new_d.close()
            new_components.append(new_d)
        lsm_scanner.close()
        del lsm_scanner
        total_num_write = 0
        for n_d in new_components:
            num_write = 0
            for n_records in n_d.num_records():
                num_write += n_records
            write_cost = math.ceil(num_write / self.IO_unit)
            total_write_cost += write_cost
        total_write_cost += len(new_components)

        return new_components, total_num_write, total_write_cost

    def key_move_to_complementary(self, key, complementary_ranges) -> bool:

        for b_range_min_key, b_range_max_key, idx in complementary_ranges:
            if b_range_min_key <= key <= b_range_max_key:
                return True

        return False

    def merge_keep_sp(self, level_from, components: List[SpecialDiskComponent], max_records: int, do_rename):
        # print("keep sentinel component!")
        level_to = level_from + 1
        if do_rename:
            new_components = []
            for src in components:
                new_id = self._next_level_id(level_to)
                while src.is_reading():
                    time.sleep(0.5)
                src.rename(level_to, new_id)
                new_components.append(src)
            return new_components, 0, len(new_components)

        # flush part of normal component
        # leave the part overlap with special component range (get tree root subtree_range) in L0
        # flush rest of normal component to lower level
        ist_tree_root: IntervalNode = self.ist.get_root()
        root_subtree_min = ist_tree_root.subtree_min
        root_subtree_max = ist_tree_root.subtree_max
        # 10000, 600000
        # generator = SequentialNumericStringGenerator(0, 20)
        # root_subtree_min = generator.bytes_from_number(10000)
        # root_subtree_max = generator.bytes_from_number(604000)
        # print("root_subtree_min = ", root_subtree_min, ", root_subtree_max = ", root_subtree_max)
        scanners = []
        if self._impl == 1:
            for d in components:
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskDBScanner(d, ridx))
        else:
            for d in components:
                for ridx in d.overlapping_range_indexes(None, True, None, True):
                    scanners.append(DiskScanner(d, ridx, self.IO_unit))
        lsm_scanner = LSMScanner(scanners, -1)
        new_components = []
        new_d = None
        complement_d = None
        lsm_scanner.open()
        while True:
            d_name, ridx, key, data = lsm_scanner.next()
            if key is None:
                break
            if new_d is None:
                # TODO: is_all_merge is defined by mergable components
                is_all_merge = False
                new_d = self._new_disk_component(level_to, self._next_level_id(level_to), is_all_merge)
                new_d.open()
            # write besides istree root range

            if key < root_subtree_min:
                new_d.write_key_data(key, data)
            elif key >= root_subtree_min and key <= root_subtree_max:
                if new_d.actual_num_records() > 0:
                    new_d.close()
                    new_components.append(new_d)
                    new_d = None
                if complement_d is None:
                    complement_d = self._new_disk_component(level_from, self._next_level_id(level_from), True)
                    complement_d.open()
                complement_d.write_key_data(key, data)
            else:
                new_d.write_key_data(key, data)

            # if new_d.num_records() == max_records:
            if new_d is not None and new_d.actual_num_records() == max_records:
                new_d.close()
                new_components.append(new_d)
                new_d = None
        if new_d.actual_num_records() > 0:
            new_d.close()
            new_components.append(new_d)
        if complement_d is not None:
            complement_d.close()
            new_components.append(complement_d)
            self.safe_special_components_names.add(complement_d.name())
            # print("A complement component, name = ", complement_d.name())
        lsm_scanner.close()
        del lsm_scanner
        total_num_write = 0
        total_write_cost = 0
        for n_d in new_components:
            num_write = 0
            for n_records in n_d.num_records():
                num_write += n_records
            write_cost = math.ceil(num_write / self.IO_unit)
            total_write_cost += write_cost
        total_write_cost += len(new_components)

        return new_components, total_num_write, total_write_cost

    def create_new_ds(self, lsm_scanner, level_to, max_records):
        new_components = []
        new_d = None
        lsm_scanner.open()
        while True:
            d_name, ridx, key, data = lsm_scanner.next()
            if key is None:
                break
            if new_d is None:
                is_all_merge = False
                new_d = self._new_disk_component(level_to, self._next_level_id(level_to), is_all_merge)
                new_d.open()
            if not new_d.write_key_data(key, data):
                new_name = new_d.name()
                new_d.close()
                if os.path.isfile(new_d.get_binary_path()):
                    os.remove(new_d.get_binary_path())
                del new_d
                for new_d in new_components:
                    if os.path.isfile(new_d.get_binary_path()):
                        os.remove(new_d.get_binary_path())
                    del new_d
                lsm_scanner.close()
                del lsm_scanner
                raise IOError("Error bulk-loading to merged component {0}".format(new_name))
            # if new_d.num_records() == max_records:
            if new_d.actual_num_records() == max_records:
                new_d.close()
                new_components.append(new_d)
                new_d = None
        if new_d is not None:
            new_d.close()
            new_components.append(new_d)
        lsm_scanner.close()
        num_write = 0
        for n_d in new_components:
            for n_records in n_d.num_records():
                num_write += n_records
        write_cost = math.ceil(num_write / self.IO_unit)
        return new_components, num_write, write_cost

    def rename(self, rename_components: List[SpecialDiskComponent], level_to: int) -> List[SpecialDiskComponent]:
        # Rename
        # print("rename")
        new_components = []
        for r_c in rename_components:
            if r_c not in self.invalid_component():
                new_id = self._next_level_id(level_to)
                while r_c.is_reading():
                    time.sleep(0.5)
                r_c.rename(level_to, new_id)
                r_c.is_special = True
                new_components.append(r_c)
                if r_c not in self.safe_special_components:
                    self.safe_special_components.append(r_c)
                # print("new id=", r_c.min_id(), "-", r_c.max_id())
        return new_components

    def _add_components(self, new_components: List[DiskComponent]) -> None:
        for new_d in new_components:
            lv = new_d.min_id()
            if lv >= len(self._disks):
                self._disks.append([new_d, ])
            else:
                self._disks[lv] = [new_d, ] + self._disks[lv]

    def _remove_components(self, old_components: List[SpecialDiskComponent]) -> None:
        for old_d in old_components.copy():
            lv = old_d.min_id()
            removed = False
            try:
                self._disks[lv].remove(old_d)
                removed = True
                if len(self._disks[lv]) == 0:
                    self._disks[lv] = []
            except (ValueError, IndexError) as e:
                # This may happen for a renamed component. Its min_id had incremented.
                pass
            while not removed and lv > 0:
                lv = lv - 1
                try:
                    # This may happen for a renamed component. Its min_id had incremented.
                    self._disks[lv].remove(old_d)
                    if len(self._disks[lv]) == 0:
                        self._disks[lv] = []
                except (ValueError, IndexError):
                    pass

    def _components_info(self) -> []:
        info = []
        if self._mem is not None:
            info.append({
                "name": self._mem.name(),
                "size": self._mem.component_size(),
                "records": self._mem.num_records(),
                "min-key": "" if self._mem.min_key() is None else self._mem.min_key().hex(),
                "max-key": "" if self._mem.max_key() is None else self._mem.max_key().hex(),
            })
        for level in self._disks:
            level_info = []
            for d in level:
                level_info.append({
                    "name": d.name(),
                    "size": d.component_size(),
                    "records": d.num_records(),
                    "min-key": d.min_key().hex(),
                    "max-key": d.max_key().hex(),
                })
            info.append(level_info)
        return info

    def create_scanner(self, min_key: bytes, include_min: bool, max_key: bytes, include_max: bool, length, update) \
            -> (Optional[LSMScanner], int, int):
        scanners_scans = []
        if self._mem.num_records()[0] > 0 and \
                self.overlapping(min_key, include_min, max_key, include_max, self._mem.min_key(), self._mem.max_key()):
            if self._impl == 1:
                scanner = MemoryDBScanner(self._mem)
            else:
                scanner = MemoryScanner(self._mem)
            if min_key is not None:
                scanner.set_min_key(min_key, include_min)
            if max_key is not None:
                scanner.set_max_key(max_key, include_max)
            scanners_scans.append(scanner)
        disks = self._disk_components()
        results, total_read_cost, write_cost, num_cache_hit, num_r, new_merged_components = self.search_from_disk_baseG1(disks, min_key, include_min, max_key,
                                                                           include_max, length, scanners_scans)
        if results is not None:
            # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results, 5: new component
            return results, total_read_cost, write_cost, num_cache_hit, num_r, new_merged_components
        else:
            return None, None, None, None, None, None

    def get_ranges_by_partition_idx(self, range_idxs: list):
        list_ranges = []
        for idx in range_idxs:
            partition_min_key = self.partition.min_key
            partition_unit = self.partition.partition_unit
            range_min_key = idx * partition_unit + partition_min_key
            range_max_key = range_min_key + partition_unit - partition_min_key
            b_range_min_key = self.generator.bytes_from_number(range_min_key)
            b_range_max_key = self.generator.bytes_from_number(range_max_key)
            list_ranges.append((b_range_min_key, b_range_max_key, idx))
        return list_ranges

    def create_scanner_basePartition_save_cost(self, min_key: bytes, include_min: bool, max_key: bytes, include_max: bool, query_size, update):

        # update counters for half of the queries
        scanner, read_cost, r_write_cost, num_cache_hit, num_r, map_range_level \
            = self.create_scanner_read_only(min_key, include_min, max_key, include_max, query_size, update)
        # map_range_level
        # key: ranges, value: different levels. Save = # different levels - 1
        if self._mem is None:
            cur_mem_comp_size = 0
        else:
            cur_mem_comp_size = self._mem.num_records()[0]
        if len(map_range_level) > 0:
            range_idxs = self.partition.insert_range_read_cost(map_range_level, cur_mem_comp_size)
        else:
            range_idxs = []

        if (len(range_idxs) > 0):
            # create sentinel components
            list_ranges = self.get_ranges_by_partition_idx(range_idxs)
            # build sentinel components
            total_write_cost, total_num_write, new_merged_components = self.create_sentinel_component_scanners(
                list_ranges)
            # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results, 5: new_merged_components
            # 6: # of component, 7: # of level
            return scanner, read_cost, total_write_cost, num_cache_hit, num_r, new_merged_components
        else:
            # return query answer without building sentinel components
            # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results, 5: new_merged_components
            # 6: # of component, 7: # of level
            return scanner, read_cost, r_write_cost, num_cache_hit, num_r, []


    def create_scanner_basePartition_hit_frequency(self, min_key: bytes, include_min: bool, max_key: bytes, include_max: bool, query_size, update):

        # insert query into read counter, get top-k frequent ranges
        range_idxs = self.partition.insert_range_read(min_key, max_key, 1)
        # print("query = ", int(min_key), "-", int(max_key))
        if (len(range_idxs) > 0):
            # step-1 answer query
            scanner, read_cost, r_write_cost, num_cache_hit, num_r, map_range_level \
                = self.create_scanner_read_only(min_key, include_min, max_key, include_max, query_size, update)
            # step-2 build sentinel components
            # create sentinel components
            list_ranges = self.get_ranges_by_partition_idx(range_idxs)
            # build sentinel components
            total_write_cost, total_num_write, new_merged_components = self.create_sentinel_component_scanners(list_ranges)
            # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
            return scanner, read_cost, total_write_cost, num_cache_hit, num_r, new_merged_components
        else:
            # return query answer without building sentinel components
            # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
            scanner, read_cost, r_write_cost, num_cache_hit, num_r, map_range_level \
                = self.create_scanner_read_only(min_key, include_min, max_key, include_max, query_size, update)
            return scanner, read_cost, r_write_cost, num_cache_hit, num_r, []

    def create_scanner_read_only(self, min_key: bytes, include_min: bool, max_key: bytes, include_max: bool, length: int, update) \
            -> Optional[LSMScanner]:

        query_ranges = []
        query_ranges.append((min_key, max_key))

        scanners = []
        if self._mem.num_records()[0] > 0 and \
                self.overlapping(min_key, include_min, max_key, include_max, self._mem.min_key(), self._mem.max_key()):
            if self._impl == 1:
                scanner = MemoryDBScanner(self._mem)
            else:
                scanner = MemoryScanner(self._mem)
            if min_key is not None:
                scanner.set_min_key(min_key, include_min)
            if max_key is not None:
                scanner.set_max_key(max_key, include_max)
            scanners.append(scanner)
        if self.complementary_set is not None and \
                self.overlapping(min_key, include_min, max_key, include_max, self.complementary_set.min_key(), self.complementary_set.max_key()):
            scanner = MemoryScanner(self.complementary_set)
            if min_key is not None:
                scanner.set_min_key(min_key, include_min)
            if max_key is not None:
                scanner.set_max_key(max_key, include_max)
            scanners.append(scanner)
        disks = self.disk_components().copy()
        for d in disks:
            for tmp_min_key, tmp_max_key in query_ranges:
                if tmp_min_key == min_key:
                    include_min = True
                else:
                    include_min = False
                if tmp_max_key == max_key:
                    include_max = True
                else:
                    include_max = False
                is_oc = False
                for ridx in d.overlapping_range_indexes(tmp_min_key, include_min, tmp_max_key, include_max):
                    scanner = DiskScanner(d, ridx, self.IO_unit)
                    if min_key is not None:
                        scanner.set_min_key(tmp_min_key, include_min)
                    if max_key is not None:
                        scanner.set_max_key(tmp_max_key, include_max)
                    scanners.append(scanner)
                    is_oc = True
                if is_oc and d.is_special:
                    # count sp usage
                    # 0: c_name, 1: c_range, 2, 1
                    # str_sp = d.name() + "," + str(int(d.min_key())) + "-" + str(
                    #     int(d.max_key())) + "," + "1" + "\n"
                    # self.sp_usage.write(str_sp)

                    if d.min_key() <= tmp_min_key < tmp_max_key <= d.max_key():
                        # q_range is a subrange of sp's range
                        query_ranges.remove((tmp_min_key, tmp_max_key))
                    elif d.min_key() <= tmp_min_key < d.max_key() <= tmp_max_key:
                        # tmp_min_key = d.max_key()
                        query_ranges.remove((tmp_min_key, tmp_max_key))
                        tmp = int(d.max_key())
                        query_ranges.append((self.generator.bytes_from_number(tmp), tmp_max_key))
                    elif tmp_min_key <= d.min_key() < tmp_max_key <= d.max_key():
                        # tmp_max_key = d.min_key()
                        query_ranges.remove((tmp_min_key, tmp_max_key))
                        tmp = int(d.min_key())
                        query_ranges.append((tmp_min_key, self.generator.bytes_from_number(tmp)))
                    elif tmp_min_key < d.min_key() < d.max_key() < tmp_max_key:
                        # tmp_min_key - d.min_key(), d.max_key() - tmp_max_key
                        query_ranges.remove((tmp_min_key, tmp_max_key))
                        tmp_min = int(d.min_key())
                        tmp_max = int(d.max_key())
                        query_ranges.append((tmp_min_key, (self.generator.bytes_from_number(tmp_min))))
                        query_ranges.append(((self.generator.bytes_from_number(tmp_max)), tmp_max_key))
        if self.do_update_counter:
            self.num_query = 0
            self.do_update_counter = False
            # if do count levels for different ranges
            range_idxs = self.partition.get_ranges_by_range((min_key, max_key))
            list_ranges = self.get_ranges_by_partition_idx(range_idxs)
            map_range_level = {}  # key: ranges, value: different levels
            num_r = 0
            operational_components = set()
            if len(scanners) > 0:
                lsm_scanners = LSMScanner(scanners, length, self.IO_unit)
                lsm_scanners.open()
                while True:
                    d_name, ridx, key, data = lsm_scanners.next()
                    if key is None:
                        break
                    else:
                        # print("    {0} from {1}".format(key, d_name))
                        num_r += 1
                        # key_cname = str(int(key)) + " from " + str(d_name)
                        # self.query_results.write(key_cname)
                        # self.query_results.write('\n')
                        operational_components.add(d_name)
                        # count levels
                        tem = d_name.split("-")
                        d_min_id = tem[0]
                        for r_min_key, r_max_key, idx in list_ranges:
                            if r_min_key <= key <= r_max_key:
                                if (r_min_key, r_max_key, idx) not in map_range_level.keys():
                                    map_range_level[(r_min_key, r_max_key, idx)] = []
                                    map_range_level[(r_min_key, r_max_key, idx)].append(d_min_id)
                                else:
                                    current_levels = map_range_level[(r_min_key, r_max_key, idx)]
                                    if d_min_id not in current_levels:
                                        current_levels.append(d_min_id)
                                        map_range_level[(r_min_key, r_max_key, idx)] = current_levels

                # num_cache_hit = 0
                num_cache_hit, num_read_page, num_read_page_memory = self.update_cache(lsm_scanners)
                lsm_scanners.close()
                total_read_cost = num_read_page

                # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
                return lsm_scanners, total_read_cost, 0, num_cache_hit, num_r, map_range_level
            else:
                return None, None, None, None, None, None, None
        else:
            self.num_query += 1
            if self.num_query == self.do_update_counter_unit:
                self.do_update_counter = True
            map_range_level = {}  # key: ranges, value: different levels
            num_r = 0

            if len(scanners) > 0:
                lsm_scanners = LSMScanner(scanners, length, self.IO_unit)
                lsm_scanners.open()
                while True:
                    d_name, ridx, key, data = lsm_scanners.next()
                    if key is None:
                        break
                    else:
                        # print("    {0} from {1}".format(key, d_name))
                        num_r += 1
                        # key_cname = str(int(key)) + " from " + str(d_name)
                        # self.query_results.write(key_cname)
                        # self.query_results.write('\n')

                # num_cache_hit = 0
                num_cache_hit, num_read_page, num_read_page_memory = self.update_cache(lsm_scanners)
                lsm_scanners.close()
                total_read_cost = num_read_page

                # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
                return lsm_scanners, total_read_cost, 0, num_cache_hit, num_r, map_range_level
            else:
                return None, None, None, None, None

    def create_sentinel_component_scanners(self, list_ranges: list):

        total_write_cost = 0;
        total_num_write = 0;

        new_merged_components = []

        for range_min_key, range_max_key, idx in list_ranges:
            scanners_scans = []
            if self._mem.num_records()[0] > 0 and \
                    self.overlapping(range_min_key, True, range_max_key, True, self._mem.min_key(),
                                     self._mem.max_key()):
                scanner = MemoryScanner(self._mem)
                scanner.set_min_key(range_min_key, True)
                scanner.set_max_key(range_max_key, True)
                scanners_scans.append(scanner)
            disks = self._disk_components()
            find_from_single_level = True
            pre_min_id = None
            for d in disks:
                is_oc = False
                for ridx in d.overlapping_range_indexes(range_min_key, True, range_max_key, True):
                    scanner = DiskScanner(d, ridx, self.IO_unit)
                    scanner.set_min_key(range_min_key, True)
                    scanner.set_max_key(range_max_key, True)
                    scanners_scans.append(scanner)
                    is_oc = True
                if is_oc:
                    if d.is_special:
                        break
                    if pre_min_id is None:
                        pre_min_id = d.min_id()
                    else:
                        if pre_min_id != d.min_id():
                            find_from_single_level = False
            # if find from single level
            # this is range is in sentinel component
            # or all the range in mem_table + level 1
            if not find_from_single_level:
                # create sentinel components
                lsm_scanners = LSMScanner(scanners_scans, self.partition.partition_unit, self.IO_unit)
                write_cost, num_write, new_merged_component = self.insert_basePartition(range_min_key, range_max_key, lsm_scanners)
                new_merged_components += new_merged_component
                total_write_cost += write_cost
                total_num_write += num_write

        return total_write_cost, total_num_write, new_merged_components

    def search_from_disk_baseG1(self, disks: List[DiskComponent], min_key: bytes, include_min: bool, max_key: bytes,
                         include_max: bool, query_size, scanners_scans) -> (Optional[LSMScanner], int, int):
        total_ranges = 0
        find_from_single_level = True
        find_from_special_component = True
        pre_min_id = None
        group_1 = []  # disjoint oc
        group_2_name = []
        group_2 = []  # overlap oc
        for d in disks:
            num_ranges = len(d.key_ranges())
            total_ranges += num_ranges
            is_oc = False
            scanner_d = []
            for ridx in d.overlapping_range_indexes(min_key, include_min, max_key, include_max):
                scanner = DiskScanner(d, ridx, self.IO_unit)
                if min_key is not None:
                    scanner.set_min_key(min_key, include_min)
                if max_key is not None:
                    scanner.set_max_key(max_key, include_max)
                scanner_d.append(scanner)
                scanners_scans.append(scanner)
                is_oc = True
                # group1 and group2
            if is_oc:
                if d.is_special == False:
                    find_from_special_component = False
                else:
                    # count sp usage
                    # 0: idx, 1: c_name, 2: c_range, 3, 0
                    str_sp = d.name() + "," + str(int(d.min_key())) + "-" + str(
                        int(d.max_key())) + "," + "1" + "\n"
                    self.sp_usage.write(str_sp)
                remove_from_g1 = []
                for d_1 in group_1:
                    if self.is_unsafe(d_1, d):
                        # put d_1 into group2
                        group_2.append(d_1)
                        group_2_name.append(d_1.name())
                        remove_from_g1.append(d_1)
                        if d not in group_2:
                            group_2.append(d)
                            group_2_name.append(d.name())

                for d_rm_g1 in remove_from_g1:
                    group_1.remove(d_rm_g1)

                if d not in group_2:
                    for d_2 in group_2:
                        if self.is_unsafe(d_2, d):
                            group_2.append(d)
                            group_2_name.append(d.name())
                            break

                if (d not in group_2) and (d not in group_1):
                    if min_key <= d.min_min_key() < d.max_max_key() <= max_key:
                        d.is_special = True
                    group_1.append(d)
                if pre_min_id == None:
                    pre_min_id = d.min_id()
                else:
                    if pre_min_id != d.min_id():
                        find_from_single_level = False

        if (len(group_1) + len(group_2)) > 0:
            # print("create merged component")
            # self.query_results.write('create merged component\n')
            create_mc = False
            if len(group_2) > 0:
                # if len(group_2) > 0:
                create_mc = True
                if find_from_single_level or find_from_special_component:
                    create_mc = False

            if create_mc:
                # print("length of scanners = ", len(scanners_scans))
                # print("group 1 = ", len(group_1))
                # for c in group_1:
                #     print(c.name(), ", its ranges = ", c.key_ranges())
                # print("group 2 = ", len(group_2))
                # for c in group_2:
                #     print(c.name(), "its ranges = ", c.key_ranges())
                # print()

                # read from group 1
                group1_ranges = []
                for oc in group_1:
                    # group_1_name.append(oc.name())
                    for oc_min, oc_max in oc.key_ranges():
                        group1_ranges.append((oc_min, oc_max))

                group1_ranges = sorted(group1_ranges)
                # print("info of g1 ranges = ", group1_ranges)
                # print(group_1_name)
                lsm_scanners = LSMScanner(scanners_scans, query_size, self.IO_unit)
                # new_merged_components, read_cost, total_write_cost, num_cache_hit, num_r
                read_cost, write_cost, num_cache_hit, num_r, new_merged_components = self.insert_query_results_baseG1(min_key, max_key, lsm_scanners, group1_ranges, group_2_name)
                # read_cost, write_cost, num_cache_hit, num_r, new_merged_components = self.insert_query_results(min_key, max_key, lsm_scanners, group1_ranges, group_2_name)
                # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
                return lsm_scanners, read_cost, write_cost, num_cache_hit, num_r, new_merged_components
            else:
                lsm_scanners = LSMScanner(scanners_scans, query_size, self.IO_unit)
                lsm_scanners.open()
                num_r = 0
                # print("does not create new sp")
                while True:
                    d_name, ridx, key, data = lsm_scanners.next()
                    # print(" {0} from {1}".format(key, d_name))
                    if key is None:
                        break
                    else:
                        # print(" {0} from {1}".format(key, d_name))
                        # print("d_name = {0}, ridx = {1}, key = {2}".format(d_name, ridx, key))
                        num_r += 1
                        # key_cname = str(int(key)) + " from " + str(d_name)
                        # self.query_results.write(key_cname)
                        # self.query_results.write('\n')
                        # operational_components.add(d_name)
                # num_cache_hit = 0
                num_cache_hit, num_read_page, num_read_page_memory = self.update_cache(lsm_scanners)
                lsm_scanners.close()
                # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
                return lsm_scanners, num_read_page, 0, num_cache_hit, num_r, []
        else:
            print("query = [", min_key, "-", max_key, "]")
            return None, None, None, None, None, None

    # create sentinel components based on input (min_key,max_key)
    def insert_basePartition(self, min_key, max_key, scanners_scans: LSMScanner):

        new_merged_components = []

        scanners_scans.open()
        # metadata
        metadata_map = {}
        for scanner in scanners_scans.scanners():
            if scanner.is_memory == False:
                d = scanner.component()
                pos = scanner.get_pos()
                pos -= d.record_length()
                min_max_id = str(d.min_id()) + "-" + str(d.max_id())
                metadata_map[min_max_id] = pos

        num_r = 0
        num_new_special_component = 0
        num_write = 0
        while True:
            d_name, ridx, key, data = scanners_scans.next()
            # print(" {0} from {1}".format(key, d_name))
            if key is None:
                # if self._mem.num_records()[0] > 0:
                if self._mem_special.num_records()[0] > 0:
                    # f_d = self._flush(True)
                    f_d = self._flush_mem_special(True)
                    # for n_records in f_d.num_records():
                    #     total_write_cost += (n_records / self.IO_unit)
                    if f_d is None:
                        raise IOError("Failed to flush the memory component")
                    self._add_components([f_d, ])
                    self.safe_special_components[f_d.name()] = f_d
                    self.safe_special_components_names.add(f_d.name())
                    new_merged_components.append(f_d)
                    next_min, next_max = self._next_memory_component_ids()
                    del self._mem_special
                    self._mem_special = self._new_memory_component(next_min, next_max)
                    num_new_special_component += 1
                break
            else:
                if self._mem_special is None:
                    next_min, next_max = self._next_memory_component_ids()
                    self._mem_special = self._new_memory_component(next_min, next_max)
                num_r += 1
                if 'mem' in d_name:
                    # self._mem_special.write_key_data(key, data)
                    continue
                    # if self.complementart_set is None:
                    #     next_min, next_max = self._next_memory_component_ids()
                    #     self.complementart_set = self._new_memory_component(next_min, next_max)
                    # self.complementart_set.write_key_data(key, data)
                else:
                    self._mem_special.write_key_data(key, data)
                    num_write += 1

                if self._mem_special.is_full():
                    # f_d = self._flush(True)
                    f_d = self._flush_mem_special(True)
                    # for n_records in f_d.num_records():
                    #     total_write_cost += (n_records / self.IO_unit)
                    if f_d is None:
                        raise IOError("Failed to flush the memory component")
                    self._add_components([f_d, ])
                    self.safe_special_components[f_d.name()] = f_d
                    self.safe_special_components_names.add(f_d.name())
                    new_merged_components.append(f_d)
                    next_min, next_max = self._next_memory_component_ids()
                    del self._mem_special
                    self._mem_special = self._new_memory_component(next_min, next_max)
                    num_new_special_component += 1

        # update metadata
        invalid_components, fragment_components = self.update_metadata(min_key, max_key, scanners_scans, metadata_map,
                                                                       [])
        self.insert_component_into_cache(new_merged_components)
        num_cache_hit, num_read_page, num_read_page_memory = self.update_cache(scanners_scans)
        scanners_scans.close()
        read_cost = num_read_page
        total_write_cost = 0
        for n_d in new_merged_components:
            total_write_cost += (n_d.num_records()[0] / self.IO_unit)
        # delete invalid components
        if len(invalid_components) > 0:
            # delete from cache
            for invalid_c in invalid_components:
                self.delete_page_by_component(invalid_c)
            self._remove_components(invalid_components)
            for i in range(len(invalid_components)):
                md: AbstractDiskComponent = invalid_components[i]
                # print(md.name())
                md.remove_files()
                del md

        if len(new_merged_components) == 1:
            new_merged_components[0]._min_key = min_key
            new_merged_components[0]._max_key = max_key

        return total_write_cost, num_write, new_merged_components

    def insert_query_results(self, min_key, max_key,scanners_scans: LSMScanner, group1_ranges: List[tuple], group_2_name: List[str]) -> (List[AbstractDiskComponent], int, int, int):

        new_merged_components = []
        total_write_cost = 0
        scanners_scans.open()
        # metadata
        metadata_map = {}
        for scanner in scanners_scans.scanners():
            if scanner.is_memory == False and scanner.component().name() in group_2_name:
                d = scanner.component()
                pos = scanner.get_pos()
                pos -= d.record_length()
                min_max_id = str(d.min_id()) + "-" + str(d.max_id())
                metadata_map[min_max_id] = pos

        num_r = 0
        num_new_special_component = 0
        while True:
            d_name, ridx, key, data = scanners_scans.next()
            # print(" {0} from {1}".format(key, d_name))
            if key is None:
                # if self._mem.num_records()[0] > 0:
                break
            else:
                if self._mem_special is None:
                    next_min, next_max = self._next_memory_component_ids()
                    self._mem_special = self._new_memory_component(next_min, next_max)
                self._mem_special.write_key_data(key, data)
                num_r += 1

                if self._mem_special.is_full():
                    # f_d = self._flush(True)
                    f_d = self._flush_mem_special(True)
                    # for n_records in f_d.num_records():
                    #     total_write_cost += (n_records / self.IO_unit)
                    if f_d is None:
                        raise IOError("Failed to flush the memory component")
                    self._add_components([f_d, ])
                    self.safe_special_components[f_d.name()] = f_d
                    self.safe_special_components_names.add(f_d.name())
                    new_merged_components.append(f_d)
                    next_min, next_max = self._next_memory_component_ids()
                    # del self._mem
                    # self._mem = self._new_memory_component(next_min, next_max)
                    del self._mem_special
                    self._mem_special = self._new_memory_component(next_min, next_max)
                    num_new_special_component += 1

        #update metadata
        invalid_components, fragment_components = self.update_metadata(min_key, max_key, scanners_scans, metadata_map, group_2_name)

        self.insert_component_into_cache(new_merged_components)
        num_cache_hit, num_read_page, num_read_page_memory = self.update_cache(scanners_scans)
        scanners_scans.close()
        read_cost = num_read_page
        total_write_cost = 0
        for n_d in new_merged_components:
            total_write_cost += (n_d.num_records()[0] / self.IO_unit)
        # delete invalid components
        if len(invalid_components) > 0:
            # delete from cache
            for invalid_c in invalid_components:
                self.delete_page_by_component(invalid_c)
            self._remove_components(invalid_components)
            for i in range(len(invalid_components)):
                md: AbstractDiskComponent = invalid_components[i]
                # print(md.name())
                md.remove_files()
                del md

        # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
        return read_cost, total_write_cost, num_cache_hit, num_r, new_merged_components

    def insert_query_results_baseG1(self, min_key, max_key,scanners_scans: LSMScanner, group1_ranges: List[tuple], group_2_name: List[str]) -> (List[AbstractDiskComponent], int, int, int):

        new_merged_components = []
        total_write_cost = 0
        scanners_scans.open()
        # metadata
        metadata_map = {}
        for scanner in scanners_scans.scanners():
            if scanner.is_memory == False and scanner.component().name() in group_2_name:
                d = scanner.component()
                pos = scanner.get_pos()
                pos -= d.record_length()
                min_max_id = str(d.min_id()) + "-" + str(d.max_id())
                metadata_map[min_max_id] = pos
        # print(metadata_map)
        # print()

        g1_min = -1
        if len(group1_ranges) > 0:
            idx_of_g1 = 0
            g1_min, g1_max = group1_ranges[idx_of_g1]

        num_r = 0
        num_new_special_component = 0
        num_write = 0
        while True:
            d_name, ridx, key, data = scanners_scans.next()
            # print(" {0} from {1}".format(key, d_name))
            if key is None:
                # if self._mem.num_records()[0] > 0:
                if self._mem_special.num_records()[0] > 0:
                    # f_d = self._flush(True)
                    f_d = self._flush_mem_special(True)
                    # for n_records in f_d.num_records():
                    #     total_write_cost += (n_records / self.IO_unit)
                    if f_d is None:
                        raise IOError("Failed to flush the memory component")
                    self._add_components([f_d, ])
                    self.safe_special_components[f_d.name()] = f_d
                    self.safe_special_components_names.add(f_d.name())
                    new_merged_components.append(f_d)
                    next_min, next_max = self._next_memory_component_ids()
                    # del self._mem
                    # self._mem = self._new_memory_component(next_min, next_max)
                    del self._mem_special
                    self._mem_special = self._new_memory_component(next_min, next_max)
                    num_new_special_component += 1
                break
            else:
                if self._mem_special is None:
                    next_min, next_max = self._next_memory_component_ids()
                    self._mem_special = self._new_memory_component(next_min, next_max)
                num_r += 1
                # if 'mem' in d_name:
                #     # self._mem_special.write_key_data(key, data)
                #     # print(key, " write by mem")
                #     continue
                # elif d_name in self.safe_special_components:
                #     continue
                if g1_min == -1:
                    self._mem_special.write_key_data(key, data)
                    num_write+=1
                    # print(key, " write by g1_min == -1")
                elif key < g1_min:
                    self._mem_special.write_key_data(key, data)
                    num_write += 1
                    # print(key, " write by key < g1_min")
                elif key == g1_max:
                    # if self._mem.num_records()[0] > 0:
                    if self._mem_special.num_records()[0] > 0:
                        # f_d = self._flush(True)
                        f_d = self._flush_mem_special(True)
                        # for n_records in f_d.num_records():
                        #     total_write_cost += (n_records / self.IO_unit)
                        if f_d is None:
                            raise IOError("Failed to flush the memory component")
                        self._add_components([f_d, ])
                        self.safe_special_components[f_d.name()] = f_d
                        self.safe_special_components_names.add(f_d.name())
                        new_merged_components.append(f_d)
                        next_min, next_max = self._next_memory_component_ids()
                        # del self._mem
                        # self._mem = self._new_memory_component(next_min, next_max)
                        del self._mem_special
                        self._mem_special = self._new_memory_component(next_min, next_max)
                        num_new_special_component += 1
                    idx_of_g1 += 1
                    if len(group1_ranges) > idx_of_g1:
                        g1_min, g1_max = group1_ranges[idx_of_g1]
                elif key > g1_max:
                    self._mem_special.write_key_data(key, data)
                    num_write += 1
                    # print(key, " write by key > g1_max")

                if self._mem_special.is_full():
                    # f_d = self._flush(True)
                    f_d = self._flush_mem_special(True)
                    # for n_records in f_d.num_records():
                    #     total_write_cost += (n_records / self.IO_unit)
                    if f_d is None:
                        raise IOError("Failed to flush the memory component")
                    self._add_components([f_d, ])
                    self.safe_special_components[f_d.name()] = f_d
                    self.safe_special_components_names.add(f_d.name())
                    new_merged_components.append(f_d)
                    next_min, next_max = self._next_memory_component_ids()
                    # del self._mem
                    # self._mem = self._new_memory_component(next_min, next_max)
                    del self._mem_special
                    self._mem_special = self._new_memory_component(next_min, next_max)
                    num_new_special_component += 1

        #update metadata
        invalid_components, fragment_components = self.update_metadata(min_key, max_key, scanners_scans, metadata_map, group_2_name)
        # num_cache_hit = 0
        # print("info of new sp")
        # for sp in new_merged_components:
        #     print(sp.name(), ", key ranges = ", sp.key_ranges())

        self.insert_component_into_cache(new_merged_components)
        num_cache_hit, num_read_page, num_read_page_memory = self.update_cache(scanners_scans)
        scanners_scans.close()
        read_cost = num_read_page
        total_write_cost = 0
        for n_d in new_merged_components:
            total_write_cost += (n_d.num_records()[0] / self.IO_unit)
        # delete invalid components
        if len(invalid_components) > 0:
            # delete from cache
            for invalid_c in invalid_components:
                self.delete_page_by_component(invalid_c)
            self._remove_components(invalid_components)
            for i in range(len(invalid_components)):
                md: AbstractDiskComponent = invalid_components[i]
                # print(md.name())
                md.remove_files()
                del md

        # 0: scanner, 1: search read, 2: search write, 3: num_cache_hit, 4: num_results
        return read_cost, total_write_cost, num_cache_hit, num_r, new_merged_components

    def update_metadata(self, min_key, max_key, scanners_scans: LSMScanner, metadata_map: map, group_2_name: List[str]) -> (Optional[List[SpecialDiskComponent]], Optional[List[SpecialDiskComponent]]):
        fragment_components = []
        invalid_components = []
        # metadata

        for scanner in scanners_scans.scanners():
            if scanner.is_memory == False and scanner.component().is_special == False and scanner.num_read_records > 0:
                scanner: DiskScanner
                if scanner.num_read_records > 0 and scanner.component().name() in group_2_name:
                    # print(scanner.component().name(), "update metadata")
                    d = scanner.component()
                    pos = scanner.get_pos()
                    pos -= d.record_length()
                    # print("d_min_id = ", d.min_id(), ", d_max_id = ", d.max_id())

                    target = self.find_pos(d, pos)
                    while target > scanner._max_key:
                        pos -= d.record_length()
                        target = self.find_pos(d, pos)
                    min_max_id = str(d.min_id()) + "-" + str(d.max_id())
                    pre_pos = metadata_map[min_max_id]
                    start_target = self.find_pos(d, pre_pos)
                    v_ranges = []
                    v_records = []
                    d_min_max_keys = d.key_ranges()
                    current_num_records = 0
                    record_length = self._mem.record_length()
                    max_records_size = self._mem.max_records()
                    for i in range(len(d_min_max_keys)):
                        cur_tuple = d_min_max_keys[i]
                        d_cur_num_records = d.num_records()[i]
                        # d_cur_com_size = d.component_sizes()[i]
                        cur_v_min_key = cur_tuple[0]
                        cur_v_max_key = cur_tuple[1]
                        if cur_v_min_key <= target <= cur_v_max_key:
                            min_pos = 0
                            tem_min_target = self.find_pos(d, min_pos)
                            while tem_min_target != cur_v_min_key:
                                min_pos += d.record_length()
                                tem_min_target = self.find_pos(d, min_pos)
                            if start_target == cur_v_min_key:
                                # min_key = next of target
                                pos += d.record_length()
                                if target != cur_v_max_key:
                                    target = self.find_pos(d, pos)
                                    v_record = d_cur_num_records - int((pos - min_pos) / d.record_length())
                                    v_min_key = target
                                    v_max_key = cur_v_max_key
                                    # v_record = d_cur_num_records - int((pos - pre_pos) / d.record_length())
                                    v_range = (v_min_key, v_max_key)
                                    v_ranges.append(v_range)
                                    v_records.append(v_record)
                            elif target == cur_v_max_key:
                                # max_may = pre of pre_pos
                                v_min_key = cur_v_min_key
                                # v_max_key = self.find_pos(d, pre_pos)
                                v_max_key = start_target
                                v_record = int((pre_pos - min_pos) / d.record_length())
                                if v_max_key >= scanner._min_key:
                                    pre_pos -= d.record_length()
                                    v_max_key = self.find_pos(d, pre_pos)
                                v_range = (v_min_key, v_max_key)
                                v_ranges.append(v_range)
                                v_records.append(v_record)
                            else:
                                # virtual split
                                v_min_key_1 = cur_v_min_key
                                # v_max_key_1 = self.find_pos(d, pre_pos)
                                v_max_key_1 = start_target
                                v_record_1 = int((pre_pos - min_pos) / d.record_length())
                                if v_max_key_1 >= scanner._min_key:
                                    pre_pos -= d.record_length()
                                    v_max_key_1 = self.find_pos(d, pre_pos)
                                v_range_1 = (v_min_key_1, v_max_key_1)
                                v_ranges.append(v_range_1)
                                v_records.append(v_record_1)
                                pos += d.record_length()
                                if target != cur_v_max_key:
                                    target = self.find_pos(d, pos)
                                    v_record = d_cur_num_records - int((pos - min_pos) / d.record_length())
                                    v_min_key = target
                                    v_max_key = cur_v_max_key
                                    # v_record = d_cur_num_records - int((pos - pre_pos) / d.record_length())
                                    v_range = (v_min_key, v_max_key)
                                    v_ranges.append(v_range)
                                    v_records.append(v_record)
                        else:
                            v_ranges.append(cur_tuple)
                            v_records.append(d_cur_num_records)
                            # print()
                    self.update_virtual_metadata(d, v_ranges, v_records)
                    num_of_c = 0
                    for v_num in d.num_records():
                        num_of_c += v_num
                    if num_of_c == 0:
                        # d is empty
                        invalid_components.append(d)
                        if d in self.normal_components:
                            self.normal_components.remove(d)
                        elif d.name() in self.safe_special_components:
                            self.safe_special_components.pop(d.name())

        return invalid_components, fragment_components

    def find_pos(self, d: SpecialDiskComponent, pos: int) -> bytes:
        path = d.get_binary_path()
        if os.path.isfile(path):
            f = open(path, "rb")
            f.seek(pos)
            rKey = f.read(d.key_length())
        f.close()
        return rKey

    def update_virtual_metadata(self, d: SpecialDiskComponent, v_ranges: List[tuple], v_records: List[int]):
        d._v_ranges = v_ranges
        d._v_records = v_records
        # d._v_sizes = [self._record_len * vr for vr in v_records]
        d._save_meta()

    def remove_fragment(self) -> (int, int):
        total_num_write = 0
        total_write_cost = 0
        for level in range(len(self._disks)):
            if level == 0:
                continue
            else:
                current_level_components = self._disks[level]
                new_current_level_components = []
                idx = 0
                while idx < len(current_level_components):
                    # print("idx = ", idx)
                    d_c = current_level_components[idx]
                    if len(d_c.key_ranges()) > self._size_t:
                        # rewrite current d_c
                        old_components = [d_c]
                        new_components, num_write, write_cost = self.rewrite_components(old_components)
                        total_num_write += num_write
                        total_write_cost += write_cost
                        new_current_level_components += new_components
                        # all_old_components.append(d_c)
                        # all_new_components += new_components
                    else:
                        new_current_level_components.append(d_c)
                    idx += 1
            self._disks[level] = new_current_level_components
        return total_num_write, total_write_cost

    def rewrite_components(self, old_components: List[SpecialDiskComponent]) -> (List[SpecialDiskComponent], int, int):
        scanners = []
        for d in old_components:
            for ridx in d.overlapping_range_indexes(None, True, None, True):
                scanners.append(DiskScanner(d, ridx))
        lsm_scanner = LSMScanner(scanners, -1)
        rewrite_records = []
        lsm_scanner.open()
        while True:
            d_name, ridx, key, data = lsm_scanner.next()
            if key is None:
                break
            else:
                rewrite_records.append((key, data))
        lsm_scanner.close()
        del lsm_scanner
        for i in range(len(old_components)):
            md: AbstractDiskComponent = old_components[i]
            md.remove_files()
            del md
        new_components = []
        new_d = None
        idx = 0
        # print("length of rewrite_records = ", len(rewrite_records))
        if len(old_components) > 1:
            is_special = False
        else:
            is_special = old_components[0].is_special
        for tuple in rewrite_records:
            if new_d is None:
                min_id = old_components[0].min_id()
                max_id = old_components[idx].max_id()
                # print("new min_id = ", min_id, "new max_id = ", max_id)
                idx += 1
                new_d = self._new_disk_component(min_id, max_id, is_special)
                new_d.open()
            new_d.write_key_data(tuple[0], tuple[1])
            # if new_d.num_records() == max_records:
            if new_d.actual_num_records() == self._mem.max_records():
                new_d.close()
                new_components.append(new_d)
                new_d = None
        if new_d is not None:
            new_d.close()
            new_components.append(new_d)
        total_num_write = 0
        total_write_cost = 0
        for n_d in new_components:
            num_write = 0
            for n_records in n_d.num_records():
                num_write += n_records
            write_cost = math.ceil(num_write / self.IO_unit)
            total_write_cost += write_cost
        total_write_cost += len(new_components)
        return new_components, total_num_write, total_write_cost

    def is_unsafe(self, d1: SpecialDiskComponent, d2: SpecialDiskComponent) -> bool:
        for min1, max1 in d1.key_ranges():
            for min2, max2 in d2.key_ranges():
                if self.overlapping(min1, True, max1, True, min2, max2):
                    return True
        return False

    def search(self, key: bytes) -> (int, Optional[bytes], Optional[bytes]):
        if self._mem.num_records()[0] > 0 and self._mem.min_key() <= key <= self._mem.max_key():
            rkey, rdata = self._mem.get_record(key)
            if rkey is not None:
                return -1, rkey, rdata, 1
        disks = self._disk_components().copy()
        num_operational_component = 1
        for idx in range(0, len(disks)):
            d = disks[idx]
            if d.is_in_range(key):
                rkey, rdata = d.get_record(key)
                if rkey is None:
                    num_operational_component += 1
                else:
                    return idx, rkey, rdata, num_operational_component
                # if rkey is not None:
                #     return idx, rkey, rdata
        return -1, None, None, num_operational_component
