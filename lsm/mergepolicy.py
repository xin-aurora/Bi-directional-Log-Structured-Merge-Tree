import collections
import json
import math
import os
import random
import sys
from collections import OrderedDict
from itertools import permutations
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

from .component import AbstractDiskComponent, SpecialDiskComponent


class MergePolicy:
    """ Base class for LSM merge policy. """

    @classmethod
    def from_config(cls, config: Union[dict, str], tree):
        """ Create a merge policy from a configuration object as string or dict. """
        if isinstance(config, dict):
            policy_info = config
        elif isinstance(config, str):
            try:
                policy_info = json.loads(config)
            except json.JSONDecodeError as e:
                raise ValueError("Error parsing {0}: {1}".format(config, e))
            if type(policy_info) != dict:
                raise TypeError("{0} cannot be converted to dict".format(config))
        else:
            raise TypeError("{0} {1} is not supported".format(config, type(config)))
        from .tree import LSMTree
        if tree is None or not isinstance(tree, LSMTree):
            raise TypeError("tree {0} is not an instance of {1}".format(type(tree), LSMTree))
        pname = policy_info.get(LSMTree.PROP_MERGE_POLICY, "")
        props = policy_info.get(LSMTree.PROP_MERGE_POLICY_PROPERTIES, {})
        try:
            if pname == LeveledPolicy.policy_name():
                return LeveledPolicy(
                    tree,
                    int(props[LeveledPolicy.PROP_COMPONENTS_0]),
                    int(props[LeveledPolicy.PROP_COMPONENTS_1]),
                    float(props[LeveledPolicy.PROP_RATIO]),
                    str(props[LeveledPolicy.PROP_PICK]),
                    int(props[LeveledPolicy.PROP_OVERFLOW]))
            elif pname == LeveledNPolicy.policy_name():
                return LeveledNPolicy(
                    tree,
                    int(props[LeveledNPolicy.PROP_LEVEL_N]),
                    int(props[LeveledPolicy.PROP_COMPONENTS_0]),
                    int(props[LeveledPolicy.PROP_COMPONENTS_1]),
                    float(props[LeveledPolicy.PROP_RATIO]),
                    str(props[LeveledPolicy.PROP_PICK]))
            elif pname == TieredLeveledPolicy.policy_name():
                return TieredLeveledPolicy(
                    tree,
                    int(props[TieredLeveledPolicy.PROP_MAX_TIERS]),
                    int(props[LeveledPolicy.PROP_COMPONENTS_0]),
                    int(props[LeveledPolicy.PROP_COMPONENTS_1]),
                    float(props[LeveledPolicy.PROP_RATIO]),
                    str(props[LeveledPolicy.PROP_PICK]))
            elif pname == LazyLeveledPolicy.policy_name():
                return LazyLeveledPolicy(
                    tree,
                    int(props[LazyLeveledPolicy.PROP_COMPONENTS_0]),
                    int(props[LazyLeveledPolicy.PROP_COMPONENTS_1]),
                    float(props[LazyLeveledPolicy.PROP_RATIO]),
                    str(props[LazyLeveledPolicy.PROP_PICK]),
                    int(props[LazyLeveledPolicy.PROP_OVERFLOW]),
                    int(props[LazyLeveledPolicy.PROP_NC_T]),
                    int(props[LazyLeveledPolicy.PROP_SSC_T]),
                    bool(props[LazyLeveledPolicy.PROP_COMPONENTARY]),
                    bool(props[LazyLeveledPolicy.PROP_REBUILDING]))
            else:
                if len(pname) == 0 or pname == NoMergePolicy.policy_name():
                    return NoMergePolicy(tree)
                elif pname == ConstantPolicy.policy_name():
                    return ConstantPolicy(
                        tree,
                        int(props[ConstantPolicy.PROP_NUM_COMPONENTS]))
                elif pname == BigtablePolicy.policy_name():
                    return BigtablePolicy(
                        tree,
                        int(props[BigtablePolicy.PROP_NUM_COMPONENTS]))
                elif pname == ExploringPolicy.policy_name():
                    return ExploringPolicy(
                        tree,
                        int(props[ExploringPolicy.PROP_MIN_COMPONENTS]),
                        int(props[ExploringPolicy.PROP_MAX_COMPONENTS]),
                        float(props[ExploringPolicy.PROP_RATIO]))
                elif pname == SizeTieredPolicy.policy_name():
                    return SizeTieredPolicy(
                        tree,
                        int(props[SizeTieredPolicy.PROP_MIN_COMPONENTS]),
                        int(props[SizeTieredPolicy.PROP_MAX_COMPONENTS]),
                        float(props[SizeTieredPolicy.PROP_LOW_BUCKET]),
                        float(props[SizeTieredPolicy.PROP_HIGH_BUCKET]),
                        props[SizeTieredPolicy.PROP_MIN_COMPONENT_SIZE])
                elif pname == TieredPolicy.policy_name():
                    return TieredPolicy(
                        tree,
                        int(props[TieredPolicy.PROP_RATIO]),
                        float(props[TieredPolicy.PROP_SCALE]))
                elif pname == BinomialPolicy.policy_name():
                    return BinomialPolicy(
                        tree,
                        int(props[BinomialPolicy.PROP_NUM_COMPONENTS]))
                elif pname == MinLatencyPolicy.policy_name():
                    return MinLatencyPolicy(
                        tree,
                        int(props[MinLatencyPolicy.PROP_NUM_COMPONENTS]))
                else:
                    raise NotImplementedError("Unsupported merge policy: {0}".format(pname))
        except (KeyError, ValueError) as e:
            raise IOError("Error loading {0}: {1}".format(os.path.abspath(config_path), e))

    @classmethod
    def from_config_file(cls, config_path: str, tree):
        """ Create a merge policy from configuration file. """
        if not os.path.isfile(config_path):
            raise FileNotFoundError("{0} does not exist".format(os.path.abspath(config_path)))
        with open(config_path, "r") as cfgf:
            return MergePolicy.from_config(cfgf.read(), tree)

    @classmethod
    def from_base_dir(cls, base_dir: str, tree):
        if not os.path.isdir(base_dir):
            raise FileNotFoundError("{0} does not exist".format(os.path.abspath(base_dir)))
        return MergePolicy.from_config(os.path.join(base_dir, "config.json"), tree)

    def _validate_tree(self, tree) -> bool:
        from .tree import LSMTree
        if tree is None or not isinstance(tree, LSMTree):
            raise TypeError("tree {0} is not an instance of {1}".format(type(tree), LSMTree))
        self._lsmtree: LSMTree = tree
        return True

    def __init__(self, tree):
        self._lsmtree = None
        self._valid = self._validate_tree(tree)
        self._props = {}

    @staticmethod
    def policy_name() -> str:
        return ""

    @staticmethod
    def default_properties() -> dict:
        return {}

    def is_valid(self) -> bool:
        return self._valid

    def properties(self) -> Dict[str, Any]:
        return self._props

    def lsm_tree(self):
        return self._lsmtree

    def get_mergable_components(self, is_flush: bool) -> (List[AbstractDiskComponent], int):
        """
        Args:
            is_flush: Whether the merge is triggered by a flush.
        Returns:
            (
                List of components to be merged.
                Maximum number of records allowed in every new component (< 1 is unlimited).
            )
        """
        return [], -1

    def sort_components(self) \
            -> Union[List[AbstractDiskComponent], List[List[AbstractDiskComponent]]]:
        return []


class StackPolicy(MergePolicy):
    """ Base class for stack-based merge policy. """

    def _validate_tree(self, tree) -> bool:
        from .tree import StackLSMTree
        if tree is None or not isinstance(tree, StackLSMTree):
            raise TypeError("tree {0} is not an instance of {1}".format(type(tree), StackLSMTree))
        self._lsmtree: StackLSMTree = tree
        return True

    def __init__(self, tree):
        super().__init__(tree)
        self._props = {}

    def get_mergable_components(self, is_flush: bool) -> (List[AbstractDiskComponent], int):
        """
        Args:
            is_flush: Whether the merge is triggered by a flush.
        Returns:
            (
                List of components to be merged.
                Maximum number of records allowed in every new component (< 1 is unlimited).
            )
        """
        components = self._lsmtree.disk_components().copy()
        if len(components) < 2:
            return [], -1
        return self._get_mergable_components(is_flush, components), -1

    def _get_mergable_components(self, is_flush: bool, components: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        """
        Args:
            is_flush: Whether the merge is triggered by a flush.
            components: Possible components to be merged
        Returns:
                List of components to be merged.
        """
        return []

    def sort_components(self) -> List[AbstractDiskComponent]:
        to_sort = []
        for d in self._lsmtree.disk_components():
            to_sort.append((d.min_id(), d))
        to_sort.sort(key=itemgetter(0), reverse=True)
        return [d for i, d in to_sort]


class LeveledPolicy(MergePolicy):
    """ Base class for the standard Leveled merge policy. """
    PROP_COMPONENTS_0: str = "components_0"
    PROP_COMPONENTS_1: str = "components_1"
    PROP_RATIO: str = "ratio"
    PROP_PICK: str = "pick"
    PROP_OVERFLOW: str = "overflow"
    PICK_MIN_OVERLAP: str = "min-overlap"
    PICK_MAX_OVERLAP: str = "max-overlap"
    PICK_MIN_RANGE: str = "min-range"
    PICK_MAX_RANGE: str = "max-range"
    PICK_FIRST: str = "first"
    PICK_LAST: str = "last"
    PICK_OLDEST: str = "oldest"
    PICK_NEWEST: str = "newest"
    PICK_RANDOM: str = "random"
    PROP_LENGTH_T: str = "length-threshold"
    PROP_FRAGMENT_T: str = "fragment-threshold"
    PROP_NC_T: str = "nc-threshold"
    PROP_SSC_T: str = "safe-sp-threshold"
    PROP_USC_T: str = "unsafe-sp-threshold"
    PROP_WITH_UNSAFE: str = "with-unsafe-sp"
    PROP_PARTIAL_MERGE: str = "do-partial-merge"
    PROP_COMPONENTARY: str = "have-componentary-set"
    PROP_REBUILDING: str = "do-rebuilding"

    PICK_OPTIONS = (PICK_MIN_OVERLAP, PICK_MAX_OVERLAP, PICK_MIN_RANGE, PICK_MAX_RANGE, PICK_FIRST, PICK_LAST,
                    PICK_OLDEST, PICK_NEWEST, PICK_RANDOM)

    def _validate_tree(self, tree) -> bool:
        from .tree import LeveledLSMTree
        if tree is None or not isinstance(tree, LeveledLSMTree):
            raise TypeError("tree {0} is not an instance of {1}".format(type(tree), LeveledLSMTree))
        else:
            self._lsmtree: LeveledLSMTree = tree
            return True

    def __init__(self, tree, components_0: int, components_1: int, ratio: float, pick: str, overflow: int):
        super().__init__(tree)
        self._lv0 = components_0
        self._lv1 = components_1
        self._ratio = ratio
        self._pick = pick.lower()
        self._overflow = overflow
        if self._lv0 < 1:
            self._valid = False
            raise ValueError("components_0 ({0}) must be at least 1".format(self._lv0))
        if self._lv1 < 2:
            self._valid = False
            raise ValueError("components_1 ({0}) must be at least 2".format(self._lv1))
        if self._ratio < 1.0:
            self._valid = False
            raise ValueError("ratio ({0}) must be at least 1.0".format(self._ratio))
        if self._pick not in LeveledPolicy.PICK_OPTIONS:
            self._valid = False
            raise ValueError("pick ({0}) is invalid, valid options are ({1})"
                             .format(self._pick, ", ".join(LeveledPolicy.PROP_PICK_OPTIONS)))
        if self._overflow < 0:
            self._valid = False
            raise ValueError("overflow ({0}) must be at least 0".format(self._overflow))
        self._props[LeveledPolicy.PROP_COMPONENTS_0] = self._lv0
        self._props[LeveledPolicy.PROP_COMPONENTS_1] = self._lv1
        self._props[LeveledPolicy.PROP_RATIO] = self._ratio
        self._props[LeveledPolicy.PROP_PICK] = self._pick
        self._props[LeveledPolicy.PROP_OVERFLOW] = self._overflow

    @staticmethod
    def policy_name() -> str:
        return "leveled"

    @staticmethod
    def default_properties() -> dict:
        return {
            LeveledPolicy.PROP_COMPONENTS_0: 10,
            LeveledPolicy.PROP_COMPONENTS_1: 10,
            LeveledPolicy.PROP_RATIO: 10,
            LeveledPolicy.PROP_PICK: LeveledPolicy.PICK_MIN_OVERLAP,
            LeveledPolicy.PROP_OVERFLOW: 1,
        }

    def components_0(self) -> int:
        return self._lv0

    def components_1(self) -> int:
        return self._lv1

    def ratio(self) -> float:
        return self._ratio

    def pick(self) -> str:
        return self._pick

    def overflow(self) -> int:
        return self._overflow

    @staticmethod
    def is_overlapping(d1: AbstractDiskComponent, d2: AbstractDiskComponent) -> bool:
        from .tree import LeveledLSMTree
        for min1, max1 in d1.key_ranges():
            for min2, max2 in d2.key_ranges():
                if LeveledLSMTree.overlapping(min1, True, max1, True, min2, max2):
                    return True
        return False

    @staticmethod
    def get_overlapping_components(picked: (List[AbstractDiskComponent], Tuple[AbstractDiskComponent]),
                                   next_level: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        overlaps = set()
        for d in next_level:
            for p in picked:
                if LeveledPolicy.is_overlapping(p, d):
                    overlaps.add(d)
        overlaps = list(overlaps)
        return overlaps

    @staticmethod
    def get_by_time(components: List[AbstractDiskComponent], num_picked: int, oldest: bool = True) \
            -> List[AbstractDiskComponent]:
        """ Return a component in a level with smallest (oldest) or largest (newest) max_id. """
        cnt = len(components)
        if cnt < num_picked:
            return []
        if cnt == num_picked:
            return components
        ts = []
        lv = -1
        for d in components:
            if lv == -1:
                lv = d.min_id()
            elif d.min_id() != lv:
                raise ValueError("Components are not in the same level: {0} vs {1}"
                                 .format(components[0].name(), d.name()))
            ts.append((d.max_id(), d))
        ts.sort(key=itemgetter(0), reverse=not oldest)
        return [d for max_id, d in ts[0:num_picked]], []

    @staticmethod
    def get_by_range(components: List[AbstractDiskComponent], num_picked: int, smallest: bool = True) \
            -> List[AbstractDiskComponent]:
        """ Return a component in a level with smallest or largest key range. """

        def subtract(b1: bytes, b2: bytes) -> bytes:
            return bytes([b2[i] - b1[i] for i in range(len(b1))])

        cnt = len(components)
        if cnt < num_picked:
            return []
        if cnt == num_picked:
            return components
        picked = [(subtract(d.max_max_key(), d.min_min_key()), d) for d in components[0:num_picked]]
        lv = components[0].min_id()
        for i in range(1, num_picked):
            if components[i].min_id() != lv:
                raise ValueError("Components are not in the same level: {0} vs {1}"
                                 .format(components[0].name(), components[i].name()))
        picked.sort(key=itemgetter(0), reverse=not smallest)
        for d in components[num_picked:]:
            r = subtract(d.max_max_key(), d.min_min_key())
            last_r, last_d = picked[-1]
            if (r == last_r and d.max_id() < last_d.max_id()) or (smallest and r < last_r) or \
                    (not smallest and r > last_r):
                picked[-1] = (r, d)
                picked.sort(key=itemgetter(0), reverse=not smallest)
        picked = [d for r, d in picked]
        return [d for d in components if d in picked]

    @staticmethod
    def get_random(components: List[AbstractDiskComponent], num_picked: int) -> List[AbstractDiskComponent]:
        cnt = len(components)
        if cnt < num_picked:
            return []
        if cnt == num_picked:
            return components
        return [components[i] for i in sorted(random.sample(range(cnt), num_picked))]

    def _get_min_max_overlapping_components(self, froms: List[AbstractDiskComponent], tos: List[AbstractDiskComponent],
                                            num_picked: int, is_min: bool) -> List[AbstractDiskComponent]:
        from_cnt = len(froms)
        if from_cnt < num_picked:
            raise ValueError("No thing to pick")
        if len(tos) == 0:
            # No overlapping, return the oldest only
            return LeveledPolicy.get_by_time(froms, num_picked, True)
        if froms[0].min_id() + 1 != tos[0].min_id():
            raise ValueError("Invalid merge from level {0} to {1}".format(froms[0].min_id(), tos[0].min_id()))
        if from_cnt == num_picked:
            return froms, LeveledPolicy.get_overlapping_components(froms, tos)
        perms = list(permutations(froms, num_picked))
        picked = list(perms[0])
        overlaps = LeveledPolicy.get_overlapping_components(picked, tos)
        num_overlaps = len(overlaps)
        for i in range(1, len(perms)):
            potential = list(perms[i])
            potential_overlaps = LeveledPolicy.get_overlapping_components(potential, tos)
            potential_num_overlaps = len(potential_overlaps)
            if (is_min and potential_num_overlaps < num_overlaps) or \
                    (not is_min and potential_num_overlaps > num_overlaps):
                picked = potential
                overlaps = potential_overlaps
                num_overlaps = potential_num_overlaps
        return picked, overlaps

    def get_mergable_components(self, is_flush: bool) -> (List[AbstractDiskComponent], List[AbstractDiskComponent], int):
        levels = self._lsmtree.leveled_disk_components().copy()
        num_levels = len(levels)
        for lv in range(num_levels - 1, -1, -1):
            components = levels[lv]
            num = len(components)
            if lv == 0:
                # if num >= self._lv0 + self._overflow:
                picked = []
                if num > self._lv0:
                    picked.append(components[-1])  # The oldest in level 0
                    if num_levels == 1:
                        mergable_components = []
                    else:
                        mergable_components = self.get_overlapping_components(picked, levels[1])
                    return picked, mergable_components, self._lsmtree.memory_component().max_records()
            else:
                level_max = int(math.floor(self._lv1 * (self._ratio ** (lv - 1))))
                # if num >= level_max + self._overflow:
                if num > level_max:
                    self._overflow = num - level_max
                    if num_levels == lv + 1:
                        picked, mergable_components = LeveledPolicy.get_by_time(components, self._overflow, True)
                        return picked, mergable_components, self._lsmtree.memory_component().max_records()
                    next_level_components = levels[lv + 1]
                    if self._pick == LeveledPolicy.PICK_FIRST:
                        picked = components[0:self._overflow]
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_LAST:
                        picked = components[-self._overflow:]
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_NEWEST:
                        picked = LeveledPolicy.get_by_time(components, self._overflow, False)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_RANDOM:
                        picked = LeveledPolicy.get_random(components, self._overflow)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_MIN_OVERLAP:
                        picked, mergable_components = self._get_min_max_overlapping_components(components,
                                                                                       next_level_components,
                                                                                       self._overflow, True)
                    elif self._pick == LeveledPolicy.PICK_MAX_OVERLAP:
                        picked, mergable_components = self._get_min_max_overlapping_components(components,
                                                                                       next_level_components,
                                                                                       self._overflow, False)
                    elif self._pick == LeveledPolicy.PICK_MIN_RANGE:
                        picked = LeveledPolicy.get_by_range(components, self._overflow, True)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_MAX_RANGE:
                        picked = LeveledPolicy.get_by_range(components, self._overflow, False)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    else:
                        # Pick the oldest (with smallest max_id)
                        picked = LeveledPolicy.get_by_time(components, self._overflow, True)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    return picked, mergable_components, self._lsmtree.memory_component().max_records()
        return [], [], -1

    def sort_components(self) -> List[List[AbstractDiskComponent]]:
        new_levels = []
        levels = self._lsmtree.leveled_disk_components()
        if len(levels) > 0:
            components = levels[0]
            to_sort = []
            for d in components:
                to_sort.append((d.max_id(), d))
            to_sort.sort(key=itemgetter(0), reverse=True)
            new_levels.append([d for m, d in to_sort])
        for lv in range(1, len(levels)):
            components = levels[lv]
            to_sort = []
            for d in components:
                to_sort.append((d.min_key(), d))
            to_sort.sort(key=itemgetter(0), reverse=False)
            new_levels.append([d for m, d in to_sort])
        return new_levels


class NoMergePolicy(StackPolicy):
    """ No merge policy. """

    def __init__(self, tree):
        super().__init__(tree)

    @staticmethod
    def policy_name() -> str:
        return "no-merge"

    def properties(self) -> dict:
        return {}

    def _get_mergable_components(self, is_flush: bool, components: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        return []


class ConstantPolicy(StackPolicy):
    """ Constant merge policy. Merge into one component whenever there is k+1 or more components. """
    PROP_NUM_COMPONENTS: str = "num_components"

    def __init__(self, tree, num_components: int):
        super().__init__(tree)
        self.__k = num_components
        if self.__k < 1:
            self._valid = False
            raise ValueError("num_components ({0}) must be at least 1".format(self.__k))
        self._props[ConstantPolicy.PROP_NUM_COMPONENTS] = self.__k

    @staticmethod
    def policy_name() -> str:
        return "constant"

    @staticmethod
    def default_properties() -> dict:
        return {
            ConstantPolicy.PROP_NUM_COMPONENTS: 6,
        }

    def num_components(self) -> int:
        return self.__k

    def _get_mergable_components(self, is_flush: bool, components: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        if not is_flush:
            return []
        return components if len(components) > self.__k else []


class BigtablePolicy(StackPolicy):
    """ Bigtable merge policy (Google Bigtable default). """
    PROP_NUM_COMPONENTS: str = "num_components"

    def __init__(self, tree, num_components: int):
        super().__init__(tree)
        self.__k = num_components
        if self.__k < 1:
            self._valid = False
            raise ValueError("num_components ({0}) must be at least 1".format(self.__k))
        self._props[BigtablePolicy.PROP_NUM_COMPONENTS] = self.__k

    @staticmethod
    def policy_name() -> str:
        return "bigtable"

    @staticmethod
    def default_properties() -> dict:
        return {
            BigtablePolicy.PROP_NUM_COMPONENTS: 6,
        }

    def num_components(self) -> int:
        return self.__k

    def _get_mergable_components(self, is_flush: bool, components: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        if not is_flush:
            return []
        cnt = len(components)
        if cnt <= self.__k:
            return []  # No merge if no more than k components
        components = list(reversed(components))
        total_size = 0
        for d in components:
            total_size += d.component_size()
        end_idx = merge_idx = cnt - 2
        for i in range(0, end_idx):
            if components[i].component_size() * 2 <= total_size:
                merge_idx = i
                break
            total_size -= components[i].component_size()
        mergable_components = []
        for i in range(merge_idx, cnt):
            mergable_components.append(components[i])
        return list(reversed(mergable_components))


class ExploringPolicy(StackPolicy):
    """ Exploring merge policy (HBase default). """
    PROP_MIN_COMPONENTS: str = "min_components"
    PROP_MAX_COMPONENTS: str = "max_components"
    PROP_RATIO: str = "ratio"

    def __init__(self, tree, min_components: int, max_components: int, ratio: float):
        super().__init__(tree)
        self.__min = min_components
        self.__max = max_components
        self.__ratio = ratio
        if self.__min < 1:
            self._valid = False
            raise ValueError("min_components ({0}) must be at least 1".format(self.__min))
        if self.__max < self.__min:
            self._valid = False
            raise ValueError("max_components ({0}) must be no less than min_components ({1})"
                             .format(self.__max, self.__min))
        if self.__ratio <= 0.0:
            self._valid = False
            raise ValueError("ratio ({0}) must be positive".format(self.__ratio))
        self._props[ExploringPolicy.PROP_MIN_COMPONENTS] = self.__min
        self._props[ExploringPolicy.PROP_MAX_COMPONENTS] = self.__max
        self._props[ExploringPolicy.PROP_RATIO] = self.__ratio

    @staticmethod
    def policy_name() -> str:
        return "exploring"

    @staticmethod
    def default_properties() -> dict:
        return {
            ExploringPolicy.PROP_MIN_COMPONENTS: 3,
            ExploringPolicy.PROP_MAX_COMPONENTS: 10,
            ExploringPolicy.PROP_RATIO: 1.2,
        }

    def min_components(self) -> int:
        return self.__min

    def max_components(self) -> int:
        return self.__max

    def ratio(self) -> float:
        return self.__ratio

    @staticmethod
    def __is_better_selection(best_selection: list, best_size: int, selection: list, size: int, might_stuck: bool) \
            -> bool:
        lb = len(best_selection)
        ls = len(selection)
        if might_stuck and best_size > 0 and size > 0:
            threshold_quantity = float(lb) / best_size
            return threshold_quantity < float(ls) / size
        return ls > lb or (ls == lb and size < best_size)

    @staticmethod
    def __total_size(components: List[AbstractDiskComponent]) -> int:
        total = 0
        for d in components:
            total += d.component_size()
        return total

    @staticmethod
    def __file_in_ratio(components: List[AbstractDiskComponent], ratio: float) -> bool:
        if len(components) < 2:
            return True
        total_size = ExploringPolicy.__total_size(components)
        for d in components:
            s = d.component_size()
            total_other = total_size - s
            if s > total_other * ratio:
                return False
        return True

    def _get_mergable_components(self, is_flush: bool, components: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        cnt = len(components)
        if cnt <= self.__min - 1:
            return []
        components = list(reversed(components))
        might_stuck = cnt > self.__max
        best_selection = []
        smallest = []
        best_size = 0
        smallest_size = float("inf")
        best_start = best_end = smallest_start = smallest_end = -1
        for start in range(0, cnt):
            for current_end in range(start + self.__min - 1, cnt):
                potentials = components[start:current_end + 1]
                if len(potentials) < self.__min:
                    continue
                size = self.__total_size(potentials)
                if might_stuck and size < smallest_size:
                    smallest = potentials
                    smallest_size = size
                    smallest_start = start
                    smallest_end = current_end
                if not self.__file_in_ratio(potentials, self.__ratio):
                    continue
                if self.__is_better_selection(best_selection, best_size, potentials, size, might_stuck):
                    best_selection = potentials
                    best_size = size
                    best_start = start
                    best_end = current_end
        if len(best_selection) == 0 and might_stuck and smallest_start != -1 and smallest_end != -1:
            return list(reversed(smallest))
        elif best_start != -1 and best_end != -1:
            return list(reversed(best_selection))
        else:
            return []


class SizeTieredPolicy(StackPolicy):
    """ SizeTiered merge policy (Cassandra default). """
    PROP_MIN_COMPONENTS: str = "min_components"
    PROP_MAX_COMPONENTS: str = "max_components"
    PROP_LOW_BUCKET: str = "low_bucket"
    PROP_HIGH_BUCKET: str = "high_bucket"
    PROP_MIN_COMPONENT_SIZE: str = "min_component_size"

    def __init__(self, tree, min_components: int, max_components: int, low_bucket: float, high_bucket: float,
                 min_component_size: Union[int, str]):
        super().__init__(tree)
        self.__min = min_components
        self.__max = max_components
        self.__low = low_bucket
        self.__high = high_bucket
        if isinstance(min_component_size, int):
            self.__size = min_component_size
        elif isinstance(min_component_size, str):
            ss = min_component_size.replace(" ", "").lower()
            if ss.endswith("b"):
                self.__size = int(ss[-1])
            elif ss.endswith("k"):
                self.__size = int(round(float(ss[-1]) * 1024))
            elif ss.endswith("kb"):
                self.__size = int(round(float(ss[-2]) * 1024))
            elif ss.endswith("m"):
                self.__size = int(round(float(ss[-1]) * 1024 * 1024))
            elif ss.endswith("mb"):
                self.__size = int(round(float(ss[-2]) * 1024 * 1024))
            elif ss.endswith("g"):
                self.__size = int(round(float(ss[-1]) * 1024 * 1024 * 1024))
            elif ss.endswith("gb"):
                self.__size = int(round(float(ss[-2]) * 1024 * 1024 * 1024))
            else:
                self.__size = -1
                self.__valid = False
                raise ValueError("min_component_size ({0}) is invalid".format(min_component_size))
        else:
            self.__size = -1
            self._valid = False
            raise TypeError("min_component_size ({0} {1}) is invalid"
                            .format(min_component_size, type(min_component_size)))
        if self.__min < 2:
            self._valid = False
            raise ValueError("min_components ({0}) must be at least 2".format(self.__min))
        if self.__max < self.__min:
            self._valid = False
            raise ValueError("max_components ({0}) must be no less than min_components ({1})"
                             .format(self.__max, self.__min))
        if self.__low <= 0.0:
            self._valid = False
            raise ValueError("low_bucket ({0}) must be positive".format(self.__low))
        if self.__high < self.__low:
            self._valid = False
            raise ValueError("high_bucket ({0}) must be larger than low_bucket ({1})"
                             .format(self.__high, self.__low))
        if self.__size < 1:
            self._valid = False
            raise ValueError("min_component_size ({0}) must be positive".format(self.__size))
        self._props[SizeTieredPolicy.PROP_MIN_COMPONENTS] = self.__min
        self._props[SizeTieredPolicy.PROP_MAX_COMPONENTS] = self.__max
        self._props[SizeTieredPolicy.PROP_LOW_BUCKET] = self.__low
        self._props[SizeTieredPolicy.PROP_HIGH_BUCKET] = self.__high
        self._props[SizeTieredPolicy.PROP_MIN_COMPONENT_SIZE] = self.__size

    @staticmethod
    def policy_name() -> str:
        return "size-tiered"

    @staticmethod
    def default_properties() -> dict:
        return {
            SizeTieredPolicy.PROP_MIN_COMPONENTS: 4,
            SizeTieredPolicy.PROP_MAX_COMPONENTS: 32,
            SizeTieredPolicy.PROP_LOW_BUCKET: 0.5,
            SizeTieredPolicy.PROP_HIGH_BUCKET: 1.5,
            SizeTieredPolicy.PROP_MIN_COMPONENT_SIZE: 50 * (1024 ** 2),  # 50 MB
        }

    def min_components(self) -> int:
        return self.__min

    def max_components(self) -> int:
        return self.__max

    def low_bucket(self) -> float:
        return self.__low

    def high_bucket(self) -> float:
        return self.__high

    def min_component_size(self) -> int:
        return self.__size

    def _get_mergable_components(self, is_flush: bool, components: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        if not is_flush:
            return []
        cnt = len(components)
        for start in range(0, cnt - self.__min + 1):
            max_end = min(cnt, start + self.__max)
            for end in range(max_end - 1, start + self.__min - 1, -1):
                all_small = True
                total = 0.0
                mergable_components = []
                for i in range(start, end + 1):
                    d = components[i]
                    mergable_components.append(d)
                    total += d.component_size()
                    if d.component_size() >= self.__size:
                        all_small = False
                if all_small:
                    return mergable_components
                avg_size = total / (end - start + 1)
                is_bucket = True
                for d in mergable_components:
                    if d.component_size() < avg_size * self.__low or d.component_size() > avg_size * self.__high:
                        is_bucket = False
                        break
                if is_bucket:
                    return mergable_components
        return []


class TieredPolicy(StackPolicy):
    """ A tiered merge policy that mainly depends on one parameter, the size ratio. """
    PROP_RATIO: str = "ratio"
    PROP_SCALE: str = "scale"

    def __init__(self, tree, ratio: int, scale: float):
        super().__init__(tree)
        self.__ratio = ratio
        self.__scale = scale
        if self.__ratio < 2:
            self._valid = False
            raise ValueError("Ratio ({0}) must be at least 2".format(self.__ratio))
        if self.__scale < 1.0:
            self._valid = False
            raise ValueError("Scale ({0}) must be at least 1".format(self.__scale))
        self._props[TieredPolicy.PROP_RATIO] = self.__ratio
        self._props[TieredPolicy.PROP_SCALE] = self.__scale

    @staticmethod
    def policy_name() -> str:
        return "tiered"

    @staticmethod
    def default_properties() -> dict:
        return {
            TieredPolicy.PROP_RATIO: 10,
            TieredPolicy.PROP_SCALE: 1.2,
        }

    def ratio(self) -> int:
        return self.__ratio

    def scale(self) -> float:
        return self.__scale

    def _get_mergable_components(self, is_flush: bool, components: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        max_records = int(math.ceil(self._lsmtree.memory_component().max_records() * self.__scale))

        def get_tier(num_records: int) -> int:
            n = math.ceil(num_records / max_records)
            return int(math.ceil(math.log(n, self.__ratio))) + 1

        tiers = []
        for d in components:
            t = get_tier(sum(d.num_records()))
            tier_size = len(tiers)
            if t > tier_size:
                if t - tier_size > 1:
                    for i in range(tier_size, t - 1):
                        tiers.append([])
                tiers.append([d, ])
            else:
                tiers[-1].append(d)
        for t in range(len(tiers) - 1, -1, -1):
            tier_ds = tiers[t]
            next_tier_records = round(max_records * math.pow(self.__ratio, t + 1))
            if len(tier_ds) >= self.__ratio:
                for l in range(len(tier_ds), self.__ratio - 1, -1):
                    total = sum([sum(d.num_records()) for d in tier_ds[0:l]])
                    if total <= next_tier_records:
                        return tier_ds[0:l]
        return []


class LeveledNPolicy(LeveledPolicy):
    """ Class for the Leveled-N merge policy. """
    PROP_LEVEL_N: str = "level-n"

    def __init__(self, tree, level_n: int, components_0: int, components_1: int, ratio: float, pick: str,
                 overflow: int = 1):
        super().__init__(tree, components_0, components_1, ratio, pick, overflow)
        self.__n = level_n
        if level_n < 2:
            self._valid = False
            raise ValueError("level_n ({0}) must be at least 2".format(self.__n))
        self._props[LeveledNPolicy.LEVEL_N] = self.__n

    @staticmethod
    def policy_name() -> str:
        return "leveled-n"

    @staticmethod
    def default_properties() -> dict:
        props = LeveledPolicy.default_properties()
        props[LeveledNPolicy.PROP_LEVEL_N] = 4
        return props

    def level_n(self) -> int:
        return self.__n

    def get_mergable_components(self, is_flush: bool) -> (List[AbstractDiskComponent], int):
        levels = self._lsmtree.leveled_disk_components().copy()
        num_levels = len(levels)
        for lv in range(num_levels - 1, -1, -1):
            components = levels[lv]
            num = len(components)
            if lv == 0:
                if num >= self._lv0 + self._overflow:
                    picked = components[-self._overflow:]  # The oldest in level 0
                    if num_levels == 1:
                        mergable_components = picked
                    else:
                        mergable_components = picked + self.get_overlapping_components(picked, levels[1])
                    return mergable_components, self._lsmtree.memory_component().max_records()
            else:
                # Level 1 to N are the same, N+1 to 2N are the same, 2N+1 to 3N are the same...
                level_max = int(math.floor(self._lv1 * (self._ratio ** (math.ceil(lv / self.__n) - 1))))
                if num >= level_max + self._overflow:
                    if num_levels == lv + 1:
                        mergable_components = LeveledPolicy.get_by_time(components, self._overflow, True)
                        return mergable_components, self._lsmtree.memory_component().max_records()
                    next_level_components = levels[lv + 1]
                    if self._pick == LeveledPolicy.PICK_FIRST:
                        picked = components[0:self._overflow]
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_LAST:
                        picked = components[-self._overflow:]
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_NEWEST:
                        picked = LeveledPolicy.get_by_time(components, self._overflow, False)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_RANDOM:
                        picked = LeveledPolicy.get_random(components, self._overflow)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_MIN_OVERLAP:
                        mergable_components = self._get_min_max_overlapping_components(components,
                                                                                       next_level_components,
                                                                                       self._overflow, True)
                    elif self._pick == LeveledPolicy.PICK_MAX_OVERLAP:
                        mergable_components = self._get_min_max_overlapping_components(components,
                                                                                       next_level_components,
                                                                                       self._overflow, False)
                    elif self._pick == LeveledPolicy.PICK_MIN_RANGE:
                        picked = LeveledPolicy.get_by_range(components, self._overflow, True)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_MAX_RANGE:
                        picked = LeveledPolicy.get_by_range(components, self._overflow, False)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    else:
                        # Pick the oldest (with smallest max_id)
                        picked = LeveledPolicy.get_by_time(components, self._overflow, True)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    return mergable_components, self._lsmtree.memory_component().max_records()
        return [], -1


class TieredLeveledPolicy(LeveledPolicy):
    """ Class for the Tiered+Leveled merge policy. """
    PROP_MAX_TIERS: str = "max-tiers"

    # components_0: number of components in each tier, also the size ratio of tiered.
    # components_1: number of components in the first level.
    # ratio: size ratio of leveled.
    def __init__(self, tree, max_tiers: int, components_0: int, components_1: int, ratio: float, pick: str,
                 overflow: int = 1):
        super().__init__(tree, components_0, components_1, ratio, pick, overflow)
        self.__max_tiers = max_tiers
        if self.__max_tiers < 1:
            self._valid = False
            raise ValueError("max_tiers ({0}) must be at least 1".format(self.__max_tiers))
        self._props[TieredLeveledPolicy.PROP_MAX_TIERS] = self.__max_tiers

    @staticmethod
    def policy_name() -> str:
        return "tiered-leveled"

    @staticmethod
    def default_properties() -> dict:
        props = LeveledPolicy.default_properties()
        props[TieredLeveledPolicy.PROP_MAX_TIERS] = 4
        return props

    def max_tiers(self) -> int:
        return self.__max_tiers

    def get_mergable_components(self, is_flush: bool) -> (List[AbstractDiskComponent], int):
        levels = self._lsmtree.leveled_disk_components().copy()
        num_levels = len(levels)
        for lv in range(num_levels - 1, -1, -1):
            components = levels[lv]
            num = len(components)
            if lv < self.__max_tiers - 1:
                # Pure tiered (max_tieres = 3, lv = 0 or 1, component size = lv0 ** (lv + 1))
                max_records = self._lsmtree.memory_component().max_records() * (self._lv0 ** (lv + 1))
                total_records = sum([d.num_records() for d in components])
                if total_records >= max_records:
                    return components, max_records
            elif lv == self.__max_tiers - 1:
                # Pure tiered, but the next level is pure leveld.
                # Merge all from this level with overlapping components in the next leve.
                # max_tiers = 3, lv = 2, component size = lv0 ** max_tiers
                max_records = self._lsmtree.memory_component().max_records() * (self._lv0 ** self.__max_tiers)
                total_records = sum([sum(d.num_records()) for d in components])
                if total_records >= max_records:
                    # Select all components in this level, plus the overlapping components in the next level
                    if num_levels == lv + 1:
                        return components, max_records
                    else:
                        next_level = levels[lv + 1]
                        overlaps = LeveledPolicy.get_overlapping_components(components, next_level)
                        return components + overlaps, max_records
            else:
                # Pure leveled
                max_records = self._lsmtree.memory_component().max_records() * (self._lv0 ** self.__max_tiers)
                level_max = int(math.floor(self._lv1 * (self._ratio ** (lv - self.__max_tiers))))
                if num >= level_max + self._overflow:
                    if num_levels == lv + 1:
                        return LeveledPolicy.get_by_time(components, self._overflow, True), max_records
                    next_level_components = levels[lv + 1]
                    if self._pick == LeveledPolicy.PICK_FIRST:
                        picked = components[0:self._overflow]
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_LAST:
                        picked = components[-self._overflow:]
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_NEWEST:
                        picked = LeveledPolicy.get_by_time(components, self._overflow, False)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_RANDOM:
                        picked = LeveledPolicy.get_random(components, self._overflow)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_MIN_OVERLAP:
                        mergable_components = self._get_min_max_overlapping_components(components,
                                                                                       next_level_components,
                                                                                       self._overflow, True)
                    elif self._pick == LeveledPolicy.PICK_MAX_OVERLAP:
                        mergable_components = self._get_min_max_overlapping_components(components,
                                                                                       next_level_components,
                                                                                       self._overflow, False)
                    elif self._pick == LeveledPolicy.PICK_MIN_RANGE:
                        picked = LeveledPolicy.get_by_range(components, self._overflow, True)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    elif self._pick == LeveledPolicy.PICK_MAX_RANGE:
                        picked = LeveledPolicy.get_by_range(components, self._overflow, False)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    else:
                        # Pick the oldest (with smallest max_id)
                        picked = LeveledPolicy.get_by_time(components, self._overflow, True)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    return mergable_components, max_records
        return [], -1

    def sort_components(self) -> List[List[AbstractDiskComponent]]:
        new_levels = []
        levels = self._lsmtree.leveled_disk_components()
        num_levels = len(levels)
        # Sort as stack-based
        for lv in range(min(num_levels, self.__max_tiers)):
            components = levels[lv]
            to_sort = []
            for d in components:
                to_sort.append((d.max_id(), d))
            to_sort.sort(key=itemgetter(0), reverse=True)
            new_levels.append([d for m, d in to_sort])
        if num_levels > self.__max_tiers:
            # Sort as Leveled
            for lv in range(self.__max_tiers, num_levels):
                components = levels[lv]
                to_sort = []
                for d in components:
                    to_sort.append((d.min_key(), d))
                to_sort.sort(key=itemgetter(0), reverse=False)
                new_levels.append([d for m, d in to_sort])
        return new_levels


class BinomialPolicy(StackPolicy):
    PROP_NUM_COMPONENTS: str = "num_components"

    def __init__(self, tree, num_components: int):
        super().__init__(tree)
        self.__k = num_components
        if self.__k < 1:
            self._valid = False
            raise ValueError("num_components ({0}) must be at least 1".format(self.__k))
        self._props[BinomialPolicy.PROP_NUM_COMPONENTS] = self.__k
        self.__bin: Optional[List[int]] = None

    @staticmethod
    def policy_name() -> str:
        return "binomial"

    @staticmethod
    def default_properties() -> dict:
        return {
            BinomialPolicy.PROP_NUM_COMPONENTS: 4,
        }

    def num_components(self) -> int:
        return self.__k

    def _get_mergable_components(self, is_flush: bool, components: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        if not is_flush:
            return []
        size = len(components)

        def tree_depth(d: int) -> int:
            if d < 0:
                return 0
            else:
                return tree_depth(d - 1) + bin_choose(d + min(d, self.__k) - 1, d)

        def bin_index(d: int, h: int, t: int) -> int:
            # if t < 0 or t > bin_choose(d + h, h):
            #     raise ValueError("Illegal binomial values: d = {0}, h = {1}, t = {2}, choose = {3}"
            #                      .format(d, h, t, bin_choose(d + h, h)))
            if t == 0:
                return 0
            elif t < bin_choose(d + h - 1, h):
                return bin_index(d - 1, h, t)
            else:
                return bin_index(d, h - 1, t - bin_choose(d + h - 1, h)) + 1

        def bin_choose(n: int, k: int) -> int:
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1

            def cell(row: int, col: int) -> int:
                return row * (n + 1) + col

            if self.__bin is None or len(self.__bin) != ((n + 1) ** 2):
                self.__bin = [0, ] * ((n + 1) ** 2)
                for r in range(0, n + 1):
                    for c in range(0, min(r, k) + 1):
                        if c == 0 or c == r:
                            self.__bin[cell(r, c)] = 1
                        else:
                            self.__bin[cell(r, c)] = self.__bin[cell(r - 1, c - 1)] + self.__bin[cell(r - 1, c)]
            return self.__bin[cell(n, k)]

        num_flushes = self._lsmtree.memory_component().max_id() - 1
        depth = 0
        while tree_depth(depth) < num_flushes:
            depth += 1
        merge_idx = bin_index(depth, min(depth, self.__k) - 1, num_flushes - tree_depth(depth - 1) - 1)
        if merge_idx == size - 1:
            return []
        return components[:size - merge_idx]


class MinLatencyPolicy(StackPolicy):
    PROP_NUM_COMPONENTS: str = "num_components"

    def __init__(self, tree, num_components: int):
        super().__init__(tree)
        self.__k = num_components
        if self.__k < 1:
            self._valid = False
            raise ValueError("num_components ({0}) must be at least 1".format(self.__k))
        self._props[MinLatencyPolicy.PROP_NUM_COMPONENTS] = self.__k
        self.__bin: Optional[List[int]] = None

    @staticmethod
    def policy_name() -> str:
        return "min-latency"

    @staticmethod
    def default_properties() -> dict:
        return {
            MinLatencyPolicy.PROP_NUM_COMPONENTS: 4,
        }

    def num_components(self) -> int:
        return self.__k

    def _get_mergable_components(self, is_flush: bool, components: List[AbstractDiskComponent]) \
            -> List[AbstractDiskComponent]:
        if not is_flush:
            return []
        size = len(components)
        if size <= self.__k:
            return []

        def tree_depth(d: int) -> int:
            if d < 0:
                return 0
            else:
                return bin_choose(d + self.__k, d) - 1

        def bin_index(d: int, h: int, t: int) -> int:
            # if t < 0 or t > bin_choose(d + h, h):
            #     raise ValueError("Illegal binomial values: d = {0}, h = {1}, t = {2}, choose = {3}"
            #                      .format(d, h, t, bin_choose(d + h, h)))
            if t == 0:
                return 0
            elif t < bin_choose(d + h - 1, h):
                return bin_index(d - 1, h, t)
            else:
                return bin_index(d, h - 1, t - bin_choose(d + h - 1, h)) + 1

        def bin_choose(n: int, k: int) -> int:
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            w = n + 1

            def cell(row, col):
                return row * w + col

            if self.__bin is None or len(self.__bin) != (w ** 2):
                self.__bin = [0, ] * (w ** 2)
                for r in range(0, w):
                    for c in range(0, min(r, k) + 1):
                        if c == 0 or c == r:
                            self.__bin[cell(r, c)] = 1
                        else:
                            self.__bin[cell(r, c)] = self.__bin[cell(r - 1, c - 1)] + self.__bin[cell(r - 1, c)]
            return self.__bin[cell(n, k)]

        num_flushes = self._lsmtree.memory_component().max_id() - 1
        depth = 0
        while tree_depth(depth) < num_flushes:
            depth += 1
        merge_idx = bin_index(depth, self.__k - 1, num_flushes - tree_depth(depth - 1) - 1)
        if merge_idx == size - 1:
            return []
        return components[:size - merge_idx]


class LazyLeveledPolicy(LeveledPolicy):
    """ Class for the Leveled-N merge policy. """

    def _validate_tree(self, tree) -> bool:
        from .tree import LazyLeveledLSMTree
        if tree is None or not isinstance(tree, LazyLeveledLSMTree):
            print("tree <{0}> is not an instance of LazyLeveledLSMTree".format("None" if tree is None else type(tree)),
                  file=sys.stderr)
            return False
        else:
            self._lsmtree: LazyLeveledLSMTree = tree
            return True

    def __init__(self, tree, components_0: int, components_1: int, ratio: float, pick: str, overflow: int,
                 nc_threshold: int, ssc_threshold: int, have_componentary: bool, do_rebuilding: bool):
        super().__init__(tree, components_0, components_1, ratio, pick, overflow)
        self.nc_threshold = nc_threshold
        self.ssc_threshold = ssc_threshold
        self.have_componentary = have_componentary
        self.do_rebuilding = do_rebuilding
        self._props[LeveledPolicy.PROP_NC_T] = self.nc_threshold
        self._props[LeveledPolicy.PROP_SSC_T] = self.ssc_threshold
        self._props[LeveledPolicy.PROP_COMPONENTARY] = self.have_componentary
        self._props[LeveledPolicy.PROP_REBUILDING] = self.do_rebuilding

    @staticmethod
    def policy_name() -> str:
        return "lazy-leveled"

    @staticmethod
    def default_properties() -> dict:
        props = LeveledPolicy.default_properties()
        props[TieredLeveledPolicy.PROP_MAX_TIERS] = 4
        return props

    def get_mergable_components_old(self, is_flush: bool, is_query: bool, do_remove_fragment: bool) -> (List[AbstractDiskComponent], List[AbstractDiskComponent], int):
        levels = self._lsmtree.leveled_disk_components().copy()
        num_levels = len(levels)
        for lv in range(num_levels - 1, -1, -1):
            components = levels[lv]
            num = len(components)
            if lv == 0:
                picked, mergable_components = self.without_unsafe_SP(is_query, num_levels, levels)
                return picked, mergable_components, self._lsmtree.memory_component().max_records()
            else:
                level_max = int(math.floor(self._lv1 * (self._ratio ** (lv - 1))))
                # if num >= level_max + self._overflow:
                if num > level_max:
                    self._overflow = num - level_max
                    if num_levels == lv + 1:
                        picked, mergable_components = LeveledPolicy.get_by_time(components, self._overflow, True)
                        return picked, mergable_components, self._lsmtree.memory_component().max_records()
                    next_level_components = levels[lv + 1]
                    if self._pick == LeveledPolicy.PICK_MIN_OVERLAP:
                        picked, mergable_components = self._get_min_max_overlapping_components(components,
                                                                                               next_level_components,
                                                                                               self._overflow, True,
                                                                                               do_remove_fragment)
                        return picked, mergable_components, self._lsmtree.memory_component().max_records()
                    else:
                        # Pick the oldest (with smallest max_id)
                        picked = LeveledPolicy.get_by_time(components, self._overflow, True)
                        mergable_components = picked + \
                                              LeveledPolicy.get_overlapping_components(picked, next_level_components)
                    return picked, mergable_components, self._lsmtree.memory_component().max_records()
        return [], [], -1

    def get_mergable_components(self, is_flush: bool, is_query: bool, do_remove_fragment: bool, lv: int) -> (List[SpecialDiskComponent], List[SpecialDiskComponent], int):
        levels = self._lsmtree.leveled_disk_components()
        num_levels = len(levels)
        if lv == 0:
            picked, mergable_components, keep_sp = self.without_unsafe_SP(is_query, num_levels, levels)
            if num_levels == 1:
                lv = -1
            else:
                lv += 1
            return picked, mergable_components, self._lsmtree.memory_component().max_records(), lv, keep_sp
        else:
            components = levels[lv]
            num = len(components)
            level_max = int(math.floor(self._lv1 * (self._ratio ** (lv - 1))))
            self._overflow = num - level_max
            if num > level_max:
                if num_levels == lv + 1:
                    picked, mergable_components = self.get_by_time(components, self._overflow, True)
                    return picked, mergable_components, self._lsmtree.memory_component().max_records(), -1, False
                next_level_components = levels[lv + 1]
                if next_level_components[0].min_id() != (lv + 1):
                    next_level_components = []
                if self._pick == LeveledPolicy.PICK_MIN_OVERLAP:
                    picked, mergable_components = self._get_min_max_overlapping_components(components, next_level_components, self._overflow, True, do_remove_fragment)
                    lv += 1
                    return picked, mergable_components, self._lsmtree.memory_component().max_records(), lv, False
            else:
                lv = -1
                return [], [], -1, lv, False

    def get_by_time(self, components: List[AbstractDiskComponent], num_picked: int, oldest: bool = True) \
            -> List[AbstractDiskComponent]:
        """ Return a component in a level with smallest (oldest) or largest (newest) max_id. """
        cnt = len(components)
        if cnt < num_picked:
            return [], []
        if cnt == num_picked:
            return components, []
        ts = []
        lv = -1
        for d in components:
            if lv == -1:
                lv = d.min_id()
            elif d.min_id() != lv:
                for c in components:
                    print(c.name())
                raise ValueError("Components are not in the same level: {0} vs {1}"
                                 .format(components[0].name(), d.name()))
            ts.append((d.max_id(), d))
        ts.sort(key=itemgetter(0), reverse=not oldest)
        return [d for max_id, d in ts[0:num_picked]], []

    def without_unsafe_SP(self, is_query: bool, num_levels: int, levels):
        picked = []
        if is_query:
            # check safe special components
            safe_special_components = self._lsmtree.safe_special_components
            num_pick = len(safe_special_components) - self.ssc_threshold
            if num_pick > 0:
                for sp_name in list(safe_special_components.keys()):
                    sp = safe_special_components[sp_name]
                    picked.append(sp)
                    safe_special_components.pop(sp_name)
                    if len(picked) == num_pick:
                        break

                self._lsmtree.safe_special_components = safe_special_components
            else:
                return [], [], False
        else:
            # 1. check normal components in L0
            if len(self._lsmtree.normal_components) > self.nc_threshold:
                # key: normal component, value: list of usc tuple
                # get the oldest
                normal_component: SpecialDiskComponent = self._lsmtree.normal_components[0]
                picked.append(normal_component)
                self._lsmtree.normal_components.remove(normal_component)

                # sentinel_range_idxs = self.lsm_tree().partition.get_sentinel_ranges()
                # sentinel_components_ranges = self.lsm_tree().get_ranges_by_partition_idx(sentinel_range_idxs)
                sentinel_components_map: OrderedDict = self._lsmtree.safe_special_components

                if self.have_componentary:
                    if len(sentinel_components_map) > 0:
                        keep_sp = True
                    else:
                        keep_sp = False
                else:
                    # flush the overlapping sentinel components to lower level
                    sentinel_components = list(sentinel_components_map.values())
                    overlapping_sentinel_components = self.get_overlapping_components([normal_component, ],
                                                                                      sentinel_components)
                    picked += overlapping_sentinel_components

                    # remove overlapping_sentinel_components from current sentinel maps
                    for sc in overlapping_sentinel_components:
                        sc: SpecialDiskComponent
                        sentinel_components_map.pop(sc.name())
                        # let current sentinel range read counter to be empty
                        reset_sc_range = (sc.min_key(), sc.max_key())
                        self._lsmtree.partition.reset_read_counter_by_range(reset_sc_range)

                    self._lsmtree.safe_special_components = sentinel_components_map

                    keep_sp = False
            else:
                return [], [], False

        if len(picked) > 0:
            self._overflow = len(picked)
        else:
            self._overflow = 1

        if num_levels == 1:
            mergable_components = []
        else:
            mergable_components = self.get_overlapping_components(picked, levels[1])
        return picked, mergable_components, keep_sp

    def _get_min_max_overlapping_components(self, froms: List[SpecialDiskComponent], tos: List[SpecialDiskComponent],
                                            num_picked: int, is_min: bool, do_remove_fragment: bool) -> (List[SpecialDiskComponent], List[SpecialDiskComponent]):
        from_cnt = len(froms)
        if from_cnt < num_picked:
            raise ValueError("No thing to pick")
        if len(tos) == 0:
            # No overlapping, return the oldest only
            return self.get_by_time(froms, num_picked, True)
        if froms[0].min_id() + 1 != tos[0].min_id():
            raise ValueError("Invalid merge from level {0} to {1}".format(froms[0].min_id(), tos[0].min_id()))
            # return self.get_by_time(froms, num_picked, True)
        if from_cnt == num_picked:
            return froms, LeveledPolicy.get_overlapping_components(froms, tos)
        fragment = []
        fragment_overlaps = []
        # if self._lsmtree.remove_frag and do_remove_fragment:
        #     self._lsmtree.num_rf += 1
        #     # remove fragmet
        #     # print("remove fragment")
        #     fragment = []
        #     for c1 in froms:
        #         if len(c1.key_ranges()) > self.size_t:
        #             fragment.append(c1)
        #             froms.remove(c1)
        #     fragment_overlaps = LeveledPolicy.get_overlapping_components(fragment, tos)
        #     for c2 in fragment_overlaps:
        #         tos.remove(c2)

        # new version
        picked = fragment
        overlaps = fragment_overlaps
        to_sort = []
        dic = {}
        for i in range(len(froms)):
            current: SpecialDiskComponent = froms[i]
            # try to merge regular component first
            if not current.is_special:
                potential = [current, ]
                potential_overlaps = LeveledPolicy.get_overlapping_components(potential, tos)
                to_sort.append((froms[i], len(potential_overlaps)))
                dic[froms[i]] = potential_overlaps
        # if all the component is sentinel components, pick default
        if len(to_sort) == 0:
            for i in range(len(froms)):
                current: SpecialDiskComponent = froms[i]
                potential = [current, ]
                potential_overlaps = LeveledPolicy.get_overlapping_components(potential, tos)
                to_sort.append((froms[i], len(potential_overlaps)))
                dic[froms[i]] = potential_overlaps
        to_sort.sort(key=itemgetter(1), reverse=False)
        cnt = 0
        while len(picked) < num_picked:
            tuple_pick_overlaps = to_sort[cnt]
            picked.append(tuple_pick_overlaps[0])
            overlaps += dic[tuple_pick_overlaps[0]]
            cnt += 1
        return picked, overlaps

    def pick_fragment(self, froms: List[SpecialDiskComponent], tos: List[SpecialDiskComponent], do_remove_fragment: bool) -> (List[SpecialDiskComponent], List[SpecialDiskComponent]):
        fragment = []
        fragment_overlaps = []
        if self._lsmtree.remove_frag and do_remove_fragment:
            self._lsmtree.num_rf += 1
            fragment = []
            for c1 in froms:
                if len(c1.key_ranges()) > self.size_t:
                    fragment.append(c1)
            fragment_overlaps = LeveledPolicy.get_overlapping_components(fragment, tos)
        return fragment, fragment_overlaps

    def sort_components(self) -> List[List[SpecialDiskComponent]]:
        new_levels = []
        levels = self._lsmtree.leveled_disk_components()
        if len(levels) > 0:
            components = levels[0]
            to_sort = []
            for d in components:
                to_sort.append((d.max_id(), d))
            to_sort.sort(key=itemgetter(0), reverse=True)
            new_levels.append([d for m, d in to_sort])
        for lv in range(1, len(levels)):
            components = levels[lv]
            to_sort = []
            for d in components:
                to_sort.append((d.min_key(), d))
            to_sort.sort(key=itemgetter(0), reverse=False)
            new_levels.append([d for m, d in to_sort])
        return new_levels

StackPolicies = (
    NoMergePolicy.policy_name(),
    ConstantPolicy.policy_name(),
    ExploringPolicy.policy_name(),
    BigtablePolicy.policy_name(),
    SizeTieredPolicy.policy_name(),
    TieredPolicy.policy_name(),
    BinomialPolicy.policy_name(),
    MinLatencyPolicy.policy_name(),
)

LeveledPolicies = (
    LeveledPolicy.policy_name(),
    LeveledNPolicy.policy_name(),
    TieredLeveledPolicy.policy_name(),
    LazyLeveledPolicy.policy_name()
)
