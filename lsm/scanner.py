import math
import os
import sqlite3
from queue import PriorityQueue
from typing import List, Optional, Tuple, Union

from .component import Component, AbstractDiskComponent, AbstractMemoryComponent, DiskComponent, MemoryComponent, \
    DiskDBComponent, MemoryDBComponent, SpecialDiskComponent


class Scanner:
    """ Base class for scanning a component """

    def __init__(self, component: Component):
        """
        Scan operation

        Args:
            component: A component to be scanned.
        """
        self._component: Component = component
        self._c_min = component.min_key()
        self._c_max = component.max_key()
        self._min_key = self._c_min
        self._include_min = True
        self._max_key = self._c_max
        self._include_max = True
        self.__scanner_added = False

    def __del__(self):
        if self.is_open():
            self._close()
        if self.__scanner_added:
            self._component.remove_scanner()

    def component(self) -> Component:
        return self._component

    def open(self) -> bool:
        """ Open the component for reading. """
        self._component.add_scanner()
        self.__scanner_added = True
        return self._open()

    def close(self) -> bool:
        """ Close the component. """
        r = self._close()
        self._component.remove_scanner()
        self.__scanner_added = False
        return r

    def _open(self) -> bool:
        # Override in sub-classes
        return False

    def _close(self) -> bool:
        # Override in sub-classes
        return False

    def set_min_key(self, min_key: bytes, inclusive: bool) -> bool:
        """
        Set the min key for search. Or no min key if this function is not called.
        The value will be checked against the component's max key.
        """
        if min_key is None:
            return False
        if min_key > self._c_max or (min_key == self._c_max and not inclusive):
            raise KeyError("Search min key ({0}) is greater than the component's max key ({1})"
                           .format(min_key, self._c_max))
        # Only set a valid min key within the component's range
        if min_key < self._c_min:
            self._min_key = self._c_min
            self._include_min = True
        else:
            self._min_key = min_key
            self._include_min = inclusive
        return True

    def get_min_key(self) -> (Optional[bytes], bool):
        return self._min_key, self._include_min

    def set_max_key(self, max_key: bytes, inclusive: bool):
        """
        Set the max key for search. Or no max key if this function is not called.
        The value will be checked against the component's min key.
        """
        if max_key is None:
            return False
        if max_key < self._c_min or (max_key == self._c_min and not inclusive):
            raise KeyError("Search max key ({0}) is smaller than the component's min key ({1})"
                           .format(max_key, self._c_min))
        # Only set a valid max key within the component's range
        if max_key > self._c_max:
            self._max_key = self._c_max
            self._include_max = True
        else:
            self._max_key = max_key
            self._include_max = inclusive
        return True

    def get_max_key(self) -> (Optional[bytes], bool):
        return self._max_key, self._include_max

    def next(self) -> (Optional[bytes], Optional[bytes]):
        return None, None

    def is_open(self) -> bool:
        return False


class AbstractDiskScanner(Scanner):
    def __init__(self, component: AbstractDiskComponent, ridx: int):
        super().__init__(component)
        self._component: AbstractDiskComponent = self._component
        if ridx < 0 or ridx >= len(self._component.key_ranges()):
            raise IndexError("Invalid virtual range index {0}, max={1}".format(ridx, len(self._component.key_ranges())))
        self._ridx = ridx
        self._c_min, self._c_max = self._component.key_ranges()[ridx]
        self._min_key = self._c_min
        self._include_min = True
        self._max_key = self._c_max
        self._include_max = True

    def range_index(self) -> int:
        return self._ridx


class AbstractMemoryScanner(Scanner):
    def __init__(self, component: AbstractMemoryComponent):
        super().__init__(component)
        self._component: AbstractMemoryComponent = self._component


class DiskScanner(AbstractDiskScanner):
    def __init__(self, component: DiskComponent, ridx: int, IO_unit):
        super().__init__(component, ridx)
        self._component: DiskComponent = self._component
        self.start_pos = -1
        self.__pos = -1
        self.__file = None
        self.__stop = False
        self.num_read_records = 0
        self.is_memory = False
        self.end_pos = -1
        self.page_end = []
        self.IO_unit = IO_unit

    def __del__(self):
        super().__del__()

    def _open(self) -> bool:
        """ Open the component file and seek to the smallest record that matches the search condition. """
        path = self._component.get_binary_path()
        if os.path.isfile(path):
            try:
                self.__file = open(path, "rb")
                if self._min_key is None or self._min_key == self._component.min_key():
                    # Start from the beginning
                    self.__pos = 0
                # elif self._min_key == self._component.min_key():
                #     self.__pos = 0
                    self.start_pos = 0
                    self.page_end = self.IO_unit * self._component.record_length()
                    return True
                else:
                    # Find the first record that matches the condition
                    st = 0  # Binary search start
                    ed = self._component.actual_num_records() - 1  # Binary search end
                    if st == ed:
                        # Component has only 1 record, this record must satisfy the search condition.
                        # Because it's checked in set_min_key()
                        self.__pos = 0
                        self.__stop = False
                        self.start_pos = self.__pos
                        self.page_end = self.start_pos + self.IO_unit * self._component.record_length()
                        return True
                    while ed >= st:
                        mid = int(math.floor((ed + st) / 2))
                        p = self._component.key_pos(mid)
                        self.__file.seek(p)  # Seek to the beginning position of the mid record
                        rkey = self.__file.read(self._component.key_length())  # Read certain bytes as key
                        if rkey == self._min_key:
                            if self._include_min:
                                self.__pos = p
                            else:
                                self.__pos = p + self._component.record_length()
                            self.__stop = False
                            self.start_pos = self.__pos
                            self.page_end = self.start_pos + self.IO_unit * self._component.record_length()
                            return True
                        else:
                            if st == ed:
                                # rkey is the first that is larger than min_key
                                self.__pos = p
                                self.__stop = False
                                self.start_pos = self.__pos
                                self.page_end = self.start_pos + self.IO_unit * self._component.record_length()
                                return True
                            if rkey < self._min_key:
                                # Search in the second half
                                st = mid if ed > st + 1 else ed
                            else:
                                # Search in the first half
                                ed = mid if ed > st + 1 else st
            except IOError as e:
                raise IOError("Failed to open {0}: {1}".format(path, e))
        else:
            raise FileNotFoundError("Cannot find {0}".format(path))
        self.__stop = True
        return False

    def _close(self) -> bool:
        """ Close the component file. """
        if self.__file is not None:
            if not self.__file.closed:
                try:
                    self.__file.close()
                    self.__file = None
                    return True
                except IOError as e:
                    raise IOError("Failed to close: {0}".format(e))
            else:
                raise IOError("File is not open")
        else:
            raise FileNotFoundError("File does not exist")

    def is_open(self):
        return self.__file is not None and not self.__file.closed

    def next(self) -> (Optional[bytes], Optional[bytes]):
        if self.__stop:
            return None, None
        if self.__file is None or self.__file.closed:
            raise IOError("File is not open")
        if self.__pos >= self._component.actual_component_size():
            self.__stop = True
            if self.end_pos == -1:
                self.end_pos = -2
            # Reached the end
            return None, None
        self.__file.seek(self.__pos)
        key = self.__file.read(self._component.key_length())
        data = self.__file.read(self._component.data_length()) if self._component.is_primary() else None
        self.__pos += self._component.record_length()
        if (self._max_key is not None) and (key > self._max_key or (not self._include_max and key == self._max_key)):
            # With max, stop when a larger key is found
            if self.end_pos == -1:
                self.end_pos = self.__pos
            self.__stop = True
            return None, None
        self.num_read_records += 1
        return key, data

    def get_pos(self) -> int:
        return self.__pos

    def decrease_pos(self):
        self.__pos -= self.get_component().record_length()

    def get_component(self) -> SpecialDiskComponent:
        return self._component


class MemoryScanner(AbstractMemoryScanner):
    def __init__(self, component: MemoryComponent):
        super().__init__(component)
        self._component: MemoryComponent = self._component
        self.__idx = -1
        self.__keys = None
        self.__stop = False
        self.is_memory = True
        self.num_read_records = 0
        self.read_records = []

    def __del__(self):
        super().__del__()

    def is_open(self) -> bool:
        return self.__keys is not None

    def _open(self) -> bool:
        if self.__keys is None:
            self.__keys = self._component.keys()
            if self._min_key is None:
                # Start from the beginning
                self.__idx = 0
                self.__stop = False
            else:
                # Find the first record that matches the condition
                for i in range(len(self.__keys)):
                    key = self.__keys[i]
                    if key > self._min_key or (self._include_min and key == self._min_key):
                        self.__idx = i
                        break
            self.__stop = False
            return True
        else:
            self.__stop = True
            return False

    def _close(self) -> bool:
        if self.__keys is not None:
            del self.__keys
            self.__keys = None
            return True
        else:
            return False

    def next(self) -> (Optional[bytes], Optional[bytes]):
        if self.__stop:
            return None, None
        if self.__keys is None:
            raise IOError("Memory scanner is not open")
        if self.__idx >= len(self.__keys):
            self.__stop = True
            return None, None
        key = self.__keys[self.__idx]
        if (self._max_key is not None) and (key > self._max_key or (not self._include_max and key == self._max_key)):
            # With max, stop when a larger key is found
            self.__stop = True
            return None, None
        rkey, rdata = self._component.get_record(key)
        self.__idx += 1
        self.num_read_records += 1
        self.read_records.append(rkey)
        return rkey, rdata


class DiskDBScanner(AbstractDiskScanner):
    def __init__(self, component: DiskDBComponent, ridx: int):
        super().__init__(component, ridx)
        self._component: DiskDBComponent = self._component
        self.__cursor: Optional[sqlite3.Cursor] = None
        self.__conn: Optional[sqlite3.Connection] = None

    def __del__(self):
        super().__del__()

    def _open(self) -> bool:
        """ Construct a scan query and get the cursor. """
        self.__conn, self.__cursor = self._component.create_scan_cursor(self._ridx, self._min_key, self._include_min,
                                                                        self._max_key, self._include_max)
        return False if self.__cursor is None else True

    def _close(self) -> bool:
        """ Close the sqlite3 db file. """
        if self.__conn is not None:
            self._component.close_scan_cursor(self.__conn)
            self.__cursor = None
            self.__conn = None
            return True
        else:
            return False

    def is_open(self):
        return self.__cursor is not None

    def next(self) -> (Optional[bytes], Optional[bytes]):
        if self.__cursor is None:
            raise IOError("Connection is not open: {0}".format(self._component.name()))
        r = self.__cursor.fetchone()
        if r is None:
            return None, None
        else:
            return r[0], r[1] if self._component.is_primary() else None


class MemoryDBScanner(AbstractMemoryScanner):
    def __init__(self, component: MemoryDBComponent):
        super().__init__(component)
        self._component: MemoryDBComponent = self._component
        self.__cursor: Optional[sqlite3.Cursor] = None
        self.__conn: Optional[sqlite3.Connection] = None

    def __del__(self):
        super().__del__()

    def _open(self) -> bool:
        """ Construct a scan query and get the cursor. """
        self.__conn, self.__cursor = self._component.create_scan_cursor(-1, self._min_key, self._include_min,
                                                                        self._max_key, self._include_max)
        return False if self.__cursor is None else True

    def _close(self) -> bool:
        """ Close the sqlite3 db file. """
        if self.__conn is not None:
            self._component.close_scan_cursor(self.__conn)
            self.__cursor = None
            self.__conn = None
            return True
        else:
            return False

    def is_open(self):
        return self.__cursor is not None

    def next(self) -> (Optional[bytes], Optional[bytes]):
        if self.__cursor is None:
            raise IOError("Connection is not open: {0}".format(self._component.name()))
        r = self.__cursor.fetchone()
        if r is None:
            return None, None
        else:
            return r[0], r[1] if self._component.is_primary() else None


class LSMScanner:

    def __init__(self, scanners: Union[List[Scanner], Tuple[Scanner]], length: int = -1, IO_unit: int = 4):
        """
        Args:
            scanners: List or tuple of component scanners
            length: Scan length (maximum number of records returned), 0 or negative value means returned all
        """
        self.__scanners = scanners
        self.__length = length
        self.__pq = PriorityQueue()
        self.__last_key = None
        self.__is_open = False
        self.__cnt = 0
        self.IO_unit = IO_unit
        self.num_scanners = 0

    def __del__(self):
        if self.__is_open:
            self.close()
        for scanner in self.__scanners:
            del scanner
        del self.__pq

    def open(self):
        if not self.__is_open:
            self.num_scanners = len(self.__scanners)
            for idx in range(0, len(self.__scanners)):
                scanner = self.__scanners[idx]
                scanner.open()
                key, data = scanner.next()
                if key is not None:
                    if isinstance(scanner, AbstractDiskScanner):
                        self.__pq.put((key, idx, scanner.range_index(), data))
                    else:
                        self.__pq.put((key, idx, -1, data))
            self.__is_open = True

    def close(self) -> int:
        total_read_cost = 0
        if self.__is_open:
            for scanner in self.__scanners:
                # if scanner.is_memory == False:
                #     num_read = scanner.num_read_records
                #     num_read = math.ceil(num_read / self.IO_unit)
                #     total_read_cost += num_read
                    scanner.close()
            self.__is_open = False
        # total_read_cost += self.num_scanners
        return total_read_cost

    def is_open(self):
        return self.__is_open

    def scanners(self):
        return self.__scanners

    def next(self) -> (Optional[str], Optional[int], Optional[bytes], Optional[bytes]):
        while not self.__pq.empty() and \
                (self.__length < 1  # No scan length specified
                 or 0 <= self.__cnt < self.__length):  # Number of returned records is less than scan length
            key, idx, ridx, data = self.__pq.get()
            scanner = self.__scanners[idx]
            next_key, next_data = scanner.next()
            if next_key is not None:
                if isinstance(scanner, AbstractDiskScanner):
                    self.__pq.put((next_key, idx, scanner.range_index(), next_data))
                else:
                    self.__pq.put((next_key, idx, -1, next_data))
            else:
                for i in range(0, len(self.__scanners)):
                    if i != idx:
                        scanner = self.__scanners[i]
                        next_key, next_data = scanner.next()
                        if next_key is not None:
                            if isinstance(scanner, AbstractDiskScanner):
                                self.__pq.put((next_key, i, scanner.range_index(), next_data))
                            else:
                                self.__pq.put((next_key, i, -1, next_data))
                            break
            if self.__last_key is None or key > self.__last_key:
                # Otherwise they are obsolete
                self.__last_key = key
                self.__cnt += 1
                scanner = self.__scanners[idx]
                if isinstance(scanner, AbstractDiskScanner):
                    return scanner.component().name(), scanner.range_index(), key, data
                else:
                    return scanner.component().name(), -1, key, data
        return None, None, None, None
