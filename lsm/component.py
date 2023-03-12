import base64
import json
import math
import os
import sqlite3
from typing import List, Optional
from threading import Lock


class Component:
    """ Base class for LSM component """

    DISK_META_EXT: str = "meta"  # Metadata file extension
    DISK_BIN_EXT: str  = "bin"  # Binary file extension
    MEM_EXT: str       = "mem"  # Memory extension
    DISK_DB_EXT: str   = "db"  # Disk extension using SQLite3
    MEM_DB_EXT: str    = "mem_db"  # Memory extension using SQLite3

    METAKEY_NUM_RECORDS: str     = "num-records"
    METAKEY_COMPONENT_SIZE: str  = "size"
    METAKEY_MIN_KEY: str         = "min-key"
    METAKEY_MAX_KEY: str         = "max-key"
    METAKEY_VIRTUAL_RANGES: str = "v-ranges"
    METAKEY_VIRTUAL_RECORDS: str = "v-records"
    METAKEY_VIRTUAL_SIZES: str    = "v-sizes"

    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int):
        """
        Args:
            min_id: Minimum ID of the component.
            max_id: Maximum ID of the component.
            key_len: Length of a record key.
            data_len: Length of a record data.
        """
        self._min_id = min_id
        self._max_id = max_id
        self._key_len = key_len
        self._data_len = data_len
        self._record_len = key_len + data_len
        self.__num_reads = 0
        self.__num_scans = 0
        self._r_lock = Lock()
        self._min_key = None
        self._max_key = None

    def __del__(self):
        pass

    def is_primary(self):
        return self._data_len > 0

    def name(self) -> Optional[str]:
        """
        Returns:
            Name of the component.
        """
        return None

    def min_id(self) -> int:
        """
        Returns:
            Minimum component ID.
        """
        return self._min_id

    def max_id(self) -> int:
        """
        Returns:
            Maximum component ID.
        """
        return self._max_id

    def num_records(self) -> List[int]:
        """
        Returns:
            Number of records in the component.
        """
        return [0, ]

    def component_sizes(self) -> List[int]:
        """
        Returns:
            Size of the component.
        """
        return [0, ]

    def key_length(self) -> int:
        """
        Returns:
            Length of a record key.
        """
        return self._key_len

    def data_length(self) -> int:
        """
        Returns:
            Length of a record data.
        """
        return self._data_len

    def record_length(self) -> int:
        """
        Returns:
            Length of a record.
        """
        return self._record_len

    def get_record(self, key: bytes) -> (Optional[bytes], Optional[bytes]):
        """
        Get the data of a key to be searched.

        Args:
            key: The record key to be searched.
        Returns:
            The key and data of the record if found. Both None if error of record not found.
        """
        with self._r_lock:
            self.__num_reads += 1
        rkey, rdata = self._get_record(key)
        with self._r_lock:
            self.__num_reads -= 1
        return rkey, rdata

    def _get_record(self, key: bytes) -> (Optional[bytes], Optional[bytes]):
        # Override in sub-classes
        return None, None

    def min_key(self) -> Optional[bytes]:
        """
        Returns:
            The minimum record key.
        """
        return self._min_key

    def max_key(self) -> Optional[bytes]:
        """
        Returns:
            The minimum record key.
        """
        return self._max_key

    def key_ranges(self) -> List[tuple]:
        """
        Returns:
            List of key ranges (min_key, max_key)
        """
        return [(self._min_key, self._max_key), ]

    def add_scanner(self) -> int:
        with self._r_lock:
            self.__num_scans += 1
            return self.__num_scans

    def remove_scanner(self) -> int:
        with self._r_lock:
            self.__num_scans -= 1
            return self.__num_scans

    def is_reading(self) -> bool:
        """
        Returns:
            The component is being read.
        """
        return self.__num_reads > 0 or self.__num_scans > 0


class AbstractDiskComponent(Component):
    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int, base_dir: str):
        """
        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            key_len: Length of a record key.
            data_len: Length of a record data.
            base_dir: Directory that saves all the component files.
        """
        super().__init__(min_id, max_id, key_len, data_len)
        self._base_dir = os.path.abspath(base_dir)
        self._num_records = 0
        self._component_size = 0
        self._writing = False
        self._v_ranges: List[(Optional[bytes], Optional[bytes])] = [(self._min_key, self._max_key), ]
        self._v_records: List[int] = [self._num_records, ]
        self._v_sizes: List[int] = [self._component_size, ]

    def rename(self, min_id: int, max_id: int) -> bool:
        return False

    def get_binary_path(self) -> str:
        return ""

    def num_records(self) -> List[int]:
        """
        Returns:
            Number of records in the component as stored in the metadata.
        """
        return self._v_records

    def actual_num_records(self) -> int:
        """
        Returns:
            Number of records in the component.
        """
        return self._num_records

    def component_sizes(self) -> List[int]:
        """
        Returns:
            Size of the component as stored in the metadata.
        """
        return self._v_sizes

    def actual_component_size(self) -> int:
        """
        Returns:
            Size of the component.
        """
        return self._component_size

    def key_ranges(self) -> List[tuple]:
        """
        Returns:
            List of key ranges (min_key, max_key)
        """
        return self._v_ranges

    def min_min_key(self) -> Optional[bytes]:
        return self._v_ranges[0][0] if len(self._v_records) > 0 else None

    def max_min_key(self) -> Optional[bytes]:
        return self._v_ranges[-1][0] if len(self._v_records) > 0 else None

    def min_max_key(self) -> Optional[bytes]:
        return self._v_ranges[0][1] if len(self._v_records) > 0 else None

    def max_max_key(self) -> Optional[bytes]:
        return self._v_ranges[-1][1] if len(self._v_records) > 0 else None

    def open(self) -> bool:
        """ Open the component for loading.

        Returns:
            True on success.
        """
        return False

    def close(self) -> bool:
        """ Close the component for loading.

        Returns:
            True on success.
        """
        return False

    def write_key_data(self, key: bytes, data: Optional[bytes]) -> bool:
        """
        Insert / update a record with its key and data.
        This can only be called during disk component creation.
        This function is not thread-safe, make sure only one thread is used to create disk component.

        Args:
            key: The record key to be inserted.
            data: The record data to be inserted.
        Returns:
            True if success.
        """
        return False

    def write_record(self, record: bytes) -> bool:
        """
        Insert / update a record.
        This can only be called during disk component creation.
        This function is not thread-safe, make sure only one thread is used to create disk component.

        Args:
            record: The record to be inserted.
        Returns:
            True if success.
        """
        if len(record) != self._record_len:
            return False
        key = record[:self._key_len]
        data = record[self._key_len:]
        return self.write_key_data(key, data)

    def is_writing(self) -> bool:
        """
        Returns:
            The disk component is being loaded.
        """
        return self._writing

    def files(self) -> list:
        return []

    def get_meta_path(self) -> str:
        return ""

    def remove_files(self):
        if self.is_reading() or self.is_writing():
            raise IOError("The component ({0}) is being read or written".format(self.name()))
        for f in self.files():
            try:
                os.remove(f)
            except IOError as e:
                raise IOError("Failed to remove {0}: {1}".format(f, e))

    @staticmethod
    def base64_encode(bs: bytes) -> str:
        return base64.standard_b64encode(bs).decode("ascii")

    @staticmethod
    def base64_decode(s: str) -> bytes:
        return base64.standard_b64decode(s.encode("ascii"))

    def update_virtual_metadata(self, v_ranges: List[tuple], v_records: List[int]):
        """
        Update metadata to use virtual key ranges and num_recods.
        Args:
            v_ranges: Virtual ranges of (min_key, max_key)
            v_records: Virtual number of records
        """
        # if self.is_reading() or self.is_writing():
        #     raise IOError("Component {0} is being read or written".format(self.name()))
        num_ranges = len(v_ranges)
        if num_ranges == 0:
            raise ValueError("Virtual ranges keys cannot be empty")
        for i in range(num_ranges):
            v_min, v_max = v_ranges[i]
            if not isinstance(v_min, bytes) or not isinstance(v_max, bytes):
                raise TypeError("v_ranges must be list of pair of <bytes, bytes>")
            if v_min > v_max:
                raise ValueError("Invalid range at {0}: min={1} and max={2}".format(i, v_min, v_max))
            if i == 0 and v_min < self._min_key:
                raise ValueError("Minimum virtual min key ({0}) is smaller than the actual min key ({1})"
                                 .format(v_min, self._min_key))
            if i == num_ranges - 1 and v_max > self._max_key:
                raise ValueError("Maximum virtual max key ({0}) is greater than the actual max key ({1})"
                                 .format(v_max, self._max_key))
            if i < num_ranges - 1:
                next_min = v_ranges[i + 1][0]
                if v_max >= next_min:
                    raise ValueError("Invalid ranges at {0}:max={1} and {2}:min={3}".format(i, v_max, i + 1, next_min))
        lr = len(v_records)
        if lr != num_ranges:
            raise ValueError("Size mismatch of virtual number of records ({0}) and ranges ({1})"
                             .format(lr, num_ranges))
        total_vr = 0
        for i in range(num_ranges):
            vr = v_records[i]
            if vr < 1:
                raise ValueError("Invalid virtual number of records ({0}) at position {1}".format(vr, i))
            total_vr += vr
        if total_vr > self._num_records:
            raise ValueError("The total virtual number of records ({0}) is larger the actual number of records ({1})"
                             .format(total_vr, self._num_records))
        self._v_ranges = v_ranges
        self._v_records = v_records
        self._v_sizes = [self._record_len * vr for vr in v_records]
        self._save_meta()

    def is_in_range(self, key: bytes) -> bool:
        for v_min, v_max in self._v_ranges:
            if v_min <= key <= v_max:
                return True
        return False

    @staticmethod
    def is_range_overlapping(search_min: Optional[bytes], include_min: bool,
                             search_max: Optional[bytes], include_max: bool,
                             min_key: bytes, max_key: bytes) -> bool:
        if search_min is not None and (search_min > max_key or (search_min == max_key and not include_min)):
            return False
        if search_max is not None and (search_max < min_key or (search_max == min_key and not include_max)):
            return False
        return True

    def overlapping_range_indexes(self, search_min: Optional[bytes], include_min: bool,
                                  search_max: Optional[bytes], include_max: bool) -> List[int]:
        ret = []
        for i in range(len(self._v_ranges)):
            v_min, v_max = self._v_ranges[i]
            if self.is_range_overlapping(search_min, include_min, search_max, include_max, v_min, v_max):
                ret.append(i)
        return ret


class AbstractMemoryComponent(Component):
    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int, budget: int):
        """
        Args:
            min_id: Minimum ID of the memory component.
            max_id: Maximum ID of the memory component.
            key_len: Length of a record key.
            data_len: Length of a record data.
            budget: Budget of a memory component in bytes
        """
        super().__init__(min_id, max_id, key_len, data_len)
        self._max_records = int(math.floor(float(budget) / self._record_len))
        if self._max_records < 1:
            raise ValueError("Budget {0} is too low for record size {1}".format(budget, self._record_len))
        self._rw_lock = Lock()

    def __del__(self):
        del self._rw_lock

    def max_records(self) -> int:
        return self._max_records

    def is_full(self) -> bool:
        """
        Returns:
            True if the memory component is full.
        """
        return False

    def flush(self, min_id: int, max_id: int, base_dir: str) -> Optional[AbstractDiskComponent]:
        return None

    def write_key_data(self, key: bytes, data: Optional[bytes]) -> bool:
        """
        Insert / update a record with its key and data.

        Args:
            key: The record key to be inserted.
            data: The record data to be inserted.
        Returns:
            True if success.
        """
        return False

    def write_record(self, record: bytes) -> bool:
        """
        Insert / update a record.

        Args:
            record: The record to be inserted.
        Returns:
            True if success.
        """
        if len(record) != self._record_len or self.is_full():
            return False
        key = record[:self._key_len]
        data = record[self._key_len:]
        return self.write_key_data(key, data)


class DiskComponent(AbstractDiskComponent):
    """ Class for disk component """

    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int, base_dir: str):
        """
        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            key_len: Length of a record key.
            data_len: Length of a record data.
            base_dir: Directory that saves all the component files.
        """
        super().__init__(min_id, max_id, key_len, data_len, base_dir)
        self.__name = "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_BIN_EXT)
        self.__bin_path = os.path.join(self._base_dir, self.__name)
        self.__meta_path = os.path.join(self._base_dir, "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_META_EXT))
        self.__bin_file = None
        if os.path.isfile(self.__bin_path) and os.path.isfile(self.__meta_path):
            # Load disk component metadata if file exists
            with open(self.__meta_path, "r") as metaf:
                meta = json.loads(metaf.read())
            metaf.close()
            self._component_size = int(meta[Component.METAKEY_COMPONENT_SIZE])
            if self._component_size != os.path.getsize(self.__bin_path):
                raise ValueError("Component size in metadata is {0}, while the file size is {1}"
                                 .format(self._component_size, os.path.getsize(self.__bin_path)))
            if self._component_size % self._record_len != 0:
                raise ValueError("Component size {0} cannot be divided by record length {1}"
                                 .format(self._component_size, self._record_len))
            self._num_records = int(meta[Component.METAKEY_NUM_RECORDS])
            if self._num_records < 1:
                raise ValueError("Component {0} has no record".format(self.__name))
            if self._num_records != int(self._component_size / self._record_len):
                raise ValueError("Numer of records in metadata is {0} while it should be {1}"
                                 .format(self._num_records, int(self._component_size / self._record_len)))
            self._min_key = self.base64_decode(meta[Component.METAKEY_MIN_KEY])
            self._max_key = self.base64_decode(meta[Component.METAKEY_MAX_KEY])
            if len(self._min_key) != self._key_len or len(self._max_key) != self._key_len:
                raise ValueError("Min key length is {0}, max key length is {1}, while they should be {2}"
                                 .format(len(self._min_key), len(self._max_key), self._key_len))
            if self._min_key >= self._max_key:
                raise ValueError("Min key ({0}) must be smaller than max key ({1})"
                                 .format(self._min_key, self._max_key))
            if Component.METAKEY_VIRTUAL_RANGES in meta:
                self._v_ranges = [(self.base64_decode(v_min), self.base64_decode(v_max))
                                  for v_min, v_max in meta[Component.METAKEY_VIRTUAL_MIN_KEY]]
            else:
                self._v_ranges = [(self._min_key, self._max_key), ]
            if Component.METAKEY_VIRTUAL_RECORDS in meta:
                self._v_records = [int(vr) for vr in meta[Component.METAKEY_VIRTUAL_RECORDS]]
            else:
                self._v_records = [self._num_records, ]
            if Component.METAKEY_VIRTUAL_SIZES in meta:
                self._v_sizes = [int(vs) for vs in meta[Component.METAKEY_VIRTUAL_SIZES]]
            else:
                self._v_sizes = [self._component_size, ]
            num_ranges = len(self._v_ranges)
            num_vr = len(self._v_records)
            num_vs = len(self._v_sizes)
            if num_ranges < 1 or num_ranges != num_vr or num_ranges != num_vs:
                raise ValueError("Invalid virtual metadata: ranges={0}, records={1}, sizes={2}"
                                 .format(num_ranges, num_vr, num_vs))
            for i in range(num_ranges):
                v_min, v_max = self._v_ranges[i]
                if v_min > v_max:
                    raise ValueError("Invalid range at {0}: min={1} and max={2}".format(i, v_min, v_max))
                if i == 0 and v_min < self._min_key:
                    raise ValueError("Minimum virtual min key ({0}) is smaller than the actual min key ({1})"
                                     .format(v_min, self._min_key))
                if i == num_ranges - 1 and v_max > self._max_key:
                    raise ValueError("Maximum virtual max key ({0}) is greater than the actual max key ({1})"
                                     .format(v_max, self._max_key))
                if i < num_ranges - 1:
                    next_min = self._v_ranges[i + 1][0]
                    if v_max >= next_min:
                        raise ValueError("Invalid ranges at {0}:max={1} and {2}:min={3}"
                                         .format(i, v_max, i + 1, next_min))
                vr = self._v_records[i]
                if vr < 1:
                    raise ValueError("Invalid virtual number of records {0} at {1}".format(vr, i))
                vs = self._v_sizes[i]
                if vs < 1 or vs != vr * self._record_len != 0:
                    raise ValueError("Invalid virtual size {0} (should be {1}*{2}={3}) at {4}"
                                     .format(vs, self._record_len, vr, self._record_len * vr, i))
        elif os.path.isfile(self.__bin_path):
            raise FileNotFoundError("{0} exists but {1} not found".format(self.__bin_path, self.__meta_path))
        elif os.path.isfile(self.__meta_path):
            raise FileNotFoundError("{0} exists but {1} not found".format(self.__meta_path, self.__bin_path))

    def __del__(self):
        if self.__bin_file is not None:
            del self.__bin_file
        super().__del__()

    def __save_meta(self, path: str):
        meta = {
            Component.METAKEY_COMPONENT_SIZE: self._component_size,
            Component.METAKEY_NUM_RECORDS: self._num_records,
            Component.METAKEY_MIN_KEY: self.base64_encode(self._min_key),
            Component.METAKEY_MAX_KEY: self.base64_encode(self._max_key),
        }
        lsizes = len(self._v_sizes)
        if lsizes > 1 or (lsizes == 1 and self._v_sizes[0] != self._component_size):
            meta[Component.METAKEY_VIRTUAL_SIZES] = self._v_sizes
        lrecords = len(self._v_records)
        if lrecords > 1 or (lrecords == 1 and self._v_records[0] != self._num_records):
            meta[Component.METAKEY_VIRTUAL_RECORDS] = self._v_records
        l_ranges = len(self._v_ranges)
        if l_ranges > 1 or \
                (l_ranges == 1 and (self._v_ranges[0][0] != self._min_key or self._v_ranges[0][1] != self._max_key)):
            meta[Component.METAKEY_VIRTUAL_RANGES] = [[self.base64_encode(v_min), self.base64_encode(v_max)]
                                                      for v_min, v_max in self._v_ranges]
        with open(path, "w") as metaf:
            metaf.write(json.dumps(meta, separators=(",", ":"), sort_keys = True))
        metaf.close()

    def _save_meta(self):
        self.__save_meta(self.__meta_path)

    def rename(self, min_id: int, max_id: int) -> bool:
        """
        Rename the disk component using new min_id and max_id.
        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
        Returns:
            True on success.
        """
        if self.is_reading() or self.is_writing():
            raise IOError("The component ({0}) is being read or written".format(self.__name))
        new_path = self.component_path(min_id, max_id, self._base_dir)
        if os.path.isfile(new_path):
            raise FileExistsError("{0} already exists".format(new_path))
        new_meta_path = os.path.join(self._base_dir, "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_META_EXT))
        if os.path.isfile(new_meta_path):
            raise FileExistsError("{0} already exists".format(new_meta_path))
        self._min_id = min_id
        self._max_id = max_id
        os.rename(self.__bin_path, new_path)
        os.rename(self.__meta_path, new_meta_path)
        self.__name = os.path.basename(new_path)
        self.__bin_path = new_path
        self.__meta_path = new_meta_path
        return True

    @staticmethod
    def component_path(min_id: int, max_id: int, base_dir: str) -> str:
        """
        Get the data of a key to be searched.
        Be careful to call this only when there is no reads (including scans) or writes

        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            base_dir: Directory that saves all the component files.
        Returns:
            The path of the component's binary file.
        """
        return os.path.join(os.path.abspath(base_dir), "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_BIN_EXT))

    def name(self) -> Optional[str]:
        """
        Returns:
            Name of the component.
        """
        return self.__name

    def get_binary_path(self):
        """
        Returns:
            Path of the component binary file.
        """
        return self.__bin_path

    def key_pos(self, r: int) -> int:
        """ Get position of the r-th (from 0) record """
        if r < 0 or r >= self._num_records:
            return -1
        return r * self._record_len

    def _get_record(self, key: bytes) -> (Optional[bytes], Optional[bytes]):
        """
        Get the data of a key to be searched.

        Args:
            key: The record key to be searched.
        Returns:
            The key and data of the record if found. Both None if error of record not found.
        """
        if len(key) != self._key_len:
            raise KeyError("Key length is {0}, while {1} is required".format(len(key), self._key_len))
        if not os.path.isfile(self.__bin_path):
            raise FileNotFoundError("File not found: {0}".format(self.__bin_path))
        if self._writing:
            raise IOError("The component ({0}) is being written".format(self.__name))
        if not self.is_in_range(key):
            raise KeyError("The search key {0} is not in any of the component's key ranges [{1}]"
                           .format(key.hex(), ", ".join(["[{0}, {1}]".format(v_min.hex(), v_max.hex())
                                                         for v_min, v_max in self._v_ranges])))
        with open(self.__bin_path, "rb") as rf:
            st = 0  # Binary search start
            ed = self._num_records - 1  # Binary search end
            while st <= ed:
                mid = int(math.floor((ed + st) / 2))
                p = self.key_pos(mid)
                rf.seek(p)  # Seek to the beginning position of the mid record
                rkey = rf.read(self._key_len)  # Read certain bytes as key
                if rkey == key:
                    rdata = rf.read(self._data_len) if self.is_primary() else None  # Read certain bytes as data
                    rf.close()
                    return rkey, rdata
                else:
                    if st == ed:
                        # Record cannot be found
                        rf.close()
                        return None, None
                    if rkey < key:
                        # Search in the second half
                        st = mid if ed > st + 1 else ed
                    else:
                        # Search in the first half
                        ed = mid if ed > st + 1 else st
        rf.close()
        return None, None  # Not found

    def open(self) -> bool:
        """ Open the component's binary file for loading.

        Returns:
            True on success.
        """
        if os.path.isfile(self.__bin_path):
            return False
        self._writing = True
        try:
            self.__bin_file = open(self.__bin_path, "wb")
            return True
        except IOError as e:
            self._writing = False
            raise IOError("Failed to open {0}: {1}".format(self.__name, e))

    def close(self) -> bool:
        """ Close the component's binary file for loading.

        Returns:
            True on success.
        """
        if self._writing and self.__bin_file is not None and not self.__bin_file.closed:
            try:
                self.__bin_file.close()
                del self.__bin_file
                self.__bin_file = None
                self._v_records = [self._num_records, ]
                self._v_sizes = [self._component_size, ]
                self._v_ranges = [(self._min_key, self._max_key), ]
                self.__save_meta(self.__meta_path)
                self._writing = False
                return True
            except IOError as e:
                try:
                    self.__bin_file.flush()
                    del self.__bin_file
                    self.__bin_file = None
                    self.__save_meta(self.__meta_path)
                    self._writing = False
                    raise IOError("Failed to close {0}: {1}".format(self.__name, e))
                except IOError as e:
                    raise IOError("Failed to flush {0}: {1}".format(self.__name, e))
        return False

    def write_key_data(self, key: bytes, data: Optional[bytes]) -> bool:
        """
        Insert / update a record with its key and data.
        This can only be called during disk component creation.
        This function is not thread-safe, make sure only one thread is used to create disk component.

        Args:
            key: The record key to be inserted.
            data: The record data to be inserted.
        Returns:
            True if success.
        """
        if not self.is_writing():
            raise IOError("The component ({0}) is not open for writing".format(self.__name))
        if self.is_reading():
            raise IOError("The component ({0}) is being read".format(self.__name))
        if self.__bin_file is None:
            raise IOError("{0} is not open".format(self.__bin_path))
        if self.__bin_file.closed:
            raise IOError("{0} is closed".format(self.__bin_path))
        if len(key) != self._key_len:
            raise ValueError("Invalid key of length {0}, while {1} is required".format(len(key), self._key_len))
        if data is None or len(data) == 0:
            if self.is_primary():
                raise ValueError("Data of length {0} is required".format(self._data_len))
        else:
            if len(data) != self._data_len:
                raise ValueError("Invalid data of length {0}, while {1} is required".format(len(data), self._data_len))
        if self._data_len == 0:
            self.__bin_file.write(key)
        else:
            self.__bin_file.write(key + data)
        self._component_size += self._record_len
        self._num_records += 1
        if self._min_key is None or key < self._min_key:
            self._min_key = key
        if self._max_key is None or key > self._max_key:
            self._max_key = key
        return True

    def files(self) -> list:
        return [self.__bin_path, self.__meta_path]

    def get_meta_path(self) -> str:
        return self.__meta_path


class MemoryComponent(AbstractMemoryComponent):
    """ Class for in-memory component """

    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int, budget: int):
        """
        Args:
            min_id: Minimum ID of the memory component.
            max_id: Maximum ID of the memory component.
            key_len: Length of a record key.
            data_len: Length of a record data.
            budget: Budget of a memory component in bytes
        """
        super().__init__(min_id, max_id, key_len, data_len, budget)
        self.__name = "{0}-{1}.{2}".format(min_id, max_id, Component.MEM_EXT)
        if self.is_primary():
            self.__records = {}  # Use dict to store records as <key, data> pairs
        else:
            self.__records = set()

    def __del__(self):
        del self.__records
        super().__del__()

    def name(self) -> Optional[str]:
        """
        Returns:
            Name of the component.
        """
        return self.__name

    def num_records(self) -> List[int]:
        """
        Returns:
            Number of records in the component.
        """
        return [len(self.__records), ]

    def component_sizes(self) -> List[int]:
        """
        Returns:
            Size of the component.
        """
        return [len(self.__records) * self._record_len, ]

    def is_full(self) -> bool:
        """
        Returns:
            True if the memory component is full.
        """
        return len(self.__records) == self._max_records

    def _get_record(self, key: bytes) -> (Optional[bytes], Optional[bytes]):
        """
        Get the data of a key to be searched.

        Args:
            key: The record key to be searched.
        Returns:
            The key and data of the record if found. Both None if error of record not found.
        """
        if len(key) != self._key_len:
            raise KeyError("Key length is {0}, while {1} is required".format(len(key), self._key_len))
        if len(self.__records) == 0:
            return None, None
        if key < self._min_key or key > self._max_key:
            raise KeyError("The search key {0} is not in the component's key range [{1}, {2}]"
                           .format(key.hex(), self._min_key.hex(), self._max_key.hex()))
        with self._rw_lock:
            if self.is_primary():
                rdata = self.__records.get(key, None)
                rkey = None if rdata is None else key
            else:
                rkey = key if key in self.__records else None
                rdata = None
        return rkey, rdata

    def keys(self):
        if self.is_primary():
            return tuple(sorted(self.__records.keys()))
        else:
            return tuple(sorted(list(self.__records)))

    def write_key_data(self, key: bytes, data: Optional[bytes]) -> bool:
        """
        Insert / update a record with its key and data.

        Args:
            key: The record key to be inserted.
            data: The record data to be inserted.
        Returns:
            True if success.
        """
        if len(key) != self._key_len:
            raise ValueError("Invalid key of length {0}, while {1} is required".format(len(key), self._key_len))
        if data is None or len(data) == 0:
            if self.is_primary():
                raise ValueError("Data of length {0} is required".format(self._data_len))
        else:
            if len(data) != self._data_len:
                raise ValueError("Invalid data of length {0}, while {1} is required".format(len(data), self._data_len))
        if self.is_full():
            raise IOError("The memory component is full")
        with self._rw_lock:
            if self.is_primary():
                self.__records[key] = data
            else:
                self.__records.add(key)
            if self._min_key is None or key < self._min_key:
                self._min_key = key
            if self._max_key is None or key > self._max_key:
                self._max_key = key
        return True

    def flush(self, min_id: int, max_id: int, base_dir: str) -> Optional[DiskComponent]:
        """
        Flush the memory component to a disk component.

        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            base_dir: Directory that saves all the component files.
        Returns:
            The flushed disk component.
        """
        path = DiskComponent.component_path(min_id, max_id, base_dir)
        if os.path.isfile(path):
            raise FileExistsError("{0} already exists".format(os.path.basename(path)))
        with self._rw_lock:
            d = DiskComponent(min_id, max_id, self._key_len, self._data_len, base_dir)
            d.open()
            if self.is_primary():
                for key in sorted(self.__records.keys()):
                    d.write_key_data(key, self.__records[key])
            else:
                for key in sorted(self.__records):
                    d.write_key_data(key, None)
            d.close()
            return d


class DBComponent:
    """ Base class for component using SQLite3 """
    CONTENT_TABLE_NAME = "content"
    KEY_KEY = "key"
    KEY_DATA = "data"

    def create_scan_cursor(self, ridx: int, min_key: Optional[bytes], include_min: bool,
                           max_key: Optional[bytes], include_max: bool) \
            -> (Optional[sqlite3.Connection], Optional[sqlite3.Cursor]):
        return None, None

    def close_scan_cursor(self, conn: sqlite3.Connection) -> None:
        if conn is not None:
            conn.close()
            del conn

    def _new_connection(self) -> Optional[sqlite3.Connection]:
        return None


class DiskDBComponent(AbstractDiskComponent, DBComponent):
    """ Class for disk component using SQLite3 """

    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int, base_dir: str):
        """
        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            key_len: Length of a record key.
            data_len: Length of a record data.
            base_dir: Directory that saves all the component files.
        """
        super().__init__(min_id, max_id, key_len, data_len, base_dir)
        self.__name = "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_DB_EXT)
        self.__db_path = os.path.join(self._base_dir, self.__name)
        self.__meta_path = os.path.join(self._base_dir, "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_META_EXT))
        self.__conn: Optional[sqlite3.Connection] = None
        if self.is_primary():
            self.__point_query = "SELECT {0} FROM {1} WHERE {2} = ?;" \
                .format(DBComponent.KEY_DATA, DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY)
            self.__insert_query = "INSERT INTO {0} ({1}, {2}) VALUES (?, ?);" \
                .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY, DBComponent.KEY_DATA)
        else:
            self.__point_query = "SELECT {0} FROM {1} WHERE {0} = ?;" \
                .format(DBComponent.KEY_KEY, DBComponent.CONTENT_TABLE_NAME)
            self.__insert_query = "INSERT INTO {0} ({1}) VALUES (?);" \
                .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY)
        if os.path.isfile(self.__db_path) and os.path.isfile(self.__meta_path):
            with open(self.__meta_path, "r") as metaf:
                meta = json.loads(metaf.read())
            metaf.close()
            self._component_size = int(meta[Component.METAKEY_COMPONENT_SIZE])
            if self._component_size % self._record_len != 0:
                raise ValueError("Component size {0} cannot be divided by record length {1}"
                                 .format(self._component_size, self._record_len))
            self._num_records = int(meta[Component.METAKEY_NUM_RECORDS])
            if self._num_records < 1:
                raise ValueError("Component {0} has no record".format(self.__name))
            if self._num_records != int(self._component_size / self._record_len):
                raise ValueError("Numer of records in metadata is {0} while it should be {1}"
                                 .format(self._num_records, int(self._component_size / self._record_len)))
            self._min_key = self.base64_decode(meta[Component.METAKEY_MIN_KEY])
            self._max_key = self.base64_decode(meta[Component.METAKEY_MAX_KEY])
            if len(self._min_key) != self._key_len or len(self._max_key) != self._key_len:
                raise ValueError("Min key length is {0}, max key length is {1}, while they should be {2}"
                                 .format(len(self._min_key), len(self._max_key), self._key_len))
            if self._min_key >= self._max_key:
                raise ValueError("Min key ({0}) must be smaller than max key ({1})"
                                 .format(self._min_key, self._max_key))
            if Component.METAKEY_VIRTUAL_RANGES in meta:
                self._v_ranges = [(self.base64_decode(v_min), self.base64_decode(v_max))
                                  for v_min, v_max in meta[Component.METAKEY_VIRTUAL_MIN_KEY]]
            else:
                self._v_ranges = [(self._min_key, self._max_key), ]
            if Component.METAKEY_VIRTUAL_RECORDS in meta:
                self._v_records = [int(vr) for vr in meta[Component.METAKEY_VIRTUAL_RECORDS]]
            else:
                self._v_records = [self._num_records, ]
            if Component.METAKEY_VIRTUAL_SIZES in meta:
                self._v_sizes = [int(vs) for vs in meta[Component.METAKEY_VIRTUAL_SIZES]]
            else:
                self._v_sizes = [self._component_size, ]
            num_ranges = len(self._v_ranges)
            num_vr = len(self._v_records)
            num_vs = len(self._v_sizes)
            if num_ranges < 1 or num_ranges != num_vr or num_ranges != num_vs:
                raise ValueError("Invalid virtual metadata: ranges={0}, records={1}, sizes={2}"
                                 .format(num_ranges, num_vr, num_vs))
            for i in range(num_ranges):
                v_min, v_max = self._v_ranges[i]
                if v_min > v_max:
                    raise ValueError("Invalid range at {0}: min={1} and max={2}".format(i, v_min, v_max))
                if i == 0 and v_min < self._min_key:
                    raise ValueError("Minimum virtual min key ({0}) is smaller than the actual min key ({1})"
                                     .format(v_min, self._min_key))
                if i == num_ranges - 1 and v_max > self._max_key:
                    raise ValueError("Maximum virtual max key ({0}) is greater than the actual max key ({1})"
                                     .format(v_max, self._max_key))
                if i < num_ranges - 1:
                    next_min = self._v_ranges[i + 1][0]
                    if v_max >= next_min:
                        raise ValueError("Invalid ranges at {0}:max={1} and {2}:min={3}"
                                         .format(i, v_max, i + 1, next_min))
                vr = self._v_records[i]
                if vr < 1:
                    raise ValueError("Invalid virtual number of records {0} at {1}".format(vr, i))
                vs = self._v_sizes[i]
                if vs < 1 or vs != vr * self._record_len != 0:
                    raise ValueError("Invalid virtual size {0} (should be {1}*{2}={3}) at {4}"
                                     .format(vs, self._record_len, vr, self._record_len * vr, i))
        elif os.path.isfile(self.__db_path):
            raise FileNotFoundError("{0} exists but {1} not found".format(self.__db_path, self.__meta_path))
        elif os.path.isfile(self.__meta_path):
            raise FileNotFoundError("{0} exists but {1} not found".format(self.__meta_path, self.__db_path))

    def __del__(self):
        if self.__conn is not None:
            self.__conn.close()
            del self.__conn
        super().__del__()

    def __save_meta(self, path: str):
        meta = {
            Component.METAKEY_COMPONENT_SIZE: self._component_size,
            Component.METAKEY_NUM_RECORDS: self._num_records,
            Component.METAKEY_MIN_KEY: self.base64_encode(self._min_key),
            Component.METAKEY_MAX_KEY: self.base64_encode(self._max_key),
        }
        lsizes = len(self._v_sizes)
        if lsizes > 1 or (lsizes == 1 and self._v_sizes[0] != self._component_size):
            meta[Component.METAKEY_VIRTUAL_SIZES] = self._v_sizes
        lrecords = len(self._v_records)
        if lrecords > 1 or (lrecords == 1 and self._v_records[0] != self._num_records):
            meta[Component.METAKEY_VIRTUAL_RECORDS] = self._v_records
        l_ranges = len(self._v_ranges)
        if l_ranges > 1 or \
                (l_ranges == 1 and (self._v_ranges[0][0] != self._min_key or self._v_ranges[0][1] != self._max_key)):
            meta[Component.METAKEY_VIRTUAL_RANGES] = [[self.base64_encode(v_min), self.base64_encode(v_max)]
                                                      for v_min, v_max in self._v_ranges]
        with open(path, "w") as metaf:
            metaf.write(json.dumps(meta, separators=(",", ":"), sort_keys = True))
        metaf.close()

    def _save_meta(self):
        self.__save_meta(self.__meta_path)

    def rename(self, min_id: int, max_id: int) -> bool:
        """
        Rename the disk component using new min_id and max_id.
        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
        Returns:
            True on success.
        """
        if self.is_reading() or self.is_writing():
            raise IOError("The component is being read or written")
        new_path = self.component_path(min_id, max_id, self._base_dir)
        if os.path.isfile(new_path):
            raise FileExistsError("{0} already exists".format(new_path))
        new_meta_path = os.path.join(self._base_dir, "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_META_EXT))
        if os.path.isfile(new_meta_path):
            raise FileExistsError("{0} already exists".format(new_meta_path))
        self._min_id = min_id
        self._max_id = max_id
        os.rename(self.__db_path, new_path)
        os.rename(self.__meta_path, new_meta_path)
        self.__name = os.path.basename(new_path)
        self.__db_path = new_path
        self.__meta_path = new_meta_path
        return True

    @staticmethod
    def component_path(min_id: int, max_id: int, base_dir: str) -> str:
        """
        Get the data of a key to be searched.
        Be careful to call this only when there is no reads (including scans) or writes

        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            base_dir: Directory that saves all the component files.
        Returns:
            The path of the component's binary file.
        """
        return os.path.join(os.path.abspath(base_dir), "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_DB_EXT))

    def name(self) -> Optional[str]:
        """
        Returns:
            Name of the component.
        """
        return self.__name

    def get_binary_path(self):
        """
        Returns:
            Path of the component binary file.
        """
        return self.__db_path

    def _get_record(self, key: bytes) -> (Optional[bytes], Optional[bytes]):
        """
        Get the data of a key to be searched.

        Args:
            key: The record key to be searched.
        Returns:
            The key and data of the record if found. Both None if error of record not found.
        """
        if len(key) != self._key_len:
            raise KeyError("Key length is {0}, while {1} is required".format(len(key), self._key_len))
        if not os.path.isfile(self.__db_path):
            raise FileNotFoundError("File not found: {0}".format(self.__db_path))
        if self._writing:
            raise IOError("The component ({0}) is being written".format(self.__name))
        if not self.is_in_range(key):
            raise KeyError("The search key {0} is not in any of the component's key ranges [{1}]"
                           .format(key.hex(), ", ".join(["[{0}, {1}]".format(v_min.hex(), v_max.hex())
                                                         for v_min, v_max in self._v_ranges])))
        conn: sqlite3.Connection = sqlite3.connect(self.__create_uri(self.__db_path, "ro"), uri=True)
        cur = conn.cursor()
        r = cur.execute(self.__point_query, (sqlite3.Binary(key), )).fetchone()
        conn.close()
        if r is None:
            return None, None
        else:
            return key, r[0] if self.is_primary() else None

    def __create_uri(self, path: str, mode: str) -> str:
        if os.name == "nt":
            return "file:/{0}?mode={1}".format(path.replace("\\", "/"), mode)
        else:
            return "file:{0}?mode={1}".format(path, mode)

    def open(self) -> bool:
        """ Open the component's binary file for loading.

        Returns:
            True on success.
        """
        if os.path.isfile(self.__db_path):
            return False
        try:
            self._writing = True
            uri = self.__create_uri(self.__db_path, "rwc")
            try:
                self.__conn: sqlite3.Connection = sqlite3.connect(uri, uri=True)
            except sqlite3.OperationalError as e:
                raise sqlite3.OperationalError("Error connect to {0}: {1}".format(uri, e))
            cur = self.__conn.cursor()
            if self.is_primary():
                cur.execute("CREATE TABLE {0} ({1} BLOB NOT NULL PRIMARY KEY, {2} BLOB NOT NULL);"
                            .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY, DBComponent.KEY_DATA))
            else:
                cur.execute("CREATE TABLE {0} ({1} BLOB NOT NULL PRIMARY KEY);"
                            .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY))
            self.__conn.commit()
            return True
        except IOError as e:
            raise IOError("Failed to open {0}: {1}".format(self.__name, e))

    def close(self) -> bool:
        """ Close the component's binary file for loading.

        Returns:
            True on success.
        """
        if self.__conn is not None and self._writing:
            cur = self.__conn.cursor()
            if self._num_records % 1000 != 0:
                cur.execute("COMMIT;")
            self.__conn.commit()
            self.__conn.close()
            del self.__conn
            self.__conn = None
            self._v_records = [self._num_records, ]
            self._v_sizes = [self._component_size, ]
            self._v_ranges = [(self._min_key, self._max_key), ]
            self.__save_meta(self.__meta_path)
            self._writing = False
            return True
        return False

    def write_key_data(self, key: bytes, data: Optional[bytes]) -> bool:
        """
        Insert / update a record with its key and data.
        This can only be called during disk component creation.
        This function is not thread-safe, make sure only one thread is used to create disk component.

        Args:
            key: The record key to be inserted.
            data: The record data to be inserted.
        Returns:
            True if success.
        """
        if not self.is_writing():
            raise IOError("The component ({0}) is not open for writing".format(self.__name))
        if self.is_reading():
            raise IOError("The component ({0}) is being read".format(self.__name))
        if self.__conn is None:
            raise IOError("Connection is not open")
        if len(key) != self._key_len:
            raise ValueError("Invalid key of length {0}, while {1} is required".format(len(key), self._key_len))
        if data is None or len(data) == 0:
            if self.is_primary():
                raise ValueError("Data of length {0} is required".format(self._data_len))
        else:
            if len(data) != self._data_len:
                raise ValueError("Invalid data of length {0}, while {1} is required".format(len(data), self._data_len))
        cur = self.__conn.cursor()
        if self._num_records % 1000 == 0:
            cur.execute("BEGIN TRANSACTION;")
        if self.is_primary():
            cur.execute(self.__insert_query, (sqlite3.Binary(key), sqlite3.Binary(data)))
        else:
            cur.execute(self.__insert_query, (sqlite3.Binary(key),))
        self._num_records += 1
        self._component_size += self._record_len
        if self._min_key is None or key < self._min_key:
            self._min_key = key
        if self._max_key is None or key > self._max_key:
            self._v_max_key = self._max_key
        if self._num_records % 1000 == 0:
            cur.execute("COMMIT;")
            self.__conn.commit()
        return True

    def create_scan_cursor(self, ridx: int, min_key: Optional[bytes], include_min: bool,
                           max_key: Optional[bytes], include_max: bool) \
            -> (Optional[sqlite3.Connection], Optional[sqlite3.Cursor]):
        if ridx < 0 or ridx >= len(self._v_ranges):
            raise IndexError("Invalid virtual range index {0}, max={1}".format(ridx, len(self._v_ranges)))
        v_min, v_max = self._v_ranges[ridx]
        if min_key is not None and (min_key > v_max or (min_key == v_max and not include_min)):
            return None, None
        if max_key is not None and (max_key < v_min or (max_key == v_min and not include_max)):
            return None, None
        conn = self._new_connection()
        if conn is None:
            return None, None
        if self.is_primary():
            select = "SELECT {0}, {1} FROM {2}".format(DBComponent.KEY_KEY, DBComponent.KEY_DATA,
                                                       DBComponent.CONTENT_TABLE_NAME)
        else:
            select = "SELECT {0} FROM {1}".format(DBComponent.KEY_KEY, DBComponent.CONTENT_TABLE_NAME)
        orderby = " ORDER BY {0} ASC;".format(DBComponent.KEY_KEY)
        if min_key is None and max_key is None:
            condition = "{0} >= ? AND {0} <= ?".format(DBComponent.KEY_KEY)
            values = (sqlite3.Binary(v_min), sqlite3.Binary(v_max))
        elif min_key is not None and max_key is not None:
            if v_min <= min_key and not include_min:
                condition_min = "{0} > ?".format(DBComponent.KEY_KEY)
                value_min = sqlite3.Binary(min_key)
            else:
                condition_min = "{0} >= ?".format(DBComponent.KEY_KEY)
                value_min = sqlite3.Binary(max(min_key, v_min))
            if v_max >= max_key and not include_max:
                condition_max = "{0} < ?".format(DBComponent.KEY_KEY)
                value_max = sqlite3.Binary(max_key)
            else:
                condition_max = "{0} <= ?".format(DBComponent.KEY_KEY)
                value_max = sqlite3.Binary(max(max_key, v_max))
            condition = " WHERE {0} AND {1}".format(condition_min, condition_max)
            values = (value_min, value_max)
        elif min_key is None and max_key is not None:
            # and max_key is not None is unnecessary, however this can avoid IDE warning.
            if v_max >= max_key and not include_max:
                condition = " WHERE {0} >= ? AND {0} < ?".format(DBComponent.KEY_KEY)
                values = (sqlite3.Binary(v_min), sqlite3.Binary(max_key))
            else:
                condition = " WHERE {0} >= ? AND {0} <= ?".format(DBComponent.KEY_KEY)
                values = (sqlite3.Binary(v_min), sqlite3.Binary(max(max_key, v_max)))
        else:
            if v_min <= min_key and not include_min:
                condition = " WHERE {0} > ? AND {0} <= ?".format(DBComponent.KEY_KEY)
                values = (sqlite3.Binary(min_key), sqlite3.Binary(v_max))
            else:
                condition = " WHERE {0} >= ? AND {0} <= ?".format(DBComponent.KEY_KEY)
                values = (sqlite3.Binary(max(min_key, v_min)), sqlite3.Binary(v_max))
        query = "{0}{1}{2}".format(select, condition, orderby)
        return conn, conn.cursor().execute(query, values)

    def _new_connection(self) -> Optional[sqlite3.Connection]:
        uri = self.__create_uri(self.__db_path, "ro")
        return sqlite3.connect(uri, uri=True)

    def files(self) -> list:
        return [self.__db_path, self.__meta_path]

    def get_meta_path(self) -> str:
        return self.__meta_path


class MemoryDBComponent(AbstractMemoryComponent, DBComponent):
    """ Class for in-memory component using SQLite3 """

    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int, budget: int):
        """
        Args:
            min_id: Minimum ID of the memory component.
            max_id: Maximum ID of the memory component.
            key_len: Length of a record key.
            data_len: Length of a record data.
            budget: Budget of a memory component in bytes
        """
        super().__init__(min_id, max_id, key_len, data_len, budget)
        self.__name = "{0}-{1}.{2}".format(min_id, max_id, Component.MEM_DB_EXT)
        self.__num_records = 0
        self.__conn: sqlite3.Connection = sqlite3.connect(":memory:")
        if self.is_primary():
            self.__conn.cursor().execute("CREATE TABLE {0} ({1} BLOB NOT NULL PRIMARY KEY, {2} BLOB NOT NULL);"
                                         .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY,
                                                 DBComponent.KEY_DATA))
        else:
            self.__conn.cursor().execute("CREATE TABLE {0} ({1} BLOB NOT NULL PRIMARY KEY);"
                                         .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY))
        self.__conn.commit()
        if self.is_primary():
            self.__point_query = "SELECT {0} FROM {1} WHERE {2} = ?;" \
                .format(DBComponent.KEY_DATA, DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY)
            self.__insert_query = "INSERT INTO {0} ({1}, {2}) VALUES (?, ?);" \
                .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY, DBComponent.KEY_DATA)
            self.__update_query = "UPDATE {0} SET {1} = ? WHERE {2} = ?;" \
                .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_DATA, DBComponent.KEY_KEY)
        else:
            self.__point_query = "SELECT {0} FROM {1} WHERE {0} = ?;" \
                .format(DBComponent.KEY_KEY, DBComponent.CONTENT_TABLE_NAME)
            self.__insert_query = "INSERT INTO {0} ({1}) VALUES (?);" \
                .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY)
            self.__update_query = ""  # Secondary index does not support update

    def __del__(self):
        if self.__conn is not None:
            self.__conn.close()
            del self.__conn
        super().__del__()

    def name(self) -> Optional[str]:
        """
        Returns:
            Name of the component.
        """
        return self.__name

    def num_records(self) -> List[int]:
        """
        Returns:
            Number of records in the component.
        """
        return [self.__num_records, ]

    def component_sizes(self) -> List[int]:
        """
        Returns:
            Size of the component.
        """
        return [self.__num_records * self._record_len, ]

    def is_full(self) -> bool:
        """
        Returns:
            True if the memory component is full.
        """
        return self.__num_records == self._max_records

    def _get_record(self, key: bytes) -> (Optional[bytes], Optional[bytes]):
        """
        Get the data of a key to be searched.

        Args:
            key: The record key to be searched.
        Returns:
            The key and data of the record if found. Both None if error of record not found.
        """
        if self.__conn is None:
            raise IOError("Connection is not open")
        if len(key) != self._key_len:
            raise KeyError("Key length is {0}, while {1} is required".format(len(key), self._key_len))
        if self.__num_records == 0:
            raise IOError("Memory component is empty")
        if key < self._min_key or key > self._max_key:
            raise KeyError("The search key {0} is not in the component's key range [{1}, {2}]"
                           .format(key.hex(), self._min_key.hex(), self._max_key.hex()))
        with self._rw_lock:
            r = self.__conn.cursor().execute(self.__point_query, (sqlite3.Binary(key), )).fetchone()
            if r is None:
                return None, None
            else:
                return key, r[0] if self.is_primary() else None

    def write_key_data(self, key: bytes, data: Optional[bytes]) -> bool:
        """
        Insert / update a record with its key and data.

        Args:
            key: The record key to be inserted.
            data: The record data to be inserted.
        Returns:
            True if success.
        """
        if self.__conn is None:
            raise IOError("Connection is not open")
        if len(key) != self._key_len:
            raise ValueError("Invalid key of length {0}, while {1} is required".format(len(key), self._key_len))
        if data is None or len(data) == 0:
            if self.is_primary():
                raise ValueError("Data of length {0} is required".format(self._data_len))
        else:
            if len(data) != self._data_len:
                raise ValueError("Invalid data of length {0}, while {1} is required".format(len(data), self._data_len))
        if self.is_full():
            raise IOError("The memory component is full")
        cur = self.__conn.cursor()
        with self._rw_lock:
            if cur.execute(self.__point_query, (sqlite3.Binary(key),)).fetchone() is None:
                if self.is_primary():
                    cur.execute(self.__insert_query, (sqlite3.Binary(key), sqlite3.Binary(data)))
                else:
                    cur.execute(self.__insert_query, (sqlite3.Binary(key), ))
                if self._min_key is None or key < self._min_key:
                    self._min_key = key
                if self._max_key is None or key > self._max_key:
                    self._max_key = key
                self.__num_records += 1
            elif self.is_primary():
                cur.execute(self.__update_query, (sqlite3.Binary(data), sqlite3.Binary(key)))
            # else:
                # Do nothing to update secondary index
        return True

    def flush(self, min_id: int, max_id: int, base_dir: str) -> Optional[DiskDBComponent]:
        """
        Flush the memory component to a disk component.

        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            base_dir: Directory that saves all the component files.
        Returns:
            The flushed disk component.
        """
        path = DiskDBComponent.component_path(min_id, max_id, base_dir)
        if os.path.isfile(path):
            raise FileExistsError("{0} already exists".format(os.path.basename(path)))
        d = DiskDBComponent(min_id, max_id, self._key_len, self._data_len, base_dir)
        d.open()
        with self._rw_lock:
            if self.is_primary():
                scan_cur = self.__conn.cursor().execute("SELECT {0}, {1} FROM {2} ORDER BY {0} ASC;"
                                                        .format(DBComponent.KEY_KEY, DBComponent.KEY_DATA,
                                                                DBComponent.CONTENT_TABLE_NAME))
                while True:
                    r = scan_cur.fetchone()
                    if r is None:
                        break
                    d.write_key_data(r[0], r[1])
            else:
                scan_cur = self.__conn.cursor().execute("SELECT {0} FROM {1} ORDER BY {0} ASC;"
                                                        .format(DBComponent.KEY_KEY, DBComponent.CONTENT_TABLE_NAME))
                while True:
                    r = scan_cur.fetchone()
                    if r is None:
                        break
                    d.write_key_data(r[0], None)
        d.close()
        return d

    def create_scan_cursor(self, ridx: int, min_key: Optional[bytes], include_min: bool,
                           max_key: Optional[bytes], include_max: bool) \
            -> (Optional[sqlite3.Connection], Optional[sqlite3.Cursor]):
        if self.__conn is None:
            return None, None
        if min_key is not None and (min_key > self._max_key or (min_key == self._max_key and not include_min)):
            return None, None
        if max_key is not None and (max_key < self._min_key or (max_key == self._min_key and not include_max)):
            return None, None
        conn = self._new_connection()
        if conn is None:
            return None, None
        if self.is_primary():
            select = "SELECT {0}, {1} FROM {2}".format(DBComponent.KEY_KEY, DBComponent.KEY_DATA,
                                                       DBComponent.CONTENT_TABLE_NAME)
        else:
            select = "SELECT {0} FROM {1}".format(DBComponent.KEY_KEY, DBComponent.CONTENT_TABLE_NAME)
        orderby = " ORDER BY {0} ASC;".format(DBComponent.KEY_KEY)
        if min_key is None and max_key is None:
            condition = ""
            values = ()
        elif min_key is not None and max_key is not None:
            if self.min_key() <= min_key and not include_min:
                condition_min = "{0} > ?".format(DBComponent.KEY_KEY)
                value_min = sqlite3.Binary(min_key)
            else:
                condition_min = "{0} >= ?".format(DBComponent.KEY_KEY)
                value_min = sqlite3.Binary(max(min_key, self.min_key()))
            if self.max_key() >= max_key and not include_max:
                condition_max = "{0} < ?".format(DBComponent.KEY_KEY)
                value_max = sqlite3.Binary(max_key)
            else:
                condition_max = "{0} <= ?".format(DBComponent.KEY_KEY)
                value_max = sqlite3.Binary(max(max_key, self.max_key()))
            condition = " WHERE {0} AND {1}".format(condition_min, condition_max)
            values = (value_min, value_max)
        elif min_key is None and max_key is not None:
            # and max_key is not None is unnecessary, however this can avoid IDE warning.
            if self.max_key() >= max_key and not include_max:
                condition = " WHERE {0} < ?".format(DBComponent.KEY_KEY)
                values = (sqlite3.Binary(max_key), )
            else:
                condition = " WHERE {0} <= ?".format(DBComponent.KEY_KEY)
                values = (sqlite3.Binary(max(max_key, self.max_key())), )
        else:
            if self.min_key() <= min_key and not include_min:
                condition = " WHERE {0} > ?".format(DBComponent.KEY_KEY)
                values = (sqlite3.Binary(min_key), )
            else:
                condition = " WHERE {0} >= ?".format(DBComponent.KEY_KEY)
                values = (sqlite3.Binary(max(min_key, self.min_key())), )
        query = "{0}{1}{2}".format(select, condition, orderby)
        return conn, conn.cursor().execute(query, values)

    def _new_connection(self) -> Optional[sqlite3.Connection]:
        return self.__conn

    def close_scan_cursor(self, conn: sqlite3.Connection) -> None:
        # Do nothing
        return

class SpecialDiskComponent(AbstractDiskComponent):
    """ Class for disk component """

    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int, base_dir: str, is_special: bool):
        """
        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            key_len: Length of a record key.
            data_len: Length of a record data.
            base_dir: Directory that saves all the component files.
        """
        super().__init__(min_id, max_id, key_len, data_len, base_dir)
        self.is_special = is_special
        self.__name = "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_BIN_EXT)
        self.__bin_path = os.path.join(self._base_dir, self.__name)
        self.__meta_path = os.path.join(self._base_dir, "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_META_EXT))
        self.__bin_file = None
        if os.path.isfile(self.__bin_path) and os.path.isfile(self.__meta_path):
            # Load disk component metadata if file exists
            with open(self.__meta_path, "r") as metaf:
                meta = json.loads(metaf.read())
            metaf.close()
            self._component_size = int(meta[Component.METAKEY_COMPONENT_SIZE])
            if self._component_size != os.path.getsize(self.__bin_path):
                raise ValueError("Component size in metadata is {0}, while the file size is {1}"
                                 .format(self._component_size, os.path.getsize(self.__bin_path)))
            if self._component_size % self._record_len != 0:
                raise ValueError("Component size {0} cannot be divided by record length {1}"
                                 .format(self._component_size, self._record_len))
            self._num_records = int(meta[Component.METAKEY_NUM_RECORDS])
            if self._num_records < 1:
                raise ValueError("Component {0} has no record".format(self.__name))
            if self._num_records != int(self._component_size / self._record_len):
                raise ValueError("Numer of records in metadata is {0} while it should be {1}"
                                 .format(self._num_records, int(self._component_size / self._record_len)))
            self._min_key = self.base64_decode(meta[Component.METAKEY_MIN_KEY])
            self._max_key = self.base64_decode(meta[Component.METAKEY_MAX_KEY])
            if len(self._min_key) != self._key_len or len(self._max_key) != self._key_len:
                raise ValueError("Min key length is {0}, max key length is {1}, while they should be {2}"
                                 .format(len(self._min_key), len(self._max_key), self._key_len))
            if self._min_key >= self._max_key:
                raise ValueError("Min key ({0}) must be smaller than max key ({1})"
                                 .format(self._min_key, self._max_key))
            if Component.METAKEY_VIRTUAL_RANGES in meta:
                self._v_ranges = [(self.base64_decode(v_min), self.base64_decode(v_max))
                                  for v_min, v_max in meta[Component.METAKEY_VIRTUAL_MIN_KEY]]
            else:
                self._v_ranges = [(self._min_key, self._max_key), ]
            if Component.METAKEY_VIRTUAL_RECORDS in meta:
                self._v_records = [int(vr) for vr in meta[Component.METAKEY_VIRTUAL_RECORDS]]
            else:
                self._v_records = [self._num_records, ]
            if Component.METAKEY_VIRTUAL_SIZES in meta:
                self._v_sizes = [int(vs) for vs in meta[Component.METAKEY_VIRTUAL_SIZES]]
            else:
                self._v_sizes = [self._component_size, ]
            num_ranges = len(self._v_ranges)
            num_vr = len(self._v_records)
            num_vs = len(self._v_sizes)
            if num_ranges < 1 or num_ranges != num_vr or num_ranges != num_vs:
                raise ValueError("Invalid virtual metadata: ranges={0}, records={1}, sizes={2}"
                                 .format(num_ranges, num_vr, num_vs))
            for i in range(num_ranges):
                v_min, v_max = self._v_ranges[i]
                if v_min > v_max:
                    raise ValueError("Invalid range at {0}: min={1} and max={2}".format(i, v_min, v_max))
                if i == 0 and v_min < self._min_key:
                    raise ValueError("Minimum virtual min key ({0}) is smaller than the actual min key ({1})"
                                     .format(v_min, self._min_key))
                if i == num_ranges - 1 and v_max > self._max_key:
                    raise ValueError("Maximum virtual max key ({0}) is greater than the actual max key ({1})"
                                     .format(v_max, self._max_key))
                if i < num_ranges - 1:
                    next_min = self._v_ranges[i + 1][0]
                    if v_max >= next_min:
                        raise ValueError("Invalid ranges at {0}:max={1} and {2}:min={3}"
                                         .format(i, v_max, i + 1, next_min))
                vr = self._v_records[i]
                if vr < 1:
                    raise ValueError("Invalid virtual number of records {0} at {1}".format(vr, i))
                vs = self._v_sizes[i]
                if vs < 1 or vs != vr * self._record_len != 0:
                    raise ValueError("Invalid virtual size {0} (should be {1}*{2}={3}) at {4}"
                                     .format(vs, self._record_len, vr, self._record_len * vr, i))
        elif os.path.isfile(self.__bin_path):
            raise FileNotFoundError("{0} exists but {1} not found".format(self.__bin_path, self.__meta_path))
        elif os.path.isfile(self.__meta_path):
            raise FileNotFoundError("{0} exists but {1} not found".format(self.__meta_path, self.__bin_path))

    def __del__(self):
        if self.__bin_file is not None:
            del self.__bin_file
        super().__del__()

    def __save_meta(self, path: str):
        meta = {
            Component.METAKEY_COMPONENT_SIZE: self._component_size,
            Component.METAKEY_NUM_RECORDS: self._num_records,
            Component.METAKEY_MIN_KEY: self.base64_encode(self._min_key),
            Component.METAKEY_MAX_KEY: self.base64_encode(self._max_key),
        }
        lsizes = len(self._v_sizes)
        if lsizes > 1 or (lsizes == 1 and self._v_sizes[0] != self._component_size):
            meta[Component.METAKEY_VIRTUAL_SIZES] = self._v_sizes
        lrecords = len(self._v_records)
        if lrecords > 1 or (lrecords == 1 and self._v_records[0] != self._num_records):
            meta[Component.METAKEY_VIRTUAL_RECORDS] = self._v_records
        l_ranges = len(self._v_ranges)
        if l_ranges > 1 or \
                (l_ranges == 1 and (self._v_ranges[0][0] != self._min_key or self._v_ranges[0][1] != self._max_key)):
            meta[Component.METAKEY_VIRTUAL_RANGES] = [[self.base64_encode(v_min), self.base64_encode(v_max)]
                                                      for v_min, v_max in self._v_ranges]
        with open(path, "w") as metaf:
            metaf.write(json.dumps(meta, separators=(",", ":"), sort_keys=True))
        metaf.close()

    def _save_meta(self):
        self.__save_meta(self.__meta_path)

    def rename(self, min_id: int, max_id: int) -> bool:
        """
        Rename the disk component using new min_id and max_id.
        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
        Returns:
            True on success.
        """
        if self.is_reading() or self.is_writing():
            raise IOError("The component ({0}) is being read or written".format(self.__name))
        new_path = self.component_path(min_id, max_id, self._base_dir)
        if os.path.isfile(new_path):
            raise FileExistsError("{0} already exists".format(new_path))
        new_meta_path = os.path.join(self._base_dir, "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_META_EXT))
        if os.path.isfile(new_meta_path):
            raise FileExistsError("{0} already exists".format(new_meta_path))
        self._min_id = min_id
        self._max_id = max_id
        os.rename(self.__bin_path, new_path)
        os.rename(self.__meta_path, new_meta_path)
        self.__name = os.path.basename(new_path)
        self.__bin_path = new_path
        self.__meta_path = new_meta_path
        return True

    @staticmethod
    def component_path(min_id: int, max_id: int, base_dir: str) -> str:
        """
        Get the data of a key to be searched.
        Be careful to call this only when there is no reads (including scans) or writes

        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            base_dir: Directory that saves all the component files.
        Returns:
            The path of the component's binary file.
        """
        return os.path.join(os.path.abspath(base_dir), "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_BIN_EXT))

    def name(self) -> Optional[str]:
        """
        Returns:
            Name of the component.
        """
        return self.__name

    def get_binary_path(self):
        """
        Returns:
            Path of the component binary file.
        """
        return self.__bin_path

    def key_pos(self, r: int) -> int:
        """ Get position of the r-th (from 0) record """
        if r < 0 or r >= self._num_records:
            return -1
        return r * self._record_len

    def _get_record(self, key: bytes) -> (Optional[bytes], Optional[bytes]):
        """
        Get the data of a key to be searched.

        Args:
            key: The record key to be searched.
        Returns:
            The key and data of the record if found. Both None if error of record not found.
        """
        if len(key) != self._key_len:
            raise KeyError("Key length is {0}, while {1} is required".format(len(key), self._key_len))
        if not os.path.isfile(self.__bin_path):
            raise FileNotFoundError("File not found: {0}".format(self.__bin_path))
        if self._writing:
            raise IOError("The component ({0}) is being written".format(self.__name))
        if not self.is_in_range(key):
            raise KeyError("The search key {0} is not in any of the component's key ranges [{1}]"
                           .format(key.hex(), ", ".join(["[{0}, {1}]".format(v_min.hex(), v_max.hex())
                                                         for v_min, v_max in self._v_ranges])))
        with open(self.__bin_path, "rb") as rf:
            st = 0  # Binary search start
            ed = self._num_records - 1  # Binary search end
            while st <= ed:
                mid = int(math.floor((ed + st) / 2))
                p = self.key_pos(mid)
                rf.seek(p)  # Seek to the beginning position of the mid record
                rkey = rf.read(self._key_len)  # Read certain bytes as key
                if rkey == key:
                    rdata = rf.read(self._data_len) if self.is_primary() else None  # Read certain bytes as data
                    rf.close()
                    return rkey, rdata
                else:
                    if st == ed:
                        # Record cannot be found
                        rf.close()
                        return None, None
                    if rkey < key:
                        # Search in the second half
                        st = mid if ed > st + 1 else ed
                    else:
                        # Search in the first half
                        ed = mid if ed > st + 1 else st
        rf.close()
        return None, None  # Not found

    def open(self) -> bool:
        """ Open the component's binary file for loading.

        Returns:
            True on success.
        """
        if os.path.isfile(self.__bin_path):
            return False
        self._writing = True
        try:
            self.__bin_file = open(self.__bin_path, "wb")
            return True
        except IOError as e:
            self._writing = False
            raise IOError("Failed to open {0}: {1}".format(self.__name, e))

    def close(self) -> bool:
        """ Close the component's binary file for loading.

        Returns:
            True on success.
        """
        if self._writing and self.__bin_file is not None and not self.__bin_file.closed:
            try:
                self.__bin_file.close()
                del self.__bin_file
                self.__bin_file = None
                self._v_records = [self._num_records, ]
                self._v_sizes = [self._component_size, ]
                self._v_ranges = [(self._min_key, self._max_key), ]
                self.__save_meta(self.__meta_path)
                self._writing = False
                return True
            except IOError as e:
                try:
                    self.__bin_file.flush()
                    del self.__bin_file
                    self.__bin_file = None
                    self.__save_meta(self.__meta_path)
                    self._writing = False
                    raise IOError("Failed to close {0}: {1}".format(self.__name, e))
                except IOError as e:
                    raise IOError("Failed to flush {0}: {1}".format(self.__name, e))
        return False

    def write_key_data(self, key: bytes, data: Optional[bytes]) -> bool:
        """
        Insert / update a record with its key and data.
        This can only be called during disk component creation.
        This function is not thread-safe, make sure only one thread is used to create disk component.

        Args:
            key: The record key to be inserted.
            data: The record data to be inserted.
        Returns:
            True if success.
        """
        if not self.is_writing():
            raise IOError("The component ({0}) is not open for writing".format(self.__name))
        if self.is_reading():
            raise IOError("The component ({0}) is being read".format(self.__name))
        if self.__bin_file is None:
            raise IOError("{0} is not open".format(self.__bin_path))
        if self.__bin_file.closed:
            raise IOError("{0} is closed".format(self.__bin_path))
        if len(key) != self._key_len:
            raise ValueError("Invalid key of length {0}, while {1} is required".format(len(key), self._key_len))
        if data is None or len(data) == 0:
            if self.is_primary():
                raise ValueError("Data of length {0} is required".format(self._data_len))
        else:
            if len(data) != self._data_len:
                raise ValueError("Invalid data of length {0}, while {1} is required".format(len(data), self._data_len))
        if self._data_len == 0:
            self.__bin_file.write(key)
        else:
            self.__bin_file.write(key + data)
        self._component_size += self._record_len
        self._num_records += 1
        if self._min_key is None or key < self._min_key:
            self._min_key = key
        if self._max_key is None or key > self._max_key:
            self._max_key = key
        return True

    def files(self) -> list:
        return [self.__bin_path, self.__meta_path]

    def get_meta_path(self) -> str:
        return self.__meta_path

class SpecialMemoryComponent(AbstractMemoryComponent):
    """ Class for in-memory component """

    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int, budget: int):
        """
        Args:
            min_id: Minimum ID of the memory component.
            max_id: Maximum ID of the memory component.
            key_len: Length of a record key.
            data_len: Length of a record data.
            budget: Budget of a memory component in bytes
        """
        super().__init__(min_id, max_id, key_len, data_len, budget)
        self.__name = "{0}-{1}.{2}".format(min_id, max_id, Component.MEM_EXT)
        self.__records = {}  # Use dict to store records as <key, data> pairs
        self.invalid_ranges = []

    def __del__(self):
        del self.__records
        super().__del__()

    def name(self) -> Optional[str]:
        """
        Returns:
            Name of the component.
        """
        return self.__name

    def num_records(self) -> int:
        """
        Returns:
            Number of records in the component.
        """
        return [len(self.__records), ]

    def component_size(self) -> int:
        """
        Returns:
            Size of the component.
        """
        return [len(self.__records) * self._record_len, ]

    def is_full(self) -> bool:
        """
        Returns:
            True if the memory component is full.
        """
        return len(self.__records) == self._max_records

    def _get_record(self, key: bytes) -> Optional[bytes]:
        """
        Get the data of a key to be searched.

        Args:
            key: The record key to be searched.
        Returns:
            The key and data of the record if found. Both None if error of record not found.
        """
        if len(key) != self._key_len:
            raise KeyError("Key length is {0}, while {1} is required".format(len(key), self._key_len))
        if len(self.__records) == 0:
            return None, None
        if key < self._min_key or key > self._max_key:
            raise KeyError("The search key {0} is not in the component's key range [{1}, {2}]"
                           .format(key.hex(), self._min_key.hex(), self._max_key.hex()))
        with self._rw_lock:
            if self.is_primary():
                rdata = self.__records.get(key, None)
                rkey = None if rdata is None else key
            else:
                rkey = key if key in self.__records else None
                rdata = None
        return rkey, rdata

    def keys(self):
        return tuple(sorted(self.__records.keys()))

    def records(self):
        return self.__records

    def write_key_data(self, key: bytes, data: bytes) -> bool:
        """
        Insert / update a record with its key and data.

        Args:
            key: The record key to be inserted.
            data: The record data to be inserted.
        Returns:
            True if success.
        """
        if len(key) != self._key_len or len(data) != self._data_len or self.is_full():
            return False
        with self._rw_lock:
            self.__records[key] = data
            if self._min_key is None or key < self._min_key:
                self._min_key = key
            if self._max_key is None or key > self._max_key:
                self._max_key = key
        return True

    def delete_key_data(self, key: bytes) -> bool:
        if key in self.__records:
            del self.__records[key]
        # del self.__records[key]
        return True

    # def flush(self, min_id: int, max_id: int, base_dir: str, is_special: bool) -> Optional[SpecialDiskComponent]:
    #     """
    #     Flush the memory component to a disk component.
    #
    #     Args:
    #         min_id: Minimum ID of the disk component.
    #         max_id: Maximum ID of the disk component.
    #         base_dir: Directory that saves all the component files.
    #     Returns:
    #         The flushed disk component.
    #     """
    #     path = SpecialDiskComponent.component_path(min_id, max_id, base_dir)
    #     if os.path.isfile(path):
    #         raise FileExistsError("{0} already exists".format(os.path.basename(path)))
    #     with self._rw_lock:
    #         d = SpecialDiskComponent(min_id, max_id, self._key_len, self._data_len, base_dir, is_special)
    #         d.open()
    #         for key in sorted(self.__records.keys()):
    #             d.write_key_data(key, self.__records[key])
    #         d.close()
    #         return d

    def flush(self, min_id: int, max_id: int, base_dir: str, is_special: bool) -> Optional[SpecialDiskComponent]:
        """
        Flush the memory component to a disk component.

        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            base_dir: Directory that saves all the component files.
        Returns:
            The flushed disk component.
        """
        path = SpecialDiskComponent.component_path(min_id, max_id, base_dir)
        if os.path.isfile(path):
            raise FileExistsError("{0} already exists".format(os.path.basename(path)))
        with self._rw_lock:
            d = SpecialDiskComponent(min_id, max_id, self._key_len, self._data_len, base_dir, is_special)
            d.open()
            for key in sorted(self.__records.keys()):
                d.write_key_data(key, self.__records[key])
            d.close()
            # if len(self.invalid_ranges) > 0:
            #     invalid_ranges = sorted(self.invalid_ranges)
            #     idx_invalid_ranges = 0
            #     invalid_min, invalid_max = invalid_ranges[idx_invalid_ranges]
            #     min_key = -1
            #     num = 0
            #     valid_range = []
            #     num_records = []
            #     pre_max_key = None
            #     for key in sorted(self.__records.keys()):
            #         d.write_key_data(key, self.__records[key])
            #         if min_key == -1:
            #             min_key = key
            #         if invalid_max != -1 and key > invalid_max:
            #             max_key = pre_max_key
            #             valid_range.append((min_key, max_key))
            #             num_records.append(num)
            #             min_key = key
            #             num = 1
            #             idx_invalid_ranges += 1
            #             if len(invalid_ranges) > idx_invalid_ranges:
            #                 invalid_min, invalid_max = invalid_ranges[idx_invalid_ranges]
            #             else:
            #                 invalid_max = -1
            #         else:
            #             num += 1
            #         pre_max_key = key
            #     valid_range.append((min_key, pre_max_key))
            #     num_records.append(num)
            #     d.close()
            #     d._v_ranges = valid_range
            #     d._v_records = num_records
            #     d._save_meta()
            #     # print("new flush ", d.name(), ", its key ranges = ", d.key_ranges(), ", num_records = ", d.num_records())
            # else:
            #     for key in sorted(self.__records.keys()):
            #         d.write_key_data(key, self.__records[key])
            #     d.close()
            return d

class SpecialDiskDBComponent(AbstractDiskComponent, DBComponent):
    """ Class for disk component using SQLite3 """

    def __init__(self, min_id: int, max_id: int, key_len: int, data_len: int, base_dir: str):
        """
        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            key_len: Length of a record key.
            data_len: Length of a record data.
            base_dir: Directory that saves all the component files.
        """
        super().__init__(min_id, max_id, key_len, data_len, base_dir)
        self.__name = "{0}-{1}-{2}.{3}".format(min_id, max_id, "m" , Component.DISK_DB_EXT)
        self.__db_path = os.path.join(self._base_dir, self.__name)
        self.__meta_path = os.path.join(self._base_dir, "{0}-{1}-{2}.{3}".format(min_id, max_id, "m" , Component.DISK_META_EXT))
        self.__conn: Optional[sqlite3.Connection] = None
        self.__point_query = "SELECT {0} FROM {1} WHERE {2} = ?;" \
            .format(DBComponent.KEY_DATA, DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY)
        self.__insert_query = "INSERT INTO {0} ({1}, {2}) VALUES (?, ?);" \
            .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY, DBComponent.KEY_DATA)
        if os.path.isfile(self.__db_path) and os.path.isfile(self.__meta_path):
            with open(self.__meta_path, "r") as metaf:
                meta = json.loads(metaf.read())
            metaf.close()
            self._component_size = int(meta[Component.METAKEY_COMPONENT_SIZE])
            if self._component_size % self._record_len != 0:
                raise ValueError("Component size {0} cannot be divided by record length {1}"
                                 .format(self._component_size, self._record_len))
            self._num_records = int(meta[Component.METAKEY_NUM_RECORDS])
            if self._num_records < 1:
                raise ValueError("Component {0} has no record".format(self.__name))
            if self._num_records != int(self._component_size / self._record_len):
                raise ValueError("Numer of records in metadata is {0} while it should be {1}"
                                 .format(self._num_records, int(self._component_size / self._record_len)))
            self._min_key = self.base64_decode(meta[Component.METAKEY_MIN_KEY])
            self._max_key = self.base64_decode(meta[Component.METAKEY_MAX_KEY])
            if len(self._min_key) != self._key_len or len(self._max_key) != self._key_len:
                raise ValueError("Min key length is {0}, max key length is {1}, while they should be {2}"
                                 .format(len(self._min_key), len(self._max_key), self._key_len))
            if self._min_key >= self._max_key:
                raise ValueError("Min key ({0}) must be smaller than max key ({1})"
                                 .format(self._min_key, self._max_key))
            if Component.METAKEY_VIRTUAL_MIN_KEY in meta:
                self._v_min_key = self.base64_decode(meta[Component.METAKEY_VIRTUAL_MIN_KEY])
                if self._v_min_key < self._min_key:
                    raise ValueError("Virtual min key ({0}) is smaller than actual min key ({1})"
                                     .format(self._v_min_key, self._min_key))
            else:
                self._v_min_key = self._min_key
            if Component.METAKEY_VIRTUAL_MAX_KEY in meta:
                self._v_max_key = self.base64_decode(meta[Component.METAKEY_VIRTUAL_MAX_KEY])
                if self._v_max_key > self._max_key:
                    raise ValueError("Virtual max key ({0}) is larger than actual max key ({1})"
                                     .format(self._v_max_key, self._max_key))
            else:
                self._v_max_key = self._max_key
            if self._v_min_key > self._v_max_key:
                raise ValueError("Virtual min key ({0}) is larger than virtual max key ({1})"
                                 .format(self._v_min_key, self._v_max_key))
            if Component.METAKEY_VIRTUAL_RECORDS in meta:
                self._v_records = int(meta[Component.METAKEY_VIRTUAL_RECORDS])
                if self._v_records > self._num_records:
                    raise ValueError("Virtual number of records ({0}) is larger than actual number of records ({1})"
                                     .format(self._v_records, self._num_records))
            else:
                self._v_records = self._num_records
            if Component.METAKEY_VIRTUAL_SIZE in meta:
                self._v_size = int(meta[Component.METAKEY_VIRTUAL_SIZE])
                if self._v_size > self._component_size:
                    raise ValueError("Virtual component size ({0}) is larger than actual component size ({1})"
                                     .format(self._v_size, self._component_size))
            else:
                self._v_size = self._component_size
            if self._v_records * self._record_len != self._v_size:
                raise ValueError("Virtual numer of records ({0}) * record length ({1}) != virtual component size ({2})"
                                 .format(self._v_records, self._record_len, self._v_size))
        elif os.path.isfile(self.__db_path):
            raise FileNotFoundError("{0} exists but {1} not found".format(self.__db_path, self.__meta_path))
        elif os.path.isfile(self.__meta_path):
            raise FileNotFoundError("{0} exists but {1} not found".format(self.__meta_path, self.__db_path))

    def __del__(self):
        if self.__conn is not None:
            self.__conn.close()
            del self.__conn
        super().__del__()

    def __save_meta(self, path: str):
        meta = {
            Component.METAKEY_COMPONENT_SIZE: self._component_size,
            Component.METAKEY_NUM_RECORDS: self._num_records,
            Component.METAKEY_MIN_KEY: self.base64_encode(self._min_key),
            Component.METAKEY_MAX_KEY: self.base64_encode(self._max_key),
        }
        if self._v_size != self._component_size:
            meta[Component.METAKEY_VIRTUAL_SIZE] = self._v_size
        if self._v_records != self._num_records:
            meta[Component.METAKEY_VIRTUAL_RECORDS] = self._v_records
        if self._v_min_key != self._min_key:
            meta[Component.METAKEY_VIRTUAL_MIN_KEY] = self.base64_encode(self._v_min_key)
        if self._v_max_key != self._max_key:
            meta[Component.METAKEY_VIRTUAL_MAX_KEY] = self.base64_encode(self._v_max_key)
        with open(path, "w") as metaf:
            metaf.write(json.dumps(meta, separators=(",", ":"), sort_keys = True))
        metaf.close()

    def _save_meta(self):
        self.__save_meta(self.__meta_path)

    def rename(self, min_id: int, max_id: int) -> bool:
        """
        Rename the disk component using new min_id and max_id.
        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
        Returns:
            True on success.
        """
        if self.is_reading() or self.is_writing():
            raise IOError("The component is being read or written")
        new_path = self.component_path(min_id, max_id, self._base_dir)
        if os.path.isfile(new_path):
            raise FileExistsError("{0} already exists".format(new_path))
        new_meta_path = os.path.join(self._base_dir, "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_META_EXT))
        if os.path.isfile(new_meta_path):
            raise FileExistsError("{0} already exists".format(new_meta_path))
        self._min_id = min_id
        self._max_id = max_id
        os.rename(self.__db_path, new_path)
        os.rename(self.__meta_path, new_meta_path)
        self.__name = os.path.basename(new_path)
        self.__db_path = new_path
        self.__meta_path = new_meta_path
        return True

    @staticmethod
    def component_path(min_id: int, max_id: int, base_dir: str) -> str:
        """
        Get the data of a key to be searched.
        Be careful to call this only when there is no reads (including scans) or writes

        Args:
            min_id: Minimum ID of the disk component.
            max_id: Maximum ID of the disk component.
            base_dir: Directory that saves all the component files.
        Returns:
            The path of the component's binary file.
        """
        return os.path.join(os.path.abspath(base_dir), "{0}-{1}.{2}".format(min_id, max_id, Component.DISK_DB_EXT))

    def name(self) -> Optional[str]:
        """
        Returns:
            Name of the component.
        """
        return self.__name

    def get_binary_path(self):
        """
        Returns:
            Path of the component binary file.
        """
        return self.__db_path

    def _get_record(self, key: bytes) -> Optional[bytes]:
        """
        Get the data of a key to be searched.

        Args:
            key: The record key to be searched.
        Returns:
            The data of the record if found. None if error of record not found.
        """
        if len(key) != self._key_len or not os.path.isfile(self.__db_path) or self._writing:
            return None  # Error
        if key < self._min_key or key > self._max_key:
            return None  # Not in range
        conn: sqlite3.Connection = sqlite3.connect(self.__create_uri(self.__db_path, "ro"), uri=True)
        cur = conn.cursor()
        r = cur.execute(self.__point_query, (sqlite3.Binary(key),)).fetchone()
        conn.close()
        if r is None:
            return None
        else:
            return r[0]

    def __create_uri(self, path: str, mode: str) -> str:
        if os.name == "nt":
            return "file:/{0}?mode={1}".format(path.replace("\\", "/"), mode)
        else:
            return "file:{0}?mode={1}".format(path, mode)

    def open(self) -> bool:
        """ Open the component's binary file for loading.

        Returns:
            True on success.
        """
        if os.path.isfile(self.__db_path):
            return False
        try:
            self._writing = True
            uri = self.__create_uri(self.__db_path, "rwc")
            try:
                self.__conn: sqlite3.Connection = sqlite3.connect(uri, uri=True)
            except sqlite3.OperationalError as e:
                raise sqlite3.OperationalError("Error connect to {0}: {1}".format(uri, e))
            cur = self.__conn.cursor()
            cur.execute("CREATE TABLE {0} ({1} BLOB NOT NULL PRIMARY KEY, {2} BLOB NOT NULL);"
                        .format(DBComponent.CONTENT_TABLE_NAME, DBComponent.KEY_KEY, DBComponent.KEY_DATA))
            self.__conn.commit()
            return True
        except IOError as e:
            raise IOError("Failed to open {0}: {1}".format(self.__name, e))

    def close(self) -> bool:
        """ Close the component's binary file for loading.

        Returns:
            True on success.
        """
        if self.__conn is not None and self._writing:
            cur = self.__conn.cursor()
            if self._num_records % 1000 != 0:
                cur.execute("COMMIT;")
            self.__conn.commit()
            self.__conn.close()
            del self.__conn
            self.__conn = None
            self.__save_meta(self.__meta_path)
            self._writing = False
            return True
        return False

    def write_key_data(self, key: bytes, data: bytes) -> bool:
        """
        Insert / update a record with its key and data.
        This can only be called during disk component creation.
        This function is not thread-safe, make sure only one thread is used to create disk component.

        Args:
            key: The record key to be inserted.
            data: The record data to be inserted.
        Returns:
            True if success.
        """
        if not self.is_writing() or self.is_reading() or self.__conn is None:
            return False
        if len(key) != self._key_len or len(data) != self._data_len:
            return False
        cur = self.__conn.cursor()
        if self._num_records % 1000 == 0:
            cur.execute("BEGIN TRANSACTION;")
        cur.execute(self.__insert_query, (sqlite3.Binary(key), sqlite3.Binary(data)))
        self._num_records += 1
        self._component_size += self._record_len
        self._v_size = self._component_size
        self._v_records = self._num_records
        if self._min_key is None or key < self._min_key:
            self._min_key = key
            self._v_min_key = self._min_key
        if self._max_key is None or key > self._max_key:
            self._max_key = key
            self._v_max_key = self._max_key
        if self._num_records % 1000 == 0:
            cur.execute("COMMIT;")
            self.__conn.commit()
        return True

    def create_scan_cursor(self, min_key: Optional[bytes], include_min: bool,
                           max_key: Optional[bytes], include_max: bool) \
            -> (Optional[sqlite3.Connection], Optional[sqlite3.Cursor]):
        if min_key is not None and (min_key > self.max_key() or (min_key == self.max_key() and not include_min)):
            return None, None
        if max_key is not None and (max_key < self.min_key() or (max_key == self.min_key() and not include_max)):
            return None, None
        conn = self._new_connection()
        if conn is None:
            return None, None
        sign_min = ">=" if include_min else ">"
        sign_max = "<=" if include_max else "<"
        if min_key is None and max_key is None:
            query = "SELECT {0}, {1} FROM {2} ORDER BY {0} ASC;".format(DBComponent.KEY_KEY, DBComponent.KEY_DATA,
                                                                        DBComponent.CONTENT_TABLE_NAME)
            values = ()
        elif min_key is not None and max_key is not None:
            query = "SELECT {0}, {1} FROM {2} WHERE {0} {3} ? AND {0} {4} ? ORDER BY {0} ASC;" \
                .format(DBComponent.KEY_KEY, DBComponent.KEY_DATA, DBComponent.CONTENT_TABLE_NAME, sign_min, sign_max)
            values = (sqlite3.Binary(max(min_key, self.min_key())), sqlite3.Binary(min(max_key, self.max_key())))
        elif min_key is None and max_key is not None:
            # and max_key is not None is unnecessary, however this can avoid IDE warning.
            query = "SELECT {0}, {1} FROM {2} WHERE {0} {3} ? ORDER BY {0} ASC;" \
                .format(DBComponent.KEY_KEY, DBComponent.KEY_DATA, DBComponent.CONTENT_TABLE_NAME, sign_max)
            values = (sqlite3.Binary(min(max_key, self.max_key())), )
        else:
            query = "SELECT {0}, {1} FROM {2} WHERE {0} {3} ? ORDER BY {0} ASC;" \
                .format(DBComponent.KEY_KEY, DBComponent.KEY_DATA, DBComponent.CONTENT_TABLE_NAME, sign_min)
            values = (sqlite3.Binary(max(min_key, self.min_key())), )
        return conn, conn.cursor().execute(query, values)

    def _new_connection(self) -> Optional[sqlite3.Connection]:
        uri = self.__create_uri(self.__db_path, "ro")
        return sqlite3.connect(uri, uri=True)

    def files(self) -> list:
        return [self.__db_path, self.__meta_path]

    def get_meta_path(self) -> str:
        return self.__meta_path