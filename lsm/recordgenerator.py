import os
import sys
import uuid
import hashlib
from typing import Optional
from threading import Lock


class RecordGenerator:
    """ Base class for record generator """

    def __init__(self, start: int = 0):
        """
        Args:
            start: Start (Number of records already exists)
        """
        self._start = start

    @staticmethod
    def random_data(length: int) -> bytes:
        """
        Return random data of given length

        Args:
             length: Length of the data.
        """
        if length < 1:
            raise ValueError("Length ({0}) must be at least 1".format(length))
        return os.urandom(length)

    def bytes_length(self) -> int:
        return -1

    def hex_length(self) -> int:
        return -1

    def next(self) -> Optional[bytes]:
        """
        Returns:
            The next key in bytes.
        """
        return None

    def previous(self) -> Optional[bytes]:
        """
        Returns:
            The previous key in bytes.
        """
        return None


class SequentialNumberGenerator(RecordGenerator):
    """ Sequential numbers in unsigned long type (8 bytes) with padding in the front.  """

    def __init__(self, start: int = 0, length: int = -1):
        """
        Args:
            start: Start (Number of records already exists)
            length: Total length
        """
        super().__init__(start)
        self._len = length
        self._number = start
        self._lock = Lock()
        self._previous = None
        self._next = None
        self.__bytes_len = 8 if length <= 8 else length

    def bytes_length(self) -> int:
        return self.__bytes_len

    @staticmethod
    def int_to_bytes(n: int) -> bytes:
        return n.to_bytes(8, sys.byteorder, signed=False)

    def __add_padding(self, bs: bytes) -> bytes:
        bl = len(bs)
        if bl >= self._len:
            return bs
        padding = b"\0" * (self._len - bl)
        return padding + bs

    def next(self) -> bytes:
        with self._lock:
            self._previous = self._next
            self._number += 1
            self._next = self.__add_padding(self.int_to_bytes(self._number))
            return self._next

    def previous(self) -> Optional[bytes]:
        return self._previous

    def current(self) -> int:
        return self._number

    def bytes_from_number(self, n: int) -> Optional[bytes]:
        return self.__add_padding(self.int_to_bytes(n))


class SequentialNumericStringGenerator(SequentialNumberGenerator):
    """ Sequential numbers in unsigned long type (8 bytes) with padding in the front as string. """

    def __init__(self, start: int = 0, length: int = -1):
        """
        Args:
            start: Start (Number of records already exists)
            length: Total length
        """
        super().__init__(start, length)

    def __add_padding(self, ss: str) -> str:
        sl = len(ss)
        if sl >= self._len:
            return ss
        padding = "0" * (self._len - sl)
        return padding + ss

    def next(self) -> bytes:
        with self._lock:
            self._previous = bytes(self.__add_padding(str(self._number)), "ascii")
            self._number += 1
            return bytes(self.__add_padding(str(self._number)), "ascii")

    def bytes_from_number(self, n: int) -> Optional[bytes]:
        return bytes(self.__add_padding(str(n)), "ascii")


class UUIDGenerator(RecordGenerator):
    """ Generate UUIDs  """

    def __init__(self):
        super().__init__(-1)
        self.__lock = Lock()
        self.__previous = None
        self.__next = None
        self.__bytes_len = len(uuid.uuid4().bytes)

    def next(self) -> bytes:
        with self.__lock:
            self.__previous = self.__next
            self.__next = uuid.uuid4().bytes
            return self.__next

    def bytes_length(self) -> int:
        return self.__bytes_len

    def previous(self) -> Optional[bytes]:
        return self.__previous


class HashedNumberGenerator(SequentialNumberGenerator):
    """ Generate key based on hash function. """

    def __init__(self, start: int = 0):
        """
        Args:
            start: Start (Number of records already exists)
        """
        super().__init__(start, -1)
        self._digest_size = -1

    def bytes_length(self) -> int:
        return self._digest_size

    def hash_number(self, n: int) -> Optional[bytes]:
        return None

    def next(self) -> bytes:
        with self._lock:
            self._previous = self._next
            self._number += 1
            self._next = self.hash_number(self._number)
            return self._next

    def bytes_from_number(self, n: int) -> Optional[bytes]:
        return self.hash_number(n)


class MD5Generator(HashedNumberGenerator):
    """ Generate key based on MD5 hash. """

    def __init__(self, start: int = 0):
        """
        Args:
            start: Start (Number of records already exists)
        """
        super().__init__(start)
        self._digest_size = hashlib.md5().digest_size

    def hash_number(self, n: int) -> bytes:
        return hashlib.md5(self.int_to_bytes(n)).digest()


class SHA1Generator(HashedNumberGenerator):
    """ Generate key based on SHA1 hash. """

    def __init__(self, start: int = 0):
        """
        Args:
            start: Start (Number of records already exists)
        """
        super().__init__(start)
        self._digest_size = hashlib.sha1().digest_size

    def hash_number(self, n: int) -> bytes:
        return hashlib.sha1(self.int_to_bytes(n)).digest()


class SHA224Generator(HashedNumberGenerator):
    """ Generate key based on SHA1 hash. """

    def __init__(self, start: int = 0):
        """
        Args:
            start: Start (Number of records already exists)
        """
        super().__init__(start)
        self._digest_size = hashlib.sha224().digest_size

    def hash_number(self, n: int) -> bytes:
        return hashlib.sha224(self.int_to_bytes(n)).digest()


class SHA256Generator(HashedNumberGenerator):
    """ Generate key based on SHA1 hash. """

    def __init__(self, start: int = 0):
        """
        Args:
            start: Start (Number of records already exists)
        """
        super().__init__(start)
        self._digest_size = hashlib.sha256().digest_size

    def hash_number(self, n: int) -> bytes:
        return hashlib.sha256(self.int_to_bytes(n)).digest()


class SHA384Generator(HashedNumberGenerator):
    """ Generate key based on SHA1 hash. """

    def __init__(self, start: int = 0):
        """
        Args:
            start: Start (Number of records already exists)
        """
        super().__init__(start)
        self._digest_size = hashlib.sha384().digest_size

    def hash_number(self, n: int) -> bytes:
        return hashlib.sha384(self.int_to_bytes(n)).digest()


class SHA512Generator(HashedNumberGenerator):
    """ Generate key based on SHA1 hash. """

    def __init__(self, start: int = 0):
        """
        Args:
            start: Start (Number of records already exists)
        """
        super().__init__(start)
        self._digest_size = hashlib.sha512().digest_size

    def hash_number(self, n: int) -> bytes:
        return hashlib.sha512(self.int_to_bytes(n)).digest()
