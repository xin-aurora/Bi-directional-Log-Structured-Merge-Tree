# Generate uniform random queries

import random
import bisect
import math
from functools import reduce

PREFER_NEWER = True  # True for hot in newer records, False for hot in older records

# Zipf distribution
class OlderGenerator:
    def __init__(self, n, alpha):
        # Calculate Zeta values from 1 to n:
        tmp = [1. / (math.pow(float(i), alpha)) for i in range(1, n+1)]
        zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0])
        # Store the translation map:
        self.distMap = [x / zeta[-1] for x in zeta]

    def prob(self, v):
        return self.distMap[v] - self.distMap[v - 1]

    def next(self):
        # Take a uniform 0-1 pseudo-random value:
        u = random.random()
        # Translate the Zipf variable:
        return bisect.bisect(self.distMap, u) - 1


# Inverted zipf distribution
class NewerGenerator(OlderGenerator):
    def __init__(self, n, alpha):
        super().__init__(n, alpha)
        self.n = n

    def _prob(self, v):
        # return super(OlderGenerator, self).prob(self.n - v + 1)
        self.prob(self.n - v + 1)

    def _next(self):
        # return self.n - super(OlderGenerator, self).next()
        return self.n - self.next()


def build_hot_query(keys, num_keys, ZIPF_ALPHA, num_qs, min_len, max_len):
    queries = []
    dist = NewerGenerator(num_keys, ZIPF_ALPHA) if PREFER_NEWER else OlderGenerator(num_keys, ZIPF_ALPHA)
    for i in range(num_qs):
        while True:
            min_idx = dist.next() - 1
            min_q = keys[min_idx]
            max_idx = dist.next() - 1
            max_q = keys[max_idx]
            query_length = abs(max_q - min_q) + 1
            if min_len <= query_length <= max_len:
                if max_q < min_q:
                    tem = max_q
                    max_q = min_q
                    min_q = tem
                query = str(query_length) + "-" + str(min_q) + "-" + str(max_q)
                queries.append(query)
                break
    return queries

