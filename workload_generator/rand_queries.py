# Generate uniform random queries

import os
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

KEYS_FILENAME = "../data/random_data_100.txt"
RAND_Q_FILENAME = "../data/rand_qs.txt"

# Query length
MIN_LEN = 10  # Minimum query length
MAX_LEN = 15  # Maximum query length
NUM_QS = 50  # Number of queries to generate

keys = []
root = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(root, KEYS_FILENAME), "r") as inf:
    for line in inf:
        if len(line) > 1:
            keys.append(int(line[:-1]))
inf.close()
num_keys = len(keys)
min_key = min(keys)
max_key = max(keys)
print("# keys: {0}, min={1}, max={2}".format(num_keys, min_key, max_key))


queries = []
for i in range(NUM_QS):
    q_len = random.randint(MIN_LEN, MAX_LEN)
    min_idx = random.randint(0, num_keys - q_len - 1)
    min_q = keys[min_idx]
    max_q = keys[min_idx+q_len]
    if min_q > max_q:
        tem = min_q
        min_q = max_q
        max_q = tem
    queries.append((min_q, max_q, q_len))


# Save queries to file
with open(os.path.join(root, RAND_Q_FILENAME), "w") as outf:
    for min, max, l in queries:
        query = str(l) + "-" + str(min) + "-" + str(max)
        # outf.write("{0}\t{1}\n".format(k, l))
        outf.write(query + "\n")
outf.close()

# bar_w = 100.0 / NUM_QS
# rect_w = bar_w * 0.8
# margin = bar_w * 0.1


# def create_rect(idx):
#     min_q, q_len = queries[idx]
#     x = bar_w * idx + margin
#     y = 100.0 * (min_q - min_key) / num_keys
#     h = 100.0 * q_len / num_keys
#     return Rectangle((x, y), rect_w, h,
#                      fill=True,
#                      edgecolor="black")
#
#
# fig, ax = plt.subplots()
# for i in range(NUM_QS):
#     ax.add_patch(create_rect(i))
# ax.set_xlim(0, 100)
# ax.set_xticks([1, ] + [x * 10 for x in range(1, 11)])
# ax.set_ylim(0, 100)
# ax.set_yticks([10 * y for y in range(0, 11)])
# ax.set_yticklabels(["1", ] + [str(1000 * y) for y in range(1, 11)])
# ax.set_xlabel("Number of queries")
# ax.set_ylabel("Query range")
# plt.show()
