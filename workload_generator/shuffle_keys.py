# Generate uniform random unique keys

import os
import random
import matplotlib.pyplot as plt

MIN_KEY = 1  # Smallest key
NUM_KEYS = 1000000  # Total number of keys
# NUM_KEYS = 2000000  # Total number of keys
SHUFFLE = False  # True to generate random keys, False to generate sequential keys
KEYS_FILENAME = "../data/sequential_data_" + str(NUM_KEYS) + ".txt"

max_key = MIN_KEY + NUM_KEYS - 1

root = os.path.dirname(os.path.realpath(__file__))

sequential_keys = list(range(MIN_KEY, max_key + 1))
RANDOM_KEYS = sequential_keys.copy()
if SHUFFLE:
    random.shuffle(RANDOM_KEYS)

# Save keys to file
with open(os.path.join(root, KEYS_FILENAME), "w") as outf:
    for key in RANDOM_KEYS:
        outf.write("{0}\n".format(key))
outf.close()

# fig, ax = plt.subplots()
# l1, = ax.plot(sequential_keys, RANDOM_KEYS, "-")
# l2, = ax.plot(sequential_keys, sequential_keys, "--")
# ax.set_xlim(MIN_KEY, max_key)
# ax.set_ylim(MIN_KEY, max_key)
# ax.set_xlabel("Number of keys")
# ax.set_ylabel("Key value")
# ax.legend([l1, l2], ["Random", "Sequential"])
# plt.show()
