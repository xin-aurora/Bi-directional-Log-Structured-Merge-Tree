import os
import random
import matplotlib.pyplot as plt

MIN_KEY = 1  # Smallest key
# NUM_KEYS = 1000  # Total number of keys
NUM_KEYS = 2000000  # Total number of keys
SHUFFLE = True  # True to generate random keys, False to generate sequential keys
KEYS_FILENAME = "../data/near_sequential_data_" + str(NUM_KEYS) + ".txt"
shuffle_times = 20
shuffle_length = 3000

max_key = MIN_KEY + NUM_KEYS - 1

root = os.path.dirname(os.path.realpath(__file__))

sequential_keys = list(range(MIN_KEY, max_key + 1))

cnt = 0
# shuffle_ranges = [(8000, 11000), (12000, 15000), ()]
while cnt < shuffle_times:
    k = random.randint(10000, 1500000)
    list_0 = sequential_keys[:k]
    list_1 = sequential_keys[k:(k+shuffle_length)]
    list_2 = sequential_keys[(k+shuffle_length):]
    print(k)
    random.shuffle(list_1)
    sequential_keys = list_0 + list_1 + list_2
    cnt += 1

# Save keys to file
with open(os.path.join(root, KEYS_FILENAME), "w") as outf:
    for key in sequential_keys:
        outf.write("{0}\n".format(key))
outf.close()