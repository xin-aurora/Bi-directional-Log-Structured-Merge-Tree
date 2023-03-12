import random

# num_records = 200
num_records = 1000000

# file_name = "../data/random_data_" + str(num_records) + ".txt"
file_name_new = "../data/random_data_" + str(num_records) + "_no_update.txt"

no_update_range = list(range(300000, num_records+1))
random.shuffle(no_update_range)
update_range = list(range(1, 300000))
random.shuffle(update_range)

no_update_range += update_range
#
# print(no_update_range)
#
# print(update_range)

f_w = open(file_name_new, 'w')

for key_value in no_update_range:
    string = str(key_value) + "," + str(key_value) + "\n"
    f_w.write(string)
f_w.close()
