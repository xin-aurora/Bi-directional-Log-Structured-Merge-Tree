from lsm.recordgenerator import SequentialNumericStringGenerator

# num_records = 200
num_records = 1000000

generator = SequentialNumericStringGenerator(0, 20)

file_name = "../data/random_data_" + str(num_records) + ".txt"

file_name_new = "../data/random_data_" + str(num_records) + "_with_anti_heavy.txt"

f = open(file_name, 'r')
line = f.readline()
data_set = []
while line != "":
    line = line.replace('\n', '')
    data_set.append(int(line))
    line = f.readline()
f.close()

# data_set = sorted(data_set)

anti_range_1 = [90000, 110000]
# anti_range_1 = [70, 110]
f_w = open(file_name_new, 'w')
# load tree
for i in range(num_records):
    key = data_set[i]
    data = key
    string = str(key) + "," + str(data) + "\n"
    f_w.write(string)
i = anti_range_1[0]
while i < anti_range_1[1]:
    key = data_set[i]
    data = key + 1
    string = str(key) + "," + str(data) + "\n"
    f_w.write(string)
    i += 1

# anti_range_2 = [50, 130]
anti_range_2 = [10000, 30000]
i = anti_range_2[0]
while i < anti_range_2[1]:
    key = data_set[i]
    data = key + 2
    string = str(key) + "," + str(data) + "\n"
    f_w.write(string)
    i += 1

anti_range_5 = [760000, 810000]
i = anti_range_5[0]
while i < anti_range_5[1]:
    key = data_set[i]
    data = key + 2
    string = str(key) + "," + str(data) + "\n"
    f_w.write(string)
    i += 1

anti_range_4 = [360000, 410000]
i = anti_range_4[0]
while i < anti_range_4[1]:
    key = data_set[i]
    data = key + 1
    string = str(key) + "," + str(data) + "\n"
    f_w.write(string)
    i += 1
#
# # anti_range_3 = [70, 110]
# anti_range_3 = [80000, 110000]
# i = anti_range_3[0]
# while i < anti_range_3[1]:
#     key = data_set[i]
#     data = key + 3
#     string = str(key) + "," + str(data) + "\n"
#     f_w.write(string)
#     i += 1
#
anti_range_4 = [360000, 410000]
i = anti_range_4[0]
while i < anti_range_4[1]:
    key = data_set[i]
    data = key + 4
    string = str(key) + "," + str(data) + "\n"
    f_w.write(string)
    i += 1

anti_range_5 = [760000, 810000]
i = anti_range_5[0]
while i < anti_range_5[1]:
    key = data_set[i]
    data = key + 4
    string = str(key) + "," + str(data) + "\n"
    f_w.write(string)
    i += 1

f_w.close()



