import math
files = ['../data/8Dec/labels/250/tx/IA_DW1.txt']
for file in files:
    f = open(file, "r")
    lines = f.readlines()

    # doppler_file = pd.read_csv(lines[0][:-1])
    # quat_file = pd.read_csv(lines[1][:-1])
    # print(lines[1][:-1])
    total = []
    for x in range(2,len(lines)):
        line = lines[x].split(":")
        indexes = line[1].strip("\n").strip('][ ').split(', ')
        if len(indexes)<7:
            continue

        new_indexes = []
        midpoint = math.floor(len(indexes)/2)
        # for x in range(midpoint-3, midpoint+3): # middle 6 values 
        #     new_indexes.append(int(indexes[x]))
        total.append([indexes[midpoint-1],indexes[midpoint]])
print(total)