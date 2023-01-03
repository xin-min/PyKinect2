import os
import pandas as pd
import re


labels_dir = "../data/8Dec/labels/"
path = "250/rx/IA_DW1.txt"


f = open(labels_dir+path, "r")
lines = f.readlines()

doppler_file = pd.read_csv(lines[0][:-1])
quat_file = pd.read_csv(lines[1][:-1])
print(lines[1][:-1])
for x in range(2,len(lines)):
    line = lines[x].split(":")
    doppler_time = line[0]
    doppler = doppler_file.loc[:,doppler_time].values
    doppler = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in doppler]
    doppler = [[float(x[0]), float(x[1])] for x in doppler]
    # print(doppler)
    # print(len(doppler))
    # print(len(doppler[0]))

    indexes = line[1].strip("\n").strip('][ ').split(', ')
    for index in indexes:
        index = int(index)
        quat = quat_file.iloc[index].values[1:]
        quat = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in quat]
        quat = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in quat]
        print(quat)
        # print(quat_file.iloc[index].values) 
        break


    # print(doppler_time)
    # print(indexes)
    break




    # # print(df_matlab_file)
    # # print(df_joint_file['datetime'])
    # for i in range(len(df_joint_file['datetime'])):
    #     df_joint_file['datetime'][i]=df_joint_file['datetime'][i][11:].replace(':','').replace('.','')
    # joint_times = (df_joint_file['datetime'])
    # matlab_times = df_matlab_file.columns
    # # print(matlab_times)

    # joint_times = [int(x[:-3]) for x in joint_times]
    # matlab_times = [int(x) for x in matlab_times]
        

        
            




    #     img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    #     image = read_image(img_path)
    #     label = self.img_labels.iloc[idx, 1]
    #     # if self.transform: