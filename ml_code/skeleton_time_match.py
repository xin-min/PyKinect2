import scipy.io
import pandas as pd
import os
import time
import numpy as np
from collections import defaultdict

################# to configure ###################
buffer_window = 250 #0.25s
receiver = "rx/" # "tx/" or "rx/"
##############################################################

joint_root_dir = "../data/8Dec/joint_"+ receiver
matlab_root_dir = "../data/8Dec/matlab_data/"
output_dir = "../data/8Dec/labels/"

# files = []
# for file in os.listdir(joint_root_dir):
# 	if file.endswith(".csv"):
# 		files.append(file)
# print(files)

if receiver == "tx/":
	joint_matlab = [
	('08-12-22_14-58-38.csv', 'IAwalk1'),
	('08-12-22_15-03-26.csv', 'IAwalk2'),
	('08-12-22_15-08-19.csv', 'IA_sit1'),
	('08-12-22_15-11-37.csv', 'IA_sit2'),
	# '08-12-22_15-15-10.csv', 
	('08-12-22_15-17-24.csv', 'IA_pickup1'), #problematic
	('08-12-22_15-20-48.csv', 'IA_pickup2'), #problematic
	('08-12-22_15-40-44.csv', 'trevor_walk1'), 
	('08-12-22_15-46-58.csv', 'trevor_walk2'), 
	('08-12-22_15-50-22.csv', 'trevor_sit1'), 
	('08-12-22_15-57-02.csv', 'trevor_sit2'), 
	('08-12-22_16-01-32.csv', 'trevor_pickup1'), #problematic
	('08-12-22_16-04-10.csv', 'trevor_pickup2'), #problematic
	('08-12-22_16-15-00.csv', 'IA_SW1'),
	('08-12-22_16-20-19.csv', 'IA_SW2'),
	('08-12-22_16-22-58.csv', 'IA_DW1'),
	('08-12-22_16-25-26.csv', 'IA_DW2'),
	('08-12-22_16-28-44.csv', 'IA_Punch1'), 
	('08-12-22_16-31-21.csv', 'IA_Punch2'), 
	('08-12-22_16-34-23.csv', 'IA_Kick1'), 
	('08-12-22_16-36-55.csv', 'IA_Kick2'), 
	# '08-12-22_16-40-03.csv', 
	('08-12-22_16-43-35.csv', 'IA_free2'), 
	('08-12-22_16-46-04.csv', 'IA_free3') 
	# '08-12-22_16-48-50.csv', 
	# '08-12-22_16-51-40.csv'
	]

else: ### for "rx/"
	joint_matlab = [
	('08-12-22_14-58-38.csv', 'IAwalk1'),
	('08-12-22_15-03-26.csv', 'IAwalk2'), 
	('08-12-22_15-08-19.csv', 'IA_sit1'), 
	('08-12-22_15-11-37.csv', 'IA_sit2'), 
	# ('08-12-22_15-15-10.csv', ), 
	('08-12-22_15-17-26.csv', 'IA_pickup1'), #problematic
	('08-12-22_15-20-48.csv', 'IA_pickup2'), #problematic
	('08-12-22_15-40-55.csv', 'trevor_walk1'), 
	('08-12-22_15-46-58.csv', 'trevor_walk2'), 
	('08-12-22_15-50-22.csv', 'trevor_sit1'), 
	('08-12-22_15-57-02.csv', 'trevor_sit2'), 
	('08-12-22_16-01-32.csv', 'trevor_pickup1'), #problematic
	('08-12-22_16-04-10.csv', 'trevor_pickup2'), #problematic
	('08-12-22_16-15-00.csv', 'IA_SW1'), 
	('08-12-22_16-20-21.csv', 'IA_SW2'), 
	('08-12-22_16-22-58.csv', 'IA_DW1'), 
	('08-12-22_16-25-26.csv', 'IA_DW2'), 
	('08-12-22_16-28-43.csv', 'IA_Punch1'), 
	('08-12-22_16-31-21.csv', 'IA_Punch2'), 
	('08-12-22_16-34-23.csv', 'IA_Kick1'), 
	('08-12-22_16-36-55.csv', 'IA_Kick2'), 
	# ('08-12-22_16-40-03.csv', ), 
	('08-12-22_16-43-34.csv', 'IA_free2'), 
	('08-12-22_16-46-04.csv', 'IA_free3'), 
	# ('08-12-22_16-48-50.csv', ), 
	# ('08-12-22_16-51-40.csv', )
	]

for k in range(len(joint_matlab)):
	textfile = output_dir + str(buffer_window) + "/" + receiver + joint_matlab[k][1] +'.txt'
	f = open(textfile, 'x')

	joint_file = joint_root_dir + joint_matlab[k][0]
	if receiver =="tx/":
		matlab_file = matlab_root_dir + "08Dec_TX_" + joint_matlab[k][1] + "_ch1_header.csv"
	else:
		matlab_file = matlab_root_dir + "08Dec_RX_" + joint_matlab[k][1] + "_ch1_header.csv"

	# print(matlab_file)
	f.write(matlab_file + '\n')
	f.write(joint_file + '\n')


	df_joint_file = pd.read_csv(joint_file) # 30 fps -> every 0.033s 
	df_matlab_file = pd.read_csv(matlab_file) #every 0.5s

	# print(df_matlab_file)
	# print(df_joint_file['datetime'])
	for i in range(len(df_joint_file['datetime'])):
		df_joint_file['datetime'][i]=df_joint_file['datetime'][i][11:].replace(':','').replace('.','')
	joint_times = (df_joint_file['datetime'])
	matlab_times = df_matlab_file.columns
	# print(matlab_times)

	joint_times = [int(x[:-3]) for x in joint_times]
	matlab_times = [int(x) for x in matlab_times]

	# print("joint_times")
	# print(joint_times)
	# print("matlab_times")
	# print(matlab_times)


	# print(len(joint_times))
	matching_index = defaultdict(list)
	lowest = 0

	# for j in range(lowest, len(joint_times)):
	# 	print(j)

	for i in range(len(matlab_times)):
		# print(matlab_times[i])
		for j in range(lowest, len(joint_times)):
			if j%2==0:
				continue
			# if matlab_times[i]>145854000:
			# 	print(joint_times[j])
			if joint_times[j] < (matlab_times[i]-buffer_window):
				# lowest = joint_times[j]
				continue
			elif joint_times[j] > (matlab_times[i]+buffer_window):
				continue

			else:
				matching_index[matlab_times[i]].append(j)
		# time.sleep(2)

	# print((len(matlab_times),len(matching_index)))
	# print(matching_index)
	for k,v in matching_index.items():
		f.write(str(k) + ": " + str(v)+'\n')
	# f.write(str(matching_index))
	# break

	# print(len(matching_index))
