import scipy.io
import pandas as pd
import os
import time
import numpy as np

# [16:-11]
files = []
output_dir = "../data/8Dec/matlab_data"

for file in os.listdir(output_dir):
	if file.endswith(".mat"):
		files.append(file)
		# print(os.path.join("/mydir", file))

for file_mat in files:
	# print(file_mat[17:-11])
	# time.sleep(3)
	# mat = scipy.io.loadmat(output_dir+"/"+file_mat)

	# file_mat = file_mat[17:-11]

	# file_csv = "/08-12-22_14-58-38"+".csv" #.csv # in joint folder

	# output_dir = "output/8Dec/joint_tx_quat/"
	# if not (os.path.exists(output_dir)):
	# 	os.makedirs(output_dir) # Create a new directory because it does not exist
	# output_name = output_dir+"/"+file_mat[17:-11]+'.csv'

	mat = scipy.io.loadmat(output_dir+"/"+file_mat)
	data = pd.DataFrame(mat['S'])
	# data.style.hide_index
	# mat = {k:v for k, v in mat.items() if k[0] != '_'}
	# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()}) 
	data.to_csv(output_name, index=False, header = False)

	# for i in mat:
	# 	print(i)
		# if '__' not in i and 'readme' not in i:
		# 	np.savetxt((output_name),mat[i],delimiter=',')