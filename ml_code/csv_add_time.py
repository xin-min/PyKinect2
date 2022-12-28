import scipy.io
import pandas as pd
import os
import time
import numpy as np

comment to prevent accidental running of code

start_times = [
145845240, # ia rx walk1 #14.58.45.240 ->1458H 45.240s
150331393, # ia rx walk2
164901080, # ia rx br1
165148880, # ia rx br2
162309799, # ia rx dw
162531919,
164339334, # ia rx free
164609866,
163430946, # ia rx kick 1
163702922,
151732167, #pickup
152053223,
162850698, # punch
163138600,
150836601, # sit
151148435,
161510235, # sw
162027102,
160137399, # trevor pick up
160428587,
155048963, # trevor sit
155708139,
154123102, # trevor walk
154706297,
145846495,
150332105,
164901760,
165149755,
162310650,
162532643,
164340091,
164610791,
163431892,
163703765,
151733160,
152054208,
162849617,
163139526,
150837304,
151149172,
161511066,
162026410,
160138053,
160428176,
155104461,
155709047,
154122044,
154707153

]

# start at 0.3 seconds and add 0.05 every tick

files = []
output_dir = "../data/8Dec/matlab_data"

for file in os.listdir(output_dir):
	if file.endswith(".csv"):
		files.append(file)
		# print(os.path.join("/mydir", file))
count = 0
# print(files)
for file_csv in files:
	start_time = start_times[count]
	count +=1
	full_file_name = output_dir + "/" + file_csv
	csv_input = pd.read_csv(full_file_name)
	num_col = len(csv_input.columns)
	header = []
	for x in range(num_col):
		header.append(start_time+300+50*x)
	csv_input.columns = header
	csv_input.to_csv(full_file_name[:-4]+"_header.csv", index=False)
	# print("done")
	# time.sleep(100)