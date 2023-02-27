import time
import numpy as np
# import cv2
# import os
from scipy.spatial.transform import Rotation
# from scipy.stats import linregress
import pandas as pd
import re
# import torch
import math
# from dataloader_MDPose import velocityDataset

# x, y = velocityDataset()

# action_files=[
# 	'../data/8Dec/labels/250/tx/IAwalk1.txt', 
# 	'../data/8Dec/labels/250/tx/IAwalk2.txt', 
# 	'../data/8Dec/labels/250/tx/IA_DW1.txt', 
# 	'../data/8Dec/labels/250/tx/IA_DW2.txt', 
# 	'../data/8Dec/labels/250/tx/IA_free2.txt', 
# 	'../data/8Dec/labels/250/tx/IA_free3.txt', 
# 	'../data/8Dec/labels/250/tx/IA_Kick1.txt', 
# 	'../data/8Dec/labels/250/tx/IA_Kick2.txt', 
# 	'../data/8Dec/labels/250/tx/IA_pickup1.txt', 
# 	'../data/8Dec/labels/250/tx/IA_pickup2.txt', 
# 	'../data/8Dec/labels/250/tx/IA_Punch1.txt', 
# 	'../data/8Dec/labels/250/tx/IA_Punch2.txt', 
# 	'../data/8Dec/labels/250/tx/IA_sit1.txt', 
# 	'../data/8Dec/labels/250/tx/IA_sit2.txt', 
# 	'../data/8Dec/labels/250/tx/IA_SW1.txt', 
# 	'../data/8Dec/labels/250/tx/IA_SW2.txt', 
# 	'../data/8Dec/labels/250/tx/trevor_pickup1.txt', 
# 	'../data/8Dec/labels/250/tx/trevor_pickup2.txt', 
# 	'../data/8Dec/labels/250/tx/trevor_sit1.txt', 
# 	'../data/8Dec/labels/250/tx/trevor_sit2.txt'
# 	'../data/8Dec/labels/250/tx/trevor_walk1.txt', 
# 	'../data/8Dec/labels/250/tx/trevor_walk2.txt'
# ]
# if actions == "all":
# 	files = action_files
# else:
# 	files = []
# 	for action in actions:
# 		index = action_class.index(action)
# 		files.append(action_files[2*index])
# 		files.append(action_files[2*index+1])
# 		# print(action_files[2*index])
# 		# print(action_files[2*index+1])

file = '../data/8Dec/labels/smallest/rx/IA_free3.txt'


x_values = []
y_values = []


f = open(file, "r")
print(file)
lines = f.readlines()

doppler_file = pd.read_csv(lines[0][:-1])
coord_file = pd.read_csv(lines[1][:-1])
# print(lines[1][:-1])
f = open('velocity_file_test.txt', 'w+')
for x in range(2,len(lines)):
	line = lines[x].split(":")
	doppler_time = line[0] # 1 doppler every 0.05s = 50 in doppler time. doppler time: HHMMSS---
	
	temp_doppler = doppler_file.loc[:,doppler_time].values
	temp_doppler = np.transpose(temp_doppler)

	temp_doppler =[(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in temp_doppler]
	doppler= []
	try:
		for x in temp_doppler:
			doppler.append(float(x[0]))
			doppler.append(float(x[1]))
	except:
		continue
	# print(doppler)

	# doppler = [float(x[0]), float(x[1]) for x in doppler]


	# for temp in temp_doppler:
	# 	temp = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in temp] # extracts all numbers, removes 'j' for the complex part of the number
	# 	try:
	# 		temp = [[float(x[0]), float(x[1])] for x in temp] # converts from complex to 2d float (x + yj) to (x, y)
	# 	except:
	# 		# print(temp)
	# 		continue 
	# 	doppler.append(torch.flatten(torch.FloatTensor(temp)))
	# print(line[1].strip("\n").strip('][ ').split(', '))
	indexes = int((line[1].strip("\n").strip('][ ').split(', '))[0])-1 ###-1 for velocity, index is for quat
	# if len(indexes)<5:
	# 	continue

	# new_indexes = []
	# midpoint = math.floor(len(indexes)/2)
	# for x in range(midpoint-2, midpoint+3): # middle 5 values 
	# 	new_indexes.append(int(indexes[x])-1)
	new_indexes = [indexes-4, indexes-2, indexes, indexes+2, indexes+5]

	time_list = []
	coor_list = []


	for index in new_indexes:
		coor = coord_file.iloc[index].values
		timestamp = float(coor[0][-7:])
		coor = coor[1:]
		coor = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in coor]
		coor = [[float(x[0]), float(x[1]), float(x[2])] for x in coor]
		# coor = [[float(x[0])+1, float(x[1])+1, float(x[2])+1] for x in coor]

		coor_list.append(coor)
		time_list.append(timestamp)

	#### find gradient of time vs coor (velocity)
	# np.polynomial.polynomial.Polynomial.fit(x, y, deg, domain=None, rcond=None, full=False, w=None, window=None, symbol='x')
	# np.polynomial.polynomial.Polynomial.fit(x, y, deg, domain=None, rcond=None, full=False, w=None, window=None, symbol='x')

	## coor_list: 5x25x3 (5timesteps*25joints*3axis)
	## desired slope: 25x3 (25joints*3axis: velocity of each axis for each joint)
	velocity_list = np.zeros((25,3))
	# print(len(coor_list))
	# print(len(coor_list[0]))
	# print(len(coor_list[0][0]))

	for x in range(25):
		for y in range(3):

			temp_coor = []
			for z in range(5):
				temp_coor.append(coor_list[z][x][y])
			slope = np.polyfit(time_list,temp_coor,1)[0]
			velocity_list[x,y] = slope
	f.write(str(doppler_time) + ': ' + str(velocity_list)+'\n')