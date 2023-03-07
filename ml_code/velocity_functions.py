## read 
## use conda env pytorchpy3.9

# import csv
import time
import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
# from scipy.stats import linregress
import pandas as pd
import re
# import torch
import math
# from kinect_quat_functions import vector2screen
# import cv2
# import matplotlib.image as img
from statistics import mean

joints = [
'Head', #0
'Neck', #1
'SpineShoulder', #2
'SpineMid', #3
'SpineBase', #4
'ShoulderRight', #5
'ShoulderLeft', #6
'HipRight', #7
'HipLeft', #8
'ElbowRight', #9
'WristRight', #10
'HandRight', #11
'HandTipRight', #12
'ThumbRight', #13
'ElbowLeft', #14
'WristLeft', #15
'HandLeft', #16
'HandTipLeft', #17
'ThumbLeft', #18
'KneeRight', #19
'AnkleRight', #20
'FootRight', #21
'KneeLeft', #22
'AnkleLeft', #23
'FootLeft', #24
]
# 1-2-3-2-4-5-6-5-4-2-7-8-9-8-7-2-1-10-11-12-13-12-11-10-14-15-16-15-14-10-1
# (refer to picture in paper)

joints_tree_order = [ # kinect numbering - name - paper numbering

3,  # 'SpineMid', #1
1,  # 'Neck', #2
0,  # 'Head', #3
1,  # 'Neck', #2
6,  # 'ShoulderLeft', #4
14,  # 'ElbowLeft', #5
15,  # 'WristLeft', #6
14,  # 'ElbowLeft', #5
6,  # 'ShoulderLeft', #4
1,  # 'Neck', #2
5,  # 'ShoulderRight', #7
9,  # 'ElbowRight', #8
10,  # 'WristRight', #9
9, # 'ElbowRight', #8
5, # 'ShoulderRight', #7
1,  # 'Neck', #2
3,  # 'SpineMid', #1
4,  # 'SpineBase', #10
8,  # 'HipLeft', #11
22,  # 'KneeLeft', #12
23,  # 'AnkleLeft', #13
22,  # 'KneeLeft', #12
8,  # 'HipLeft', #11
4,  # 'SpineBase', #10
7,  # 'HipRight', #14
19,  # 'KneeRight', #15
20,  # 'AnkleRight', #16
19,  # 'KneeRight', #15
8,  # 'HipRight', #14
4,  # 'SpineBase', #10
3,  # 'SpineMid', #1
]

# # joints_tree = [ # numbering only
# 'SpineMid', #1
# 'Neck', #2
# 'Head', #3
# 'ShoulderLeft', #4
# 'ElbowLeft', #5
# 'WristLeft', #6
# 'ShoulderRight', #7
# 'ElbowRight', #8
# 'WristRight', #9
# 'SpineBase', #10
# 'HipLeft', #11
# 'KneeLeft', #12
# 'AnkleLeft', #13
# 'HipRight', #14
# 'KneeRight', #15
# 'AnkleRight', #16

# ##### unused
# # 'SpineShoulder', #
# # 'HandRight', #
# # 'HandTipRight', #
# # 'ThumbRight', #
# # 'HandLeft', #
# # 'HandTipLeft', #
# # 'ThumbLeft', #
# # 'FootRight', #
# # 'FootLeft', #
# # ]

def loadDataXYZ(tx = True, actions = "all"):

	action_class = ["walk", "DW", "free", "kick", "pickup", "punch", "sit", "SW", "all"]
	
	action_files=[
		# '../data/8Dec/labels/250/tx/IAwalk1.txt', 
		# '../data/8Dec/labels/250/tx/IAwalk2.txt', 
		# '../data/8Dec/labels/250/tx/IA_DW1.txt', 
		# '../data/8Dec/labels/250/tx/IA_DW2.txt', 
		# '../data/8Dec/labels/250/tx/IA_free2.txt', 
		# '../data/8Dec/labels/250/tx/IA_free3.txt', 
		# '../data/8Dec/labels/250/tx/IA_Kick1.txt', 
		# '../data/8Dec/labels/250/tx/IA_Kick2.txt', 
		# '../data/8Dec/labels/250/tx/IA_pickup1.txt', 
		# '../data/8Dec/labels/250/tx/IA_pickup2.txt', 
		# '../data/8Dec/labels/250/tx/IA_Punch1.txt', 
		# '../data/8Dec/labels/250/tx/IA_Punch2.txt', 
		# '../data/8Dec/labels/250/tx/IA_sit1.txt', 
		# '../data/8Dec/labels/250/tx/IA_sit2.txt', 
		# '../data/8Dec/labels/250/tx/IA_SW1.txt', 
		# '../data/8Dec/labels/250/tx/IA_SW2.txt', 
		# '../data/8Dec/labels/250/tx/trevor_pickup1.txt', 
		# '../data/8Dec/labels/250/tx/trevor_pickup2.txt', 
		# '../data/8Dec/labels/250/tx/trevor_sit1.txt', 
		# '../data/8Dec/labels/250/tx/trevor_sit2.txt'
		# '../data/8Dec/labels/250/tx/trevor_walk1.txt', 
		# '../data/8Dec/labels/250/tx/trevor_walk2.txt',

		'../data/8Dec/labels/smallest/rx/IAwalk1.txt', 
		'../data/8Dec/labels/smallest/rx/IAwalk2.txt', 
		'../data/8Dec/labels/smallest/rx/IA_DW1.txt', 
		'../data/8Dec/labels/smallest/rx/IA_DW2.txt', 
		'../data/8Dec/labels/smallest/rx/IA_free2.txt', 
		'../data/8Dec/labels/smallest/rx/IA_free3.txt', 
		'../data/8Dec/labels/smallest/rx/IA_Kick1.txt', 
		'../data/8Dec/labels/smallest/rx/IA_Kick2.txt', 
		# '../data/8Dec/labels/smallest/rx/IA_pickup1.txt', 
		# '../data/8Dec/labels/smallest/rx/IA_pickup2.txt', 
		'../data/8Dec/labels/smallest/rx/IA_Punch1.txt', 
		'../data/8Dec/labels/smallest/rx/IA_Punch2.txt', 
		# '../data/8Dec/labels/smallest/rx/IA_sit1.txt', 
		# '../data/8Dec/labels/smallest/rx/IA_sit2.txt', 
		'../data/8Dec/labels/smallest/rx/IA_SW1.txt', 
		'../data/8Dec/labels/smallest/rx/IA_SW2.txt', 
		# '../data/8Dec/labels/smallest/rx/trevor_pickup1.txt', 
		# '../data/8Dec/labels/smallest/rx/trevor_pickup2.txt', 
		# '../data/8Dec/labels/smallest/rx/trevor_sit1.txt', 
		# '../data/8Dec/labels/smallest/rx/trevor_sit2.txt',
		# '../data/8Dec/labels/smallest/rx/trevor_walk1.txt', 
		# '../data/8Dec/labels/smallest/rx/trevor_walk2.txt'
	]


	if actions == "all":
		files = action_files
	else:
		files = [] 
		for action in actions:
			index = action_class.index(action)
			files.append(action_files[2*index])
			files.append(action_files[2*index+1])
			# print(action_files[2*index])
			# print(action_files[2*index+1])

	# files = [
	# # '../data/8Dec/labels/smallest/rx/IAwalk1.txt']#, 
	# # '../data/8Dec/labels/smallest/rx/IAwalk2.txt'
	# '../data/8Dec/labels/smallest/rx/trevor_walk1.txt', 
	# # '../data/8Dec/labels/smallest/rx/trevor_walk2.txt'
	# 	]
	files = action_files
	files = [
	# '../data/8Dec/labels/smallest/rx/IA_kick2.txt', 

	# '../data/8Dec/labels/smallest/rx/IA_Punch2.txt',
	# '../data/8Dec/labels/smallest/rx/IA_Kick1.txt',

	'../data/8Dec/labels/smallest/rx/IAwalk2.txt',
	# 	'../data/8Dec/labels/smallest/rx/trevor_walk1.txt', 

	]

	f1 = open('confuseddoppler.txt', 'w+')
	x_values = []
	y_values = []
	y_coor = []
	doppler_img_list = []
	velocity_list_list = []
	# print(files)
	# error
	for file in files:
		f = open(file, "r")
		print(file)
		lines = f.readlines()

		doppler_file = pd.read_csv(lines[0][:-1])
		coord_file = pd.read_csv(lines[1][:-1])
		# print(lines[1][:-1])
		prev_idx = -1
		for x in range(2,len(lines)):
			line = lines[x].split(":")
			doppler_time = line[0] # 1 doppler every 0.05s = 50 in doppler time. doppler time: HHMMSS---

			# temp_dopplers = doppler_file.loc[:,doppler_time].values #each doppler spaced 50ms apart, so total 5 dopplers
			# print(len(temp_dopplers))
			# print(len(temp_dopplers[0]))



			temp_dopplers = doppler_file.loc[:,str(int(doppler_time)-200):doppler_time].values #each doppler spaced 50ms apart, so total 5 dopplers
			# print(len(temp_dopplers))
			# print(len(temp_dopplers[0]))
			# print(len(temp_dopplers[0][0]))
			# erorr
			dopplers = []
			
			temp_dopplers = np.transpose(temp_dopplers)
			# temp_dopplers = np.transpose(temp_dopplers, (1,2))

			# print(len(temp_dopplers))
			# print(len(temp_dopplers[0]))
			# error

			for temp_doppler in temp_dopplers:
				# print(len(temp_doppler))
				# print(len(temp_doppler[0]))
				# temp_doppler = np.transpose(temp_doppler)
				# print(len(temp_doppler))
				# print(len(temp_doppler[0]))
				# error


				temp_doppler =[(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in temp_doppler]
				doppler= []
				# doppler_img = []
				try:
					for x in temp_doppler:
						# print(x)
						# doppler.append(float(x[0]))
						# doppler.append(float(x[1]))

						doppler.append(math.sqrt(float(x[0])*float(x[0]) + float(x[1])*float(x[1])))
				except:
					# print("skipped")
					# time.sleep(1)
					continue

				mean_value = mean(doppler[:50]+doppler[-50:])
				doppler = [10*math.log10(x/mean_value) for x in doppler]
				doppler = doppler[1100:1300] ### middle 200
				for i in range(len(doppler)):
					if doppler[i]<10:
						doppler[i]=0
				dopplers.append(doppler)
			dopplers = np.array(dopplers, dtype="f")
			# print(dopplers)
			# # error
			# print(len(dopplers))
			# print(len(dopplers[0]))


			indexes = int(line[1].strip("\n").strip('][ ').split(', ')[0])
			if indexes ==prev_idx:
				continue
			else:
				prev_idx = indexes
			new_indexes = [indexes-5, indexes-3, indexes-1, indexes+1, indexes+3]

			time_list = []
			coor_list = []
			current_coor = coord_file.iloc[new_indexes[2]].values
			current_coor = current_coor[1:]
			current_coor = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in current_coor]
			current_coor = [[float(x[0]), float(x[1]), float(x[2])] for x in current_coor]


			try:


				for index in new_indexes:
					coor = coord_file.iloc[index].values
					timestamp = float(coor[0][-7:])
					coor = coor[1:]
					coor = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in coor]
					coor = [[float(x[0]), float(x[1]), float(x[2])] for x in coor]
					# coor = [[float(x[0])+1, float(x[1])+1, float(x[2])+1] for x in coor]

					coor_list.append(coor)
					time_list.append(timestamp)
			except:
				continue

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
					velocity_list[x,y] = slope#*10+1
			velocity_flattened = np.zeros((75))



			#################################### rearrange to fit tree structure (31 nodes in total = 31*3 = 93values yay)

			joints_tree_velocity = []
			for order in joints_tree_order:
				joints_tree_velocity.append(velocity_list[order])

			velocity_list = joints_tree_velocity
			velocity_flattened = np.zeros((93))


			############################ END TREE STRUCTURE ################################






			# print(velocity_list)
			# doppler = np.ones((1,4800))
			# x_values.append(doppler) # multiple quat values for the same doppler

			# fig, (ax1,ax2) = plt.subplots(1,2)

			# plot = ax1.pcolor(doppler_img, cmap='plasma')
			# plot1 = ax2.pcolor(velocity_list, cmap='hot')

			# fig.colorbar(plot, ax = [ax1], location ='left');
			# fig.colorbar(plot1);
			# # ax2.colorbar()
			# # plt.tight_layout()
			# # plt.show()
			# plt.show(block=False)
			# # plt.pause(1)
			# # plt.close()
			# # error

			# velocity_flattened = np.zeros((93))
			counter = 0
			for z in velocity_list:
				for y in z:
					if float(y)>10:
						# print("more")
						try:
							velocity_flattened[counter] = velocity_flattened[counter-1]
						except:
							velocity_flattened[counter] = 0
					else:
						velocity_flattened[counter]=float(y)
					counter+=1
					# for x in y:
					# 	velocity_flattened.append(x)
			if len(dopplers)<5:
				continue
			# print(len(dopplers))
			# print(len(dopplers[0]))
			# print(len(velocity_flattened))
			# print(len(velocity_flattened[0]))


			x_values.append(dopplers)
			y_values.append(velocity_flattened)
			y_coor.append(current_coor)

		# 	############### test velocity and doppler matching
		# 	temp_vel = []

		# 	for x in range(25):
		# 		temp = math.sqrt(velocity_list[x][0]*velocity_list[x][0]+velocity_list[x][1]*velocity_list[x][1]+velocity_list[x][2]*velocity_list[x][2])
		# 		if temp>10:
		# 			try:
		# 				temp_vel.append(temp_vel[-1])
		# 			except:
		# 				temp_vel.append(0)
		# 				print("cannot")

		# 		else: 
		# 			temp_vel.append(math.sqrt(velocity_list[x][0]*velocity_list[x][0]+velocity_list[x][1]*velocity_list[x][1]+velocity_list[x][2]*velocity_list[x][2]))
		# 	# mean_value = mean(doppler_img[:50]+doppler_img[-50:])
		# 	# print(mean_value)
		# 	# doppler_img = [10*math.log10(x/mean_value) for x in doppler_img]
		# 	# doppler_img = [10*math.log10(x/mean_value) for x in doppler_img]
		# 	# for i in range(len(doppler_img)):
		# 	# 	if doppler_img[i]<10:
		# 	# 		doppler_img[i]=0
		# 	# doppler_img = doppler_img[1100:1300] # full range from -2000 to 2000, but we want -100 to 100


		# 	# print(doppler_img)
		# 	# error

		# 	doppler_img_list.append(doppler)

		# 	# print(doppler_img)
		# 	velocity_list_list.append(temp_vel)

		# 	# x_values.append(torch.flatten(torch.Tensor(doppler))) # multiple quat values for the same doppler
		# 	# y_values.append(torch.flatten(torch.Tensor(velocity_list)))
		# 	# y_values.append(torch.nan_to_num(torch.Tensor((1,1,1))))
		# 	# print(torch.flatten(torch.Tensor(velocity_list))[:3])
		# f1.write(str(doppler_img_list)+"\n")
		# # try:
		# fig, (ax1,ax2) = plt.subplots(1,2)

		# plot = ax1.pcolor(doppler_img_list, cmap='plasma')
		# plot1 = ax2.pcolor(velocity_list_list, cmap='hot')

		# fig.colorbar(plot, ax = [ax1], location ='left');
		# fig.colorbar(plot1);
		# # ax2.colorbar()
		# # plt.tight_layout()
		# # plt.show()
		# plt.savefig(file[32:-4]+'.png')
		# # except:
		# 	# print("error")
		# # plt.show(block=False)
		# # plt.pause(1)
		# # plt.close()
		# # error
		# 	############### end test velocity and doppler matching


	return (x_values,y_values, y_coor)

# loadDataXYZ(tx = True, actions = "all")

# x, y = loadDataXYZ(actions = ["SW"])
# # f = open("velocity_list.txt", "w+")
# total_vel = []
# for i in y: #25*3
# 	body_vel = []

# 	for j in i: # 3 XZY coor
# 		single_vel = []
# 		for k in j:
# 			single_vel.append(float(k))
# 		body_vel.append(single_vel)
# 		# single_vel = []
# 	total_vel.append(body_vel)
# # print(len(total_vel))
# # print(len(total_vel[0]))
# # print(len(total_vel[0][0]))


# 	# print(i)
# 	# i = list(i)
# 	# single_vel = []
# 	# body_vel = []
# 	# for j in range(75):
# 	# 	if j%3==0 and j>0:
# 	# 		body_vel.append(single_vel)
# 	# 		single_vel =[]
# 	# 	single_vel.append(float(i[j]))
# 	# body_vel.append(single_vel)
# 	# total_vel.append(body_vel)

# 	# for k in range(25):
# 	# 	f.write(joints[k] + ': '+str(body_vel[k])+'\n')

# file = '../data/8Dec/labels/250/tx/IA_SW1.txt'
# f = open(file, "r")
# lines = f.readlines()
# coord_file = pd.read_csv(lines[1][:-1])
# line = lines[2].split(":")
# indexes = line[1].strip("\n").strip('][ ').split(', ')
# # if len(indexes)<5:
# # 	continue

# new_indexes = []
# midpoint = math.floor(len(indexes)/2)
# initial_pose = coord_file.iloc[midpoint-3].values
# initial_pose = initial_pose[1:]
# initial_pose = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in initial_pose]
# initial_pose = [[float(x[0]), float(x[1])]for x in initial_pose]
# # print(initial_pose)
# # error

# time_diff = 1/20 #20fps
# # initial_pose = [] #25joints*3XYZcoor
# # total_vel = [] #n timesteps*25joints*3XYZcoor
# new_pose = [] #25joints*3XYZcoor
# total_pose = []
# total_pose.append(initial_pose)
# for n in range(len(total_vel)):
# 	new_pose = [] #25joints*3XYZcoor
# 	skip_value = 0
	

# 	for i in range(25):
# 		if abs(total_vel[n][i][0])>3:
# 			skip_value = 1
# 		new_pose.append( [initial_pose[i][0]+total_vel[n][i][0]*time_diff,
# 			initial_pose[i][1]+total_vel[n][i][1]*time_diff#,
# 			# initial_pose[i][2]+total_vel[n][i][2]*time_diff
# 			])
# 	if skip_value==0:
# 		initial_pose = new_pose
# 		# print(initial_pose)
# 		# time.sleep(2)
# 		total_pose.append(initial_pose)

# for i in range(len(total_pose)):
# 	for j in range(25):
# 		total_pose[i][j] = (int(800-100*total_pose[i][j][0]), int(500-100*total_pose[i][j][1]))
# 	# new_pose = [(x[0], x[1]) for x in new_pose]
# 	# print(new_pose[:][0:2])
# kinect_bones = [
# (0,1),
# (1,2),
# (2,3),
# (3,4),
# (2,5),
# (2,6),
# (4,7),
# (4,8),

# (5,9),
# (9,10),
# (10,11),

# (6,14),
# (14,15),
# (15,16),

# (7,19),
# (19,20),
# # (20,21),

# (8,22),
# (22,23)#,
# # (23,24)
# ]

# kinect_bones_colour = [
# (0, 255, 255),
# (255, 255, 255),
# (255, 255, 255),
# (255, 255, 255),
# (0, 255, 0),
# (0, 0, 255),
# (255, 0, 0),
# (255, 255, 0),


# (0, 255, 0),
# (0, 255, 0),
# (0, 255, 0),

# (0, 0, 255),
# (0, 0, 255),
# (0, 0, 255),

# (255, 0, 0),
# (255, 0, 0),
# # (20,21),

# (255, 255, 0),
# (255, 255, 0)
# # (23,24)
# ]

# for pose in total_pose:
# 	canvas = np.zeros((1800, 1920, 3), np.uint8)

# 	idx = 0
# 	for bones in kinect_bones:
# 		#### joint pos: array of joint coordinates, bones[0] and bones[1]: index of parent and child respectively
# 		start = (int(pose[bones[0]][0]),int(pose[bones[0]][1])) #### picture coordinates of parent joint
# 		end = (int(pose[bones[1]][0]),int(pose[bones[1]][1])) #### picture coordinates of child joint
 
# 		cv2.line(canvas, start, end, kinect_bones_colour[idx], 8) 
# 		idx+=1
# 		# cv2.imshow("skeleton",canvas)
# 		# cv2.waitKey(0)
# 	print(pose)

# 	cv2.imshow("skeleton",canvas)
# 	cv2.waitKey(0)




# 	# for coor in pose:
# 	# 	print(str((int(800+50*coor[0]), int(100+50*coor[1]))))
# 	# 	canvas = cv2.circle(canvas, (int(800+50*coor[0]), int(100+50*coor[1])), radius=2, color=(0, 0, 255), thickness=-1)
# 	# # print(pose)
	
# 	# # vector2screen(pose, 100, (800,100), canvas)