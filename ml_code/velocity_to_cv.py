## read 
## use conda env pytorchpy3.9

# import csv
import time
import numpy as np
# import cv2
import os
from scipy.spatial.transform import Rotation
# from scipy.stats import linregress
import pandas as pd
import re
# import torch
import math
from velocity_functions import loadDataXYZ
from kinect_quat_functions import vector2screen
import cv2

joints = [
'Head',
'Neck',
'SpineShoulder',
'SpineMid',
'SpineBase',
'ShoulderRight',
'ShoulderLeft',
'HipRight',
'HipLeft',
'ElbowRight',
'WristRight',
'HandRight',
'HandTipRight',
'ThumbRight',
'ElbowLeft',
'WristLeft',
'HandLeft',
'HandTipLeft',
'ThumbLeft',
'KneeRight',
'AnkleRight',
'FootRight',
'KneeLeft',
'AnkleLeft',
'FootLeft',
]

# loadDataXYZ(tx = True, actions = "all"):


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
# name = "punch"
f = open("state_dict_model_outputlog_velocitymodel_17feb_walk_24000_trevor_MSE.txt", "r")

digit = ['1','2','3','4','5','6','7','8','9','0','-']
lines = f.readlines()
counter_xyz=0
counter_body=0

bodies = []
body_coor = []
coordinates = []
for line in lines:
	line = line.split(',')
	line = [l.strip() for l in line]
	
	for x in line:
		if x:
			if x[0]=="d":
				continue
			while x[0] not in digit:
				x = x[1:]
			while x[-1] not in digit:
				x = x[:-1]
			coordinates.append(float(x))
			counter_xyz+=1
			if counter_xyz%3==0:
				body_coor.append(coordinates)
				# print(coordinates)
				# error
				coordinates = []
				counter_body+=1
				if counter_body%25==0:
					bodies.append(body_coor)
					body_coor = []
# print(len(bodies))
# print(len(bodies[0]))
# print(len(bodies[0][0]))
# error

	# for x in coordinates:
	# 	if x>3 or x<-3:
	# 		print(line[0]+str(coordinates))
	# 		time.sleep(2)
	# 		continue

file = '../data/8Dec/labels/smallest/rx/trevor_walk1.txt'
vid_file = "trevor_walk1_24000_MSE"
f = open(file, "r")
lines = f.readlines()
coord_file = pd.read_csv(lines[1][:-1])
line = lines[2].split(":")
indexes = line[1].strip("\n").strip('][ ').split(', ')
# if len(indexes)<5:
# 	continue

midpoint = math.floor(len(indexes)/2)
idx = int(indexes[midpoint])
initial_pose = coord_file.iloc[idx-1].values
# print(initial_pose)
# error
initial_pose = initial_pose[1:]
initial_pose = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in initial_pose]
initial_pose = [[float(x[0]), float(x[1])]for x in initial_pose]
# # print(initial_pose)
# # error

time_diff = 1/20 #20fps
# initial_pose = [] #25joints*3XYZcoor
# total_vel = [] #n timesteps*25joints*3XYZcoor
new_pose = [] #25joints*3XYZcoor
total_pose = []
total_pose.append(initial_pose)

coord_file_idx = 2
total_vel = bodies
for n in range(len(total_vel)):
	new_pose = [] #25joints*3XYZcoor
	skip_value = 0
	

	for i in range(25):
		if abs(total_vel[n][i][0])>3 or abs(total_vel[n][i][1])>3:
			skip_value = 1
		# for j in range(2):
		# 	if abs(total_vel[n][i][j])<0.005:
		# 		total_vel[n][i][j]=0
		new_pose.append( [initial_pose[i][0]+total_vel[n][i][0]*time_diff,
			initial_pose[i][1]+total_vel[n][i][1]*time_diff#,
			# initial_pose[i][2]+total_vel[n][i][2]*time_diff
			])
	if skip_value==0:
		coord_file_idx+=1
		
		# try:

		line = lines[coord_file_idx].split(":")
		indexes = line[1].strip("\n").strip('][ ').split(', ')
		# if len(indexes)<5:
		# 	continue

		#### UPDATE POSE with actual pose (temporary replacement for pose estimation)
		#### every 10 frames

		if coord_file_idx%100==0:
			midpoint = math.floor(len(indexes)/2)
			idx = int(indexes[midpoint])
			# print(idx)
			initial_pose = coord_file.iloc[idx-1].values #quaternion indexes, need to -1 to get velocity indexes
			initial_pose = initial_pose[1:]
			initial_pose = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in initial_pose]
			initial_pose = [[float(x[0]), float(x[1])]for x in initial_pose]

		#### other 9 frames update pose recursively based on calculated velocity
		else: 
			initial_pose = new_pose
		# except:
		# 	print(coord_file_idx)
		# 	print("errorhere")
		# initial_pose = new_pose
		# print(initial_pose)
		# time.sleep(2)
		total_pose.append(new_pose)

for i in range(len(total_pose)):
	for j in range(25):
		total_pose[i][j] = (int(800-200*total_pose[i][j][0]), int(500-200*total_pose[i][j][1]))
	# new_pose = [(x[0], x[1]) for x in new_pose]
	# print(new_pose[:][0:2])
kinect_bones = [
(0,1),
(1,2),
(2,3),
(3,4),
(2,5),
(2,6),
(4,7),
(4,8),

(5,9),
(9,10),
(10,11),

(6,14),
(14,15),
(15,16),

(7,19),
(19,20),
# (20,21),

(8,22),
(22,23)#,
# (23,24)
]

kinect_bones_colour = [
(0, 255, 255),
(255, 255, 255),
(255, 255, 255),
(255, 255, 255),
(0, 255, 0),
(0, 0, 255),
(255, 0, 0),
(255, 255, 0),


(0, 255, 0),
(0, 255, 0),
(0, 255, 0),

(0, 0, 255),
(0, 0, 255),
(0, 0, 255),

(255, 0, 0),
(255, 0, 0),
# (20,21),

(255, 255, 0),
(255, 255, 0)
# (23,24)
]

######## video writing init

output_dir = "output/vel200epochs/"
if not (os.path.exists(output_dir)):
	os.makedirs(output_dir) # Create a new directory because it does not exist
output_name = output_dir+vid_file+'.avi'
# annotated_output_name = output_dir+now+'_annotate

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_name, fourcc, 20.0, (1920, 1800))

##### end video wriitng init



for pose in total_pose:
	canvas = np.zeros((1800, 1920, 3), np.uint8)

	idx = 0
	for bones in kinect_bones:
		#### joint pos: array of joint coordinates, bones[0] and bones[1]: index of parent and child respectively
		start = (int(pose[bones[0]][0]),int(pose[bones[0]][1])) #### picture coordinates of parent joint
		end = (int(pose[bones[1]][0]),int(pose[bones[1]][1])) #### picture coordinates of child joint
 
		cv2.line(canvas, start, end, kinect_bones_colour[idx], 8) 
		idx+=1
		# cv2.imshow("skeleton",canvas)
		# cv2.waitKey(0)
	# print(pose)

	# cv2.imshow("skeleton",canvas)
	# cv2.waitKey(0)
	out.write(canvas.astype('uint8'))
out.release()




	# for coor in pose:
	# 	print(str((int(800+50*coor[0]), int(100+50*coor[1]))))
	# 	canvas = cv2.circle(canvas, (int(800+50*coor[0]), int(100+50*coor[1])), radius=2, color=(0, 0, 255), thickness=-1)
	# # print(pose)
	
	# # vector2screen(pose, 100, (800,100), canvas)