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

f_errors = open("huber_newlstm/model_error_normalised_20frames.txt", "a+")
f_est = open("huber_newlstm/IA_DW2.txt", "r")
f_truth = open("huber_newlstm/IA_DW2_truth.txt", "r")
f_kinect = open("huber_newlstm/IA_DW2_kinect.txt", "r")

# file = '../data/8Dec/labels/smallest/rx/IA_kick1.txt'
vid_file = "IA_DW2"
update_num = 20


######## video writing init
save_vid =0
# output_dir = "huber_newlstm/"
# if not (os.path.exists(output_dir)):
# 	os.makedirs(output_dir) # Create a new directory because it does not exist
# output_name = output_dir+vid_file+'.avi'
# # annotated_output_name = output_dir+now+'_annotate

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_name, fourcc, 20.0, (1920, 1800))

##### end video wriitng init


total_bodies = []
# est = 1
for f in [f_est, f_truth]:
	f_idx = 0

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
		

	# 	############################ velocity without tree ##################################

	# 	for x in line:
	# 		if x:
	# 			if x[0]=="d":
	# 				continue
	# 			while x[0] not in digit:
	# 				x = x[1:]
	# 			while x[-1] not in digit:
	# 				x = x[:-1]
	# 			coordinates.append(float(x))
	# 			counter_xyz+=1
	# 			if counter_xyz%3==0:
	# 				body_coor.append(coordinates)
	# 				# print(coordinates)
	# 				# error
	# 				coordinates = []
	# 				counter_body+=1
	# 				if counter_body%25==0:
	# 					bodies.append(body_coor)
	# 					body_coor = []
	# total_bodies.append(bodies)


	# 	############################ velocity without tree end ##################################



		############################ velocity with tree ##################################

		for x in line:
			if x:
				if x[0]=="d":
					continue
				while x[0] not in digit:
					x = x[1:]
				while x[-1] not in digit:
					x = x[:-1]
				# print(float(x))
				coordinates.append(float(x))
				counter_xyz+=1
				if counter_xyz%3==0:
					body_coor.append(coordinates)
					# print(coordinates)
					# time.sleep(1)
					# error
					coordinates = []
					counter_body+=1
					# print(counter_body)
					if counter_body%31==0:
						# print("test")
						# time.sleep(1)
						bodies.append(body_coor)
						body_coor = []

	temp_bodies = []
	for body_frame in bodies:
		temp_body_frame = np.zeros((25,3))
		for x in range(len(body_frame)):
			temp_body_frame[joints_tree_order[x]] = body_frame[x]

		# print(temp_body_frame)
		# time.sleep(1)
		temp_bodies.append(temp_body_frame)
	# bodies = []
	# if est ==1:
	# 	for i in range(len(temp_bodies)):
	# 		if (i+1)%5==0:
	# 			bodies.append(temp_bodies[i])
	# 	est = 0
	# else:
	bodies = temp_bodies
	total_bodies.append(bodies)




	########################### velocity with tree end ##################################



############################# extract data from kinect (no tree) ############################

digit = ['1','2','3','4','5','6','7','8','9','0','-']
lines = f_kinect.readlines()
counter_xyz=0
counter_body=0

kinect_ground_truths = []
# bodies = []
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
					kinect_ground_truths.append(body_coor)
					body_coor = []

# shape of kinect ground truth: number of timestamps * 25 *3

# print(len(kinect_ground_truth))
# print(len(kinect_ground_truth[0]))
# print(len(kinect_ground_truth[0][0]))


# total_bodies.append(bodies)

############################# end extract data from kinect ############################

# print(len(bodies))
# print(len(bodies[0]))
# print(len(bodies[0][0]))
# error

	# for x in coordinates:
	# 	if x>3 or x<-3:
	# 		print(line[0]+str(coordinates))
	# 		time.sleep(2)
	# 		continue
total_poses = []
total_error = []
original_pose = []

for bodies in total_bodies:

	initial_pose = kinect_ground_truths[0].copy()
	initial_pose = [[float(x[0]), float(x[1])]for x in initial_pose]

	# print(initial_pose)


	# f = open(file, "r")
	# lines = f.readlines()
	# coord_file = pd.read_csv(lines[1][:-1])
	# line = lines[2].split(":")
	# indexes = line[1].strip("\n").strip('][ ').split(', ')
	# # if len(indexes)<5:
	# # 	continue

	# midpoint = math.floor(len(indexes)/2)
	# idx = int(indexes[midpoint])
	# initial_pose = coord_file.iloc[idx-1].values # index given is for quat, vel is quat index - 1
	# # print(initial_pose)
	# # error
	# initial_pose = initial_pose[1:] # index 0 is timestamp
	# initial_pose = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in initial_pose]
	# initial_pose = [[float(x[0]), float(x[1])]for x in initial_pose]
	# print(initial_pose)
	# error


	time_diff = 1/20 #20fps
	# initial_pose = [] #25joints*3XYZcoor
	# total_vel = [] #n timesteps*25joints*3XYZcoor
	new_pose = [] #25joints*3XYZcoor
	total_pose = []
	kinect_poses = []

	total_pose.append(initial_pose.copy())
	# kinect_poses.append(initial_pose.copy())
	# print(initial_pose)


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
		# print(new_pose)
		# time.sleep(1)
		if skip_value==0:
			coord_file_idx+=1
			
			try:
				kinect_pose = kinect_ground_truths[n].copy()

				# line = lines[coord_file_idx].split(":")
				# indexes = line[1].strip("\n").strip('][ ').split(', ')
				# midpoint = math.floor(len(indexes)/2)
				# idx = int(indexes[midpoint])
				# # print(idx)
				# kinect_pose = coord_file.iloc[idx-1].values #quaternion indexes, need to -1 to get velocity indexes
				# kinect_pose = kinect_pose[1:]
				# kinect_pose = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in kinect_pose]
				# kinect_pose = [[float(x[0]), float(x[1])]for x in kinect_pose]
				kinect_poses.append(kinect_pose)
				# if len(indexes)<5:
				# 	continue

				#### UPDATE POSE with actual pose (temporary replacement for pose estimation)
				#### every 10 frames

				if coord_file_idx%update_num==0:
					initial_pose = kinect_pose

				#### other 9 frames update pose recursively based on calculated velocity
				else: 
					initial_pose = new_pose
			except:
				print(coord_file_idx)
				print("errorhere")
			# initial_pose = new_pose
			# print(initial_pose)
			# time.sleep(2)
			total_pose.append(new_pose)

	total_pose_temp = np.empty((len(total_pose)-1,25), dtype=object)
	if f_idx ==0:
		spacing =300
		# pose_error = np.zeros((len(total_pose)-1,25))
		# pose_error_normalised = np.zeros((len(total_pose)-1,25))
		


		for i in range(len(total_pose)-1):
			for j in range(25):
				# index 3 is spine mid
				# pose_error_normalised[i][j] = math.sqrt(math.pow(((total_pose[i][j][0]-total_pose[i][3][0])-(kinect_poses[i][j][0]-kinect_poses[i][3][0])),2) + math.pow(((total_pose[i][j][1]-total_pose[i][3][1])-(kinect_poses[i][j][1]-kinect_poses[i][3][1])),2))
				# pose_error[i][j] = math.sqrt(math.pow((total_pose[i][j][0]-kinect_poses[i][j][0]),2) + math.pow((total_pose[i][j][1]-kinect_poses[i][j][1]),2))
				total_pose_temp[i][j] = ((spacing-200*total_pose[i][j][0]), (500-200*total_pose[i][j][1]))
				cv2.waitKey(0)
				
	else:
		# print("hello")
		spacing = 700
		for i in range(len(total_pose)-1):
			for j in range(25):
				total_pose_temp[i][j] = ((spacing-200*total_pose[i][j][0]), (500-200*total_pose[i][j][1]))
				kinect_poses[i][j] = ((1100-200*kinect_poses[i][j][0]), (500-200*kinect_poses[i][j][1]))


		# new_pose = [(x[0], x[1]) for x in new_pose]
		# print(new_pose[:][0:2])
	original_pose.append(total_pose)
	total_poses.append(total_pose_temp)
	# total_error.append(pose_error)
	f_idx+=1
	# print(kinect_poses[0])
	# print(kinect_poses[1])
	# error
pose_error = np.zeros((len(total_poses[0])-1,25))
pose_error_normalised = np.zeros((len(total_poses[0])-1,25))
for i in range(len(total_poses[0])-1):
	for j in range(25):
		pose_error_normalised[i][j] = math.sqrt(math.pow(((original_pose[0][i][j][0]-original_pose[0][i][3][0])-(original_pose[1][i][j][0]-original_pose[1][i][3][0])),2) + math.pow(((original_pose[0][i][j][1]-original_pose[0][i][3][1])-(original_pose[1][i][j][1]-original_pose[1][i][3][1])),2))
		pose_error[i][j] = math.sqrt(math.pow((original_pose[0][i][j][0]-original_pose[1][i][j][0]),2) + math.pow((original_pose[0][i][j][1]-original_pose[1][i][j][1]),2))
				

joint_idx = [0,1,3,4,5,6,7,8,9,10,14,15,19,20,22,23]

kinect_bones = [
(0,1),
# (1,2),
(1,3),
(3,4),
(1,5),
(1,6),
(4,7),
(4,8),

(5,9),
(9,10),
# (10,11), #hands

(6,14),
(14,15),
# (15,16), #hands

(7,19),
(19,20),
# (20,21),

(8,22),
(22,23)#,
# (23,24)
]

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
# (10,11), #hands

# (6,14),
# (14,15),
# (15,16), #hands

# (7,19),
# (19,20),
# # (20,21),

# (8,22),
# (22,23)#,
# # (23,24)
# ]

kinect_bones_colour = [
(0, 255, 255),
# (255, 255, 255),
(255, 255, 255),
(255, 255, 255),
(0, 255, 0),
(0, 0, 255),
(255, 0, 0),
(255, 255, 0),


(0, 255, 0),
(0, 255, 0),
# (0, 255, 0), #hands

(0, 0, 255),
(0, 0, 255),
# (0, 0, 255), #hands

(255, 0, 0),
(255, 0, 0),
# (20,21),

(255, 255, 0),
(255, 255, 0)
# (23,24)
]

kinect_bones_colour_ground = [
(0, 175, 175),
# (175, 175, 175),
(175, 175, 175),
(175, 175, 175),
(0, 175, 0),
(0, 0, 175),
(175, 0, 0),
(175, 175, 0),


(0, 175, 0),
(0, 175, 0),
# (0, 175, 0),

(0, 0, 175),
(0, 0, 175),
# (0, 0, 175),

(175, 0, 0),
(175, 0, 0),
# (20,21),

(175, 175, 0),
(175, 175, 0)
# (23,24)
]

# ######## video writing init

# output_dir = "newlstm/"
# if not (os.path.exists(output_dir)):
# 	os.makedirs(output_dir) # Create a new directory because it does not exist
# output_name = output_dir+vid_file+'.avi'
# # annotated_output_name = output_dir+now+'_annotate

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_name, fourcc, 20.0, (1920, 1800))

# ##### end video wriitng init

# font = cv2.FONT_HERSHEY_SIMPLEX
# # cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)


# for i in range(len(total_poses[0])-1):
# 	pose = total_poses[0][i]
# 	try:
# 		ground_pose = total_poses[1][i]
# 		kinect_pose = kinect_poses[i]
# 	except:
# 		print(i)
# 		print("error")
# 	# print(pose)
# 	canvas = np.zeros((1800, 1920, 3), np.uint8)

# 	idx = 0
# 	for bones in kinect_bones:
# 		# print(pose[bones[0]][0])


# 		#### joint pos: array of joint coordinates, bones[0] and bones[1]: index of parent and child respectively
# 		start = (int(pose[bones[0]][0]),int(pose[bones[0]][1])) #### picture coordinates of parent joint
# 		end = (int(pose[bones[1]][0]),int(pose[bones[1]][1])) #### picture coordinates of child joint
 
# 		cv2.line(canvas, start, end, kinect_bones_colour[idx], 8) 
# 		# try:

# 		start = (int(kinect_pose[bones[0]][0]),int(kinect_pose[bones[0]][1])) #### picture coordinates of parent joint
# 		end = (int(kinect_pose[bones[1]][0]),int(kinect_pose[bones[1]][1])) #### picture coordinates of child joint
	 
# 		cv2.line(canvas, start, end, kinect_bones_colour_ground[idx], 8) 
# 		# except:
# 		# 	print(idx)

# 		# try:

# 		start = (int(ground_pose[bones[0]][0]),int(ground_pose[bones[0]][1])) #### picture coordinates of parent joint
# 		end = (int(ground_pose[bones[1]][0]),int(ground_pose[bones[1]][1])) #### picture coordinates of child joint
 
# 		cv2.line(canvas, start, end, kinect_bones_colour_ground[idx], 8) 
# 		# except:
# 		# 	print(idx)

# 		idx+=1
# 	# for joint in joint_idx:
# 	# 	# print(total_error[0][i][joint])
# 	# 	# print(pose[joint])

# 	# 	cv2.putText(canvas,str("%.3f"%pose_error[i][joint]),(int(pose[joint][0]), int(pose[joint][1])), font, 0.5,(255,255,255),2,cv2.LINE_AA)

# 	cv2.putText(canvas,"model",(400,300), font, 0.5,(255,255,255),2,cv2.LINE_AA)
# 	cv2.putText(canvas,"calculated velocity",(750,300), font, 0.5,(255,255,255),2,cv2.LINE_AA)
# 	cv2.putText(canvas,"kinect ground truth",(1150,300), font, 0.5,(255,255,255),2,cv2.LINE_AA)

# 	# cv2.imshow("skeleton",canvas)
# 	# cv2.waitKey(0)
# 	# print(pose)
# 	if save_vid ==0:
# 		cv2.imshow("skeleton",canvas)
# 		cv2.waitKey(0)		
# 	else:
# 		out.write(canvas.astype('uint8'))
# if save_vid ==1:
# 	out.release()
print(vid_file + " error: " + str(np.mean(pose_error, axis=0)) + "\n")
print(vid_file + " normalised error: " + str(np.mean(pose_error_normalised, axis=0)) + "\n")
f_errors.write(vid_file + " error: " + str(np.mean(pose_error, axis=0)) + "\n")
f_errors.write(vid_file + " normalised: " + str(np.mean(pose_error_normalised, axis=0)) + "\n")





	# for coor in pose:
	# 	print(str((int(800+50*coor[0]), int(100+50*coor[1]))))
	# 	canvas = cv2.circle(canvas, (int(800+50*coor[0]), int(100+50*coor[1])), radius=2, color=(0, 0, 255), thickness=-1)
	# # print(pose)
	
	# # vector2screen(pose, 100, (800,100), canvas)