## read 
## use conda env kinectpy3.6

# import csv
import time
import numpy as np
import cv2
# import os
from scipy.spatial.transform import Rotation


KINECT_JOINTS = [
0, # PyKinectV2.JointType_Head, 
1.2, # PyKinectV2.JointType_Neck,
1, # PyKinectV2.JointType_SpineShoulder,
2,# PyKinectV2.JointType_SpineMid,
2, # PyKinectV2.JointType_SpineBase,
1, # PyKinectV2.JointType_ShoulderRight, 
1, # PyKinectV2.JointType_ShoulderLeft, 
0.8, # PyKinectV2.JointType_HipRight, 
0.8, # PyKinectV2.JointType_HipLeft, 
2, # PyKinectV2.JointType_ElbowRight, 
2, # PyKinectV2.JointType_WristRight, 
0, # PyKinectV2.JointType_HandRight, 
0, # PyKinectV2.JointType_HandTipRight, 
0, # PyKinectV2.JointType_ThumbRight, 
2, # PyKinectV2.JointType_ElbowLeft, 
2, # PyKinectV2.JointType_WristLeft, 
0, # PyKinectV2.JointType_HandLeft, 
0, # PyKinectV2.JointType_HandTipLeft, 
0, # PyKinectV2.JointType_ThumbLeft, 
2.5, # PyKinectV2.JointType_KneeRight, 
2.5, # PyKinectV2.JointType_AnkleRight, 
0, # PyKinectV2.JointType_FootRight, 
2.5, # PyKinectV2.JointType_KneeLeft, 
2.5, # PyKinectV2.JointType_AnkleLeft, 
0, # PyKinectV2.JointType_FootLeft
]

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

kinect_bones_colour_ground = [
(0, 175, 175),
(175, 175, 175),
(175, 175, 175),
(175, 175, 175),
(0, 175, 0),
(0, 0, 175),
(175, 0, 0),
(175, 175, 0),


(0, 175, 0),
(0, 175, 0),
(0, 175, 0),

(0, 0, 175),
(0, 0, 175),
(0, 0, 175),

(175, 0, 0),
(175, 0, 0),
# (20,21),

(175, 175, 0),
(175, 175, 0)
# (23,24)
]

##### function to convert velocity (x, y, z) to coordinates
##### assume doppler is 20samples/1 -> 0.05s time difference
##### inputs: array of velocities, original coordinates
##### returns: array of vectors (default: screen space 2d, else camera space 3d)

def vel2canvas(vel, ori_coor, canvas, cameraspace = False, ):
	new_coor = []
	for idx in range(25):
		vel_x, vel_y, vel_z = vel[idx]
		coor_x, coor_y, coor_z = ori_coor[idx]
		new_coor.append([coor_x + vel_x*0.05, coor_y + vel_y*0.05, coor_z + vel_z*0.05])

	idx = 0
	for bones in kinect_bones:
		#### joint pos: array of joint coordinates, bones[0] and bones[1]: index of parent and child respectively
		start = (int(new_coor[bones[0]][0]),int(new_coor[bones[0]][1])) #### picture coordinates of parent joint
		end = (int(new_coor[bones[1]][0]),int(new_coor[bones[1]][1])) #### picture coordinates of child joint

		cv2.line(canvas, start, end, kinect_bones_colour[idx], 8) 
		idx+=1
	return canvas


	# vector_array = []
	# for quat in quats:
	# 	try:
	# 		r = Rotation.from_quat(quat)
	# 		vector_array.append(r.apply(v1, inverse=False))
	# 	except:
	# 		vector_array.append([0,0,0])
	# if not cameraspace:
	# 	###### to transform to 2d picture frame (0,0) at top left
	# 	###### vector is pointing from the parent to the child
	# 	vector_array = [[-1*vector[0], -1*vector[1]] for vector in vector_array]
	# return vector_array

##### function to convert absolute quaternion (x, y, z, w) to vector (parent to child joint)
##### inputs: array of absolute quaternions (normalised)
##### returns: array of vectors (default: screen space 2d, else camera space 3d)

def quat2vector_test(quats, cameraspace = False):
	v1 = [0, 1, 0]
	vector_array = []

	for quat in quats:
		try:
			r = Rotation.from_quat(quat)
			r1 = Rotation.from_quat([-0.048561075941746204, 0.045677638790408474, 0.053275385651582706, -0.05212589089491002])
			v2 = r.apply(v1, inverse=False)
			vector_array.append(r1.apply(v2, inverse=True))
		except:
			# print(quat)
			# time.sleep(1)
			vector_array.append([0,0,0])
	if not cameraspace:
		###### to transform to 2d picture frame (0,0) at top left
		###### vector is pointing from the parent to the child
		# vector_array = [[-1*vector[0], -1*vector[1]] for vector in vector_array]
		vector_array = [[-1*vector[0], -1*vector[1], -1*vector[2]] for vector in vector_array]

	return vector_array

def quat2vector(quats, cameraspace = False):
	v1 = [0, 1, 0]
	vector_array = []

	for quat in quats:
		try:
			r = Rotation.from_quat(quat)
			# r1 = Rotation.from_quat([-0.048561075941746204, 0.045677638790408474, 0.053275385651582706, -0.05212589089491002])
			# v2 = r.apply(v1, inverse=False)
			vector_array.append(r.apply(v1, inverse=False))
		except:
			# print(quat)
			# time.sleep(1)
			vector_array.append([0,0,0])
	if not cameraspace:
		###### to transform to 2d picture frame (0,0) at top left
		###### vector is pointing from the parent to the child
		# vector_array = [[-1*vector[0], -1*vector[1]] for vector in vector_array]
		vector_array = [[-1*vector[0], -1*vector[1], -1*vector[2]] for vector in vector_array]

	return vector_array




##### function to convert vector array to display on canvas
##### inputs: vector_array (25joints x 2), scale: multiplier, 
#####        head_pos: 2d coordinate of head, canvas: 3d numpy array [length, breadth, RGB]
##### outputs: canvas (3d numpy array: [length, breadth, RGB])

def vector2screen (vector_array, scale, head_pos, canvas, ground=0):
	# print(len(vector_array))
	# print(len(vector_array[0]))

	joint_pos = [head_pos]
	# joint_pos.append((800, 150, 800))
	# joint_pos.append((800, 200, 800))
	# joint_pos.append((800, 300, 800))
	if ground==1:
		colour=kinect_bones_colour_ground
	else:
		colour = kinect_bones_colour


	for bones in kinect_bones[0:]:
		if len(joint_pos)==12 or len(joint_pos)==17:  ##### ignore thumb and finger positions 
			joint_pos.append((0.0, 0.0, 0.0)) # joint 12/17
			joint_pos.append((0.0, 0.0, 0.0)) # joint 13/18

		if len(joint_pos)==21:  ##### ignore left foot posiiton (right foot position is index 24(last), so no need to append)
			joint_pos.append((0.0, 0.0, 0.0)) # joint 21

		parent_index = bones[0] #### index of parent
		child_index = bones[1] #### index of child
		current_pos = joint_pos[parent_index] #### picture coordinates of parent joint

		vector = vector_array[child_index] #### vector pointing from parent joint to child joint
		vector = (vector[0]*scale*KINECT_JOINTS[child_index], vector[1]*scale*KINECT_JOINTS[child_index], vector[2]*scale*KINECT_JOINTS[child_index])

		#### find the child joint (parent coordinate + vector from parent to child)
		next_pos = [current_pos[0] + vector[0] , current_pos[1] + vector[1], current_pos[2] + vector[2]]
		if bones[1]<5:
			next_pos = [-vector[0] + current_pos[0], -vector[1] +current_pos[1], -vector[2] +current_pos[2]]

		joint_pos.append(next_pos)

	idx = 0
	for bones in kinect_bones:
		#### joint pos: array of joint coordinates, bones[0] and bones[1]: index of parent and child respectively
		start = (int(joint_pos[bones[0]][0]),int(joint_pos[bones[0]][1])) #### picture coordinates of parent joint
		end = (int(joint_pos[bones[1]][0]),int(joint_pos[bones[1]][1])) #### picture coordinates of child joint

		cv2.line(canvas, start, end, colour[idx], 8) 
		idx+=1
	return canvas









