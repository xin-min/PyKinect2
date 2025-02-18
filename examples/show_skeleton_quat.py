## read 
## use conda env kinectpy3.6

import csv
import time
import numpy as np
import cv2
import transforms3d
import os
from scipy.spatial.transform import Rotation



# KINECT_JOINTS = [
# PyKinectV2.JointType_Head, 
# PyKinectV2.JointType_Neck,
# PyKinectV2.JointType_SpineShoulder,
# PyKinectV2.JointType_SpineMid,
# PyKinectV2.JointType_SpineBase,
# PyKinectV2.JointType_ShoulderRight, 
# PyKinectV2.JointType_ShoulderLeft, 
# PyKinectV2.JointType_HipRight, 
# PyKinectV2.JointType_HipLeft, 
# PyKinectV2.JointType_ElbowRight, 
# PyKinectV2.JointType_WristRight, 
# PyKinectV2.JointType_HandRight, 
# PyKinectV2.JointType_HandTipRight, 
# PyKinectV2.JointType_ThumbRight, 
# PyKinectV2.JointType_ElbowLeft, 
# PyKinectV2.JointType_WristLeft, 
# PyKinectV2.JointType_HandLeft, 
# PyKinectV2.JointType_HandTipLeft, 
# PyKinectV2.JointType_ThumbLeft, 
# PyKinectV2.JointType_KneeRight, 
# PyKinectV2.JointType_AnkleRight, 
# PyKinectV2.JointType_FootRight, 
# PyKinectV2.JointType_KneeLeft, 
# PyKinectV2.JointType_AnkleLeft, 
# PyKinectV2.JointType_FootLeft
# ]

print_date = 1

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

files = []
for file in os.listdir("../data/8Dec/joint_tx"):
	if file.endswith(".csv"):
		files.append(file)
		# print(os.path.join("/mydir", file))

for file_csv in files:
	print(file_csv[:-4])
	# time.sleep(3)

	# file_csv = "/08-12-22_14-58-38"+".csv" #.csv # in joint folder

	# ##### video writing init

	# output_dir = "output/8Dec/joint_tx_quat/"
	# if not (os.path.exists(output_dir)):
	# 	os.makedirs(output_dir) # Create a new directory because it does not exist
	# output_name = output_dir+file_csv[:-4]+'.avi'
	# # annotated_output_name = output_dir+now+'_annotate

	# fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# out = cv2.VideoWriter(output_name, fourcc, 30.0, (1920, 1800))

	# ##### end video wriitng init


	with open('../data/8Dec/joint_tx/'+file_csv) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			# print(row)
			# time.sleep(2)
			if line_count > 0 and line_count%2==0:
				# row = row.replace('"', "")
				body_quat = []
				canvas = np.zeros((1800, 1920, 3), np.uint8)

				if print_date==1:
					font = cv2.FONT_HERSHEY_PLAIN
					cv2.putText(canvas, row[0], (20, 40), font, 2, (255, 255, 255), 2)

				
				
				for x in row[1:]:
					# print(x)
					x = x.replace(")", "")
					x = x.replace("(", "")
					x = x.split(", ")#[1:]
					# quat = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
					# quat = [i for i in x]

					# quat = [float(x[3]), float(x[0]), float(x[1]), float(x[2])] ## camera space (x is towards the left, y is up, z is forward)

					quat = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
					body_quat.append(quat)

				v1 = [0,1,0]

				transformed_vector = []
				for quat in body_quat:
					try:
						r = Rotation.from_quat(quat)
						transformed_vector.append(r.apply(v1, inverse=False))
					except:
						transformed_vector.append([0,0,0])
					# transformed_vector.append(transforms3d.quaternions.rotate_vector(v1, quat, is_normalized=True))
				# print(np.array(transformed_quat))

				# joint_pos = np.array([[500,500]])
				joint_pos = [(800,100)]
				tf_quats = [(0.0, 0.0)]
				# tf_quats = np.array([(0.0,0.0)],)
				for index in range(1,len(transformed_vector)):
					# if not transformed_quat(index):
					# 	continue
					vector = transformed_vector[index]
					# quat = (quat[0]*10*KINECT_JOINTS[index], quat[1]*10*KINECT_JOINTS[index])
					vector = (vector[0]*50*KINECT_JOINTS[index], vector[1]*50*KINECT_JOINTS[index])

					###### to transform to 2d picture frame (0,0) at top left
					##### vector is pointing from the parent to the child
					vector = (-1*vector[0], -1*vector[1])


					tf_quats.append(vector)

				for bones in kinect_bones:
					if len(joint_pos)==12 or len(joint_pos)==17:  ##### ignore thumb and finger positions 
						joint_pos.append((0.0, 0.0)) # joint 12/17
						joint_pos.append((0.0, 0.0)) # joint 13/18

					if len(joint_pos)==21:  ##### ignore left foot posiiton (right foot position is index 24(last), so no need to append)
						joint_pos.append((0.0, 0.0)) # joint 21


					joint1_index = bones[0] #### index of parent
					joint2_index = bones[1] #### index of child
					current_pos = joint_pos[joint1_index] #### picture coordinate of parent joint
					tf_joint = tf_quats[joint2_index] #### vector pointing from parent joint to child joint

					#### find the child joint (parent coordinate + vector from parent to child)
					next_pos = (current_pos[0] + tf_joint[0] , current_pos[1] + tf_joint[1] )
					if bones[1]<5:
						next_pos = (-tf_joint[0] + current_pos[0], -tf_joint[1] +current_pos[1])

					joint_pos.append((next_pos))

				idx = 0
				for bones in kinect_bones:
					#### joint pos: array of joint coordinates, bones[0] and bones[1]: index of parent and child respectively
					start = (int(joint_pos[bones[0]][0]),int(joint_pos[bones[0]][1])) #### picture coordinates of parent joint
					end = (int(joint_pos[bones[1]][0]),int(joint_pos[bones[1]][1])) #### picture coordinates of child joint

					cv2.line(canvas, start, end, kinect_bones_colour[idx], 8) 
					idx+=1
				cv2.imshow("skeleton_quaternions",canvas)
				cv2.waitKey(0)
				# print(len(canvas[0]))
				# out.write(canvas.astype('uint8'))

				# print(coor)
				# body_coor.append(coor)
				# row = str(row).split("(")[1:]
				# print(row)
				# time.sleep(2)
		#         print(f'Column names are {", ".join(row)}')
			line_count += 1
		out.release()
		#     else:
		#         print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
		#         line_count += 1
		# print(f'Processed {line_count} lines.')