## read 
## use conda env kinectpy3.6

import csv
import time
import numpy as np
import cv2
import transforms3d
import os

error dont overwrite file


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

	output_dir = "output/8Dec/joint_tx_quat/"
	if not (os.path.exists(output_dir)):
		os.makedirs(output_dir) # Create a new directory because it does not exist
	output_name = output_dir+file_csv[:-4]+'.avi'
	# annotated_output_name = output_dir+now+'_annotate

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_name, fourcc, 30.0, (1920, 1800))


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

				transformed_quat = []
				
				for x in row[1:]:
					# print(x)
					x = x.replace(")", "")
					x = x.replace("(", "")
					x = x.split(", ")#[1:]
					# quat = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
					# quat = [i for i in x]
					quat = [float(x[1]), -float(x[2]), -float(x[3]), float(x[0])]
					body_quat.append(quat)

				v1 = [0,1,0]
				# v1 = [1,0,0]


				for quat in body_quat:
					transformed_quat.append(transforms3d.quaternions.rotate_vector(v1, quat, is_normalized=True))
				# print(np.array(transformed_quat))

				# joint_pos = np.array([[500,500]])
				joint_pos = [(800,500)]
				tf_quats = [(0.0, 0.0)]
				# tf_quats = np.array([(0.0,0.0)],)
				for index in range(1,len(transformed_quat)):
					# if not transformed_quat(index):
					# 	continue
					quat = transformed_quat[index]
					# quat = (quat[0]*10*KINECT_JOINTS[index], quat[1]*10*KINECT_JOINTS[index])
					quat = (quat[0]*50*KINECT_JOINTS[index], quat[1]*50*KINECT_JOINTS[index])

					# quat = np.array((quat[0]*100*KINECT_JOINTS[index], quat[2]*100*KINECT_JOINTS[index]))
					# print(quat)
					tf_quats.append(quat)
					# tf_quats= np.append(tf_quats, quat)
				# print(len(tf_quats))

				for bones in kinect_bones:
					if len(joint_pos)==12 or len(joint_pos)==17:
						joint_pos.append((0.0, 0.0)) # joint 12/17
						joint_pos.append((0.0, 0.0)) # joint 13/18

						# joint_pos = np.append(joint_pos,(0,0)) # joint 12/17
						# joint_pos = np.append(joint_pos,(0,0)) # joint 13/18

					if len(joint_pos)==21:
						joint_pos.append((0.0, 0.0)) # joint 21

						# joint_pos = np.append(joint_pos,(0,0)) # joint 21

					joint1_index = bones[0]
					joint2_index = bones[1]
					current_pos = joint_pos[joint1_index]
					tf_joint = tf_quats[joint2_index]

					# print(type(tf_joint))
					# print(tf_joint)
					# print(type(current_pos))
					next_pos = (tf_joint[0] + current_pos[0], tf_joint[1] + current_pos[1])
					if bones[1]>4:
						next_pos = (-tf_joint[0] + current_pos[0], -tf_joint[1] +current_pos[1])

					joint_pos.append((next_pos))
					# np.append(joint_pos, next_pos)
					# print(current_pos)
				# print(joint_pos)
				# time.sleep(3)


				idx = 0
				for bones in kinect_bones:
					start = (int(joint_pos[bones[0]][0]),int(joint_pos[bones[0]][1]))
					end = (int(joint_pos[bones[1]][0]),int(joint_pos[bones[1]][1]))

					cv2.line(canvas, start, end, kinect_bones_colour[idx], 8) 
					idx+=1
				# cv2.imshow("skeleton_quaternions",canvas)
				# cv2.waitKey(0)
				# print(len(canvas[0]))
				out.write(canvas.astype('uint8'))

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