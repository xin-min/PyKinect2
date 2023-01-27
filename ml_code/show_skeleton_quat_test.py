## read 
## use conda env kinectpy3.6

import csv
import time
import numpy as np
import cv2
import transforms3d
import os
from scipy.spatial.transform import Rotation
from kinect_quat_functions import quat2vector, vector2screen



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
			if line_count > 0 and line_count%2==0:
				body_quat = []
				
				for x in row[1:]:
					# print(x)
					x = x.replace(")", "")
					x = x.replace("(", "")
					x = x.split(", ")

					quat = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
					body_quat.append(quat)

				vector_array = quat2vector(body_quat, cameraspace = False)
				head_pos = (800,100)
				scale = 50
				canvas = np.zeros((1800, 1920, 3), np.uint8)
				canvas = vector2screen(vector_array, scale, head_pos, canvas)

				# if print_date==1:
				# 	font = cv2.FONT_HERSHEY_PLAIN
				# 	cv2.putText(canvas, row[0], (20, 40), font, 2, (255, 255, 255), 2)

				cv2.imshow("skeleton_quaternions",canvas)
				cv2.waitKey(0)

			line_count += 1
		out.release()