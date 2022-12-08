## read 

import csv
import time
import numpy as np
import cv2
import os



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
(20,21),

(8,22),
(22,23),
(23,24)
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
(255, 0, 0),

(255, 255, 0),
(255, 255, 0),
(255, 255, 0)
]

file_csv = "07-12-22_17-06-45" #.csv # in joint folder

output_dir = "output/new/"
if not (os.path.exists(output_dir)):
    os.makedirs(output_dir) # Create a new directory because it does not exist
output_name = output_dir+file_csv+'.avi'
# annotated_output_name = output_dir+now+'_annotated.avi'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_name, fourcc, 30.0, (1920, 1800))

with open('joint/'+file_csv+'.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
    	# print(row)
    	# time.sleep(2)
    	if (line_count%2)==1:
        	# row = row.replace('"', "")
    		body_coor = []
    		canvas = np.zeros((1800, 1920, 3), np.uint8)
    		

    		if print_date==1:
    			font = cv2.FONT_HERSHEY_PLAIN
    			cv2.putText(canvas, row[0], (20, 40), font, 2, (255, 255, 255), 2)

    		for x in row[1:]:
        		# print(x)
        		x = x.replace(")", "")
        		x = x.replace("(", "")
        		x = x.split(", ")#[1:]
        		coor = [int(float(i)*500) for i in x]
        		body_coor.append(coor)


    		idx = 0
    		for bones in kinect_bones:
        		start = (body_coor[bones[0]][0]+900,500-body_coor[bones[0]][1])
        		end = (body_coor[bones[1]][0]+900,500-body_coor[bones[1]][1])

        		cv2.line(canvas, start, end, kinect_bones_colour[idx], 8) 
        		idx+=1
    		# cv2.imshow("skeleton",canvas)
    		# cv2.waitKey(0)
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