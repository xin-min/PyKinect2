## read 
## use conda env kinectpy3.6

import csv
import time
import numpy as np
import cv2
import transforms3d
import os
import math



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

# files = []
# for file in os.listdir("../data/8Dec/joint_tx"):
# 	if file.endswith(".csv"):
# 		files.append(file)
# 		# print(os.path.join("/mydir", file))

# start_quat = [
# (0.0, 0.0, 0.0, 0.0),
# (-0.04246267303824425, 0.9979426860809326, 0.0356350913643837, -0.032210011035203934),
# (-0.04057622328400612, 0.9969627261161804, 0.05762607976794243, -0.03313834220170975), 
# (-0.03827853500843048, 0.9946541786193848, 0.07187400758266449, -0.06349731236696243),
# (-0.04036831483244896, 0.9923662543296814, 0.07072141766548157, -0.0926176905632019),
# (0.6259497404098511, 0.7750303745269775, 0.08600489795207977, -0.010867989622056484),
# (0.8160269260406494, -0.5663284063339233, -0.10661493986845016, 0.04478256404399872),
# (0.6116636395454407, 0.7370321750640869, -0.14625492691993713, -0.2475091516971588),
# (0.7184550762176514, -0.6806216835975647, 0.030664101243019104, -0.1401287019252777),
# (0.8144869804382324, -0.03534862771630287, 0.5595877766609192, -0.14907319843769073),
# (0.4560256600379944, -0.4964819848537445, 0.7191545367240906, -0.16841310262680054),
# (-0.48415902256965637, 0.6002576351165771, -0.6273357272148132, 0.10830844193696976),
# (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 
# (0.872873842716217, 0.09946288913488388, -0.37463560700416565, -0.2963891923427582),
# (0.10778523236513138, -0.45753213763237, 0.838504433631897, -0.2756030261516571),
# (0.046796999871730804, -0.5665172338485718, 0.7200593948364258, -0.39797312021255493),
# (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
# (-0.48862341046333313, 0.4455340802669525, -0.5020152926445007, 0.5574290752410889),
# (0.742738664150238, 0.08381194621324539, 0.6633076667785645, 0.03657448664307594),
# (0.0, 0.0, 0.0, 0.0),
# (-0.4480212330818176, -0.4590829312801361, 0.5030765533447266, 0.5791664719581604),
# (-0.6885574460029602, 0.011318947188556194, 0.7243511080741882, 0.03280230611562729),
# (0.0, 0.0, 0.0, 0.0)
# ]

start_quat = [
(0.0, 0.0, 0.0, 0.0),
(-0.029328610748052597, 0.9989896416664124, 0.01571733132004738, 0.03020727075636387),
(-0.025912247598171234, 0.9986143112182617, 0.03488955646753311, 0.02967720665037632),
(-0.025067465379834175, 0.9980264902114868, 0.04981893301010132, -0.028858114033937454),
(-0.02783041261136532, 0.9948225021362305, 0.04832982271909714, -0.08495809882879257),
(0.6511525511741638, 0.7533252239227295, 0.08661968261003494, 0.031596630811691284),
(0.8774294257164001, -0.4794721007347107, -0.014965985901653767, -0.0001835153962019831),
(0.6066616773605347, 0.743905782699585, -0.15283969044685364, -0.23495957255363464),
(0.7191408276557922, -0.6806249022483826, 0.055296190083026886, -0.12856383621692657),
(0.7593467235565186, -0.10578184574842453, 0.5297819972038269, -0.36267590522766113),
(-0.23026873171329498, -0.3311282694339752, 0.3537304997444153, 0.8439223766326904),
(-0.10626760125160217, -0.3413722515106201, 0.18068058788776398, 0.916256844997406),
(0.0, 0.0, 0.0, 0.0),
(0.0, 0.0, 0.0, 0.0),
(-0.6074202656745911, 0.08470400422811508, 0.5681883692741394, 0.5486600399017334),
(-0.1759108304977417, 0.6156259179115295, -0.45499521493911743, 0.618901789188385),
(-0.02885306254029274, 0.6139870882034302, -0.350885272026062, 0.706446647644043),
(0.0, 0.0, 0.0, 0.0),
(0.0, 0.0, 0.0, 0.0),
(-0.4787684977054596, 0.4502279758453369, -0.509292483329773, 0.5556046962738037),
(0.7149807810783386, 0.06935683637857437, 0.6949256062507629, 0.03271733596920967),
(0.0, 0.0, 0.0, 0.0),
(-0.45334669947624207, -0.4538484811782837, 0.5092223882675171, 0.5737515687942505),
(0.7183882594108582, -0.021723121404647827, -0.6950241327285767, -0.019693393260240555),
(0.0, 0.0, 0.0, 0.0)
]

# start_quat = [[quat[1], quat[2], quat[3], quat[0]]for quat in start_quat]



files = ["./state_dict_model_outputlog_new_joint_quat_newtest.txt"]
for file in files:
	# print(file_csv[:-4])
	# time.sleep(3)

	# file_csv = "/08-12-22_14-58-38"+".csv" #.csv # in joint folder

	output_dir = "./output/change_quats/"
	if not (os.path.exists(output_dir)):
		os.makedirs(output_dir) # Create a new directory because it does not exist
	output_name = output_dir+file[:-4]+'.avi'
	# annotated_output_name = output_dir+now+'_annotate

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_name, fourcc, 30.0, (1920, 1800))

	# v1 = [0,1,0]
	# transformed_quat = []
	# canvas = np.zeros((1800, 1920, 3), np.uint8)
	# for quat in start_quat:
	# 	try:
	# 		quat = [float(quat[1]), -float(quat[2]), -float(quat[3]), float(quat[0])]
	# 		transformed_quat.append(transforms3d.quaternions.rotate_vector(v1, quat, is_normalized=True))
	# 	except:
	# 		print(quat)
	# 		print("error")
	# 		# print(np.array(transformed_quat))
	# 	# print((transformed_quat))


	# # joint_pos = np.array([[500,500]])
	# joint_pos = [(800,100)]
	# tf_quats = [(0.0, 0.0)]
	# # tf_quats = np.array([(0.0,0.0)],)
	# # print(len(transformed_quat))
	# for index in range(1,len(transformed_quat)):
	# 	# if not transformed_quat(index):
	# 	# 	continue
	# 	quat = transformed_quat[index]
	# 	# quat = (quat[0]*10*KINECT_JOINTS[index], quat[1]*10*KINECT_JOINTS[index])
	# 	quat = (quat[0]*50*KINECT_JOINTS[index], quat[1]*50*KINECT_JOINTS[index])

	# 	# quat = np.array((quat[0]*100*KINECT_JOINTS[index], quat[2]*100*KINECT_JOINTS[index]))
	# 	# print(quat)
	# 	tf_quats.append(quat)
	# 	# tf_quats= np.append(tf_quats, quat)
	# # print(len(tf_quats))

	# for bones in kinect_bones:
	# 	if len(joint_pos)==12 or len(joint_pos)==17:
	# 		joint_pos.append((0.0, 0.0)) # joint 12/17
	# 		joint_pos.append((0.0, 0.0)) # joint 13/18

	# 		# joint_pos = np.append(joint_pos,(0,0)) # joint 12/17
	# 		# joint_pos = np.append(joint_pos,(0,0)) # joint 13/18

	# 	if len(joint_pos)==21:
	# 		joint_pos.append((0.0, 0.0)) # joint 21

	# 		# joint_pos = np.append(joint_pos,(0,0)) # joint 21

	# 	joint1_index = bones[0]
	# 	joint2_index = bones[1]
	# 	current_pos = joint_pos[joint1_index]
	# 	tf_joint = tf_quats[joint2_index]

	# 	# print(type(tf_joint))
	# 	# print(tf_joint)
	# 	# print(type(current_pos))
	# 	next_pos = (tf_joint[0] + current_pos[0], tf_joint[1] + current_pos[1])
	# 	if bones[1]>4:
	# 		next_pos = (-tf_joint[0] + current_pos[0], -tf_joint[1] +current_pos[1])

	# 	joint_pos.append((next_pos))
	# 		# np.append(joint_pos, next_pos)
	# 		# print(current_pos)
	# 	# print(joint_pos)
	# 	# time.sleep(3)


	# idx = 0
	# for bones in kinect_bones:
	# 	start = (int(joint_pos[bones[0]][0]),int(joint_pos[bones[0]][1]))
	# 	end = (int(joint_pos[bones[1]][0]),int(joint_pos[bones[1]][1]))

	# 	cv2.line(canvas, start, end, kinect_bones_colour[idx], 8) 
	# 	idx+=1
	# cv2.imshow("skeleton_quaternions",canvas)
	# cv2.waitKey(0)




	quats = []
	quat = []
	with open(file) as quat_file:
		for line in quat_file:
			line = line.strip()
			if line[0]=='t':
				if (quat != []):
					quats.append(quat)
					# print(quats)
					# time.sleep(2)
				quat = []
				line = line[9:-1] # remove tensor([ and ending comma
			elif line[-2]==']':
				line = line[:-3] # remove ])
			elif line[0]=='d':
				continue
			else:
				line = line[:-1]
			line = line.split(",")
			digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
			for l in line:
				try:
					quat.append(float(l.strip()))
				except:
					# print(l)
					if l.strip()[0]=='d':
						continue
					if len(l)==0:
						continue
					else:
						while l[0] not in digits:
							l=l[1:]
							if len(l)==0:
								break
						if len(l)==0:
							break
						while l[-1] not in digits:
							l=l[:-1]
							if len(l)==0:
								break
						# l = l.strip()
						# l = l[:-2]
						quat.append(float(l.strip()))

			# line = [float(l.strip()) for l in line]
			# print(str(line))
			# time.sleep(2)
	print("done extracting quats")

	quat_file.close()

	
	print_counter = 0
	# print(len(quats))
	for quat_100 in quats:
		# print(str(print_counter))

		# print_counter +=1
		
		# transformed_quat = []
		# canvas = np.zeros((1800, 1920, 3), np.uint8)
		# if print_date==1:
		# 	font = cv2.FONT_HERSHEY_PLAIN
		# 	cv2.putText(canvas, row[0], (20, 40), font, 2, (255, 255, 255), 2)
	
		counter = 0
		body_quat = []
		# print(quat_100)
		for single in quat_100:
			# print(single)
			if counter%4==0:
				if counter>0:
					body_quat.append(single_quat)
				single_quat = []
				single_quat.append(single)
			elif counter%4 ==1:
				# single_quat.append(-1*single)
				single_quat.append(single)

			elif counter%4 ==2:
				# single_quat.append(-1*single)
				single_quat.append(single)

			else:
				single_quat.append(single)
			counter +=1
			# print(counter)
		body_quat.append(single_quat)
		# print(len(body_quat))
		# print(len(body_quat[0]))
		# print(len(body_quat[-1]))
		new_body_quat = []

		for y in range(24):
			w0, x0, y0, z0= start_quat[y]
			w1, x1, y1, z1 = body_quat[y]
			# w0, x0, y0, z0 = body_quat[y]
			# w1, x1, y1, z1 = start_quat[y]
			# modquat0 = math.sqrt(w0*w0 + x0*x0 + y0*y0 + z0*z0)

			# if modquat0 >0:
			# 	w0 = w0/modquat0
			# 	x0 = x0/modquat0
			# 	y0 = y0/modquat0
			# 	z0 = z0/modquat0
			# y0 = -y0
			# y1 = -y1
			# z0 = -z0
			# z1 = -z1
			
			quat = [-x1*x0 - y1*y0 - z1*z0 + w1*w0,
			 x1*w0 + y1*z0 - z1*y0 + w1*x0,
			-x1*z0 + y1*w0 + z1*x0 + w1*y0,
			 x1*y0 - y1*x0 + z1*w0 + w1*z0]

			# x0, y0, z0, w0 = quat
			# x1, y1, z1, w1 = [0, np.sin(np.pi/4), 0, np.cos(np.pi/4)]

			# quat = [-x1*x0 - y1*y0 - z1*z0 + w1*w0,
			#  x1*w0 + y1*z0 - z1*y0 + w1*x0,
			# -x1*z0 + y1*w0 + z1*x0 + w1*y0,
			#  x1*y0 - y1*x0 + z1*w0 + w1*z0]

			# x0, y0, z0, w0 = [0, -1*np.sin(np.pi/4), 0, np.cos(np.pi/4)]
			# x1, y1, z1, w1 = quat

			# quat = [-x1*x0 - y1*y0 - z1*z0 + w1*w0,
			#  x1*w0 + y1*z0 - z1*y0 + w1*x0,
			# -x1*z0 + y1*w0 + z1*x0 + w1*y0,
			#  x1*y0 - y1*x0 + z1*w0 + w1*z0]


			modquat0 = math.sqrt(quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3])

			if modquat0 >0:
				quat[0] = quat[0]/modquat0
				quat[1] = quat[1]/modquat0
				quat[2] = quat[2]/modquat0
				quat[3] = quat[3]/modquat0

			new_body_quat.append(quat)
		start_quat = new_body_quat
		# start_quat = [[quat[3], quat[0], -1*quat[1], -1*quat[2]] for quat in new_body_quat]





		# v1 = [0,1,0]

		v1 = [0, 1, 0]

		# print(body_quat)

		transformed_quat = []
		canvas = np.zeros((1800, 1920, 3), np.uint8)
		for quat in new_body_quat:
			try:
				quat = [float(quat[1]), -float(quat[2]), -float(quat[3]), float(quat[0])]
				transformed_quat.append(transforms3d.quaternions.rotate_vector(v1, quat, is_normalized=True))
			except:
				print(quat)
				print("error")
			# print(np.array(transformed_quat))
		# print((transformed_quat))


		# joint_pos = np.array([[500,500]])
		joint_pos = [(800,100)]
		tf_quats = [(0.0, 0.0)]
		# tf_quats = np.array([(0.0,0.0)],)
		# print(len(transformed_quat))
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
		cv2.imshow("skeleton_quaternions",canvas)
		cv2.waitKey(0)
		# print(len(canvas[0]))
		# out.write(canvas.astype('uint8'))
		# print("writing")

		# print(coor)
		# body_coor.append(coor)
		# row = str(row).split("(")[1:]
		# print(row)
		# time.sleep(2)
#         print(f'Column names are {", ".join(row)}')
	# line_count += 1
out.release()




		# csv_reader = csv.reader(quat_file, delimiter=',')
		# line_count = 0
		# for row in csv_reader:
		# 	# print(row)
		# 	# time.sleep(2)
		# 	if line_count > 0 and line_count%2==0:
		# 		# row = row.replace('"', "")
		# 		body_quat = []
		# 		canvas = np.zeros((1800, 1920, 3), np.uint8)

		# 		if print_date==1:
		# 			font = cv2.FONT_HERSHEY_PLAIN
		# 			cv2.putText(canvas, row[0], (20, 40), font, 2, (255, 255, 255), 2)

		# 		transformed_quat = []
				
		# 		for x in row[1:]:
		# 			# print(x)
		# 			x = x.replace(")", "")
		# 			x = x.replace("(", "")
		# 			x = x.split(", ")#[1:]
		# 			# quat = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
		# 			# quat = [i for i in x]
		# 			quat = [float(x[1]), -float(x[2]), -float(x[3]), float(x[0])]
		# 			body_quat.append(quat)

		# 		v1 = [0,1,0]
		# 		# v1 = [1,0,0]


		# 		for quat in body_quat:
		# 			transformed_quat.append(transforms3d.quaternions.rotate_vector(v1, quat, is_normalized=True))
		# 		# print(np.array(transformed_quat))

		# 		# joint_pos = np.array([[500,500]])
		# 		joint_pos = [(800,500)]
		# 		tf_quats = [(0.0, 0.0)]
		# 		# tf_quats = np.array([(0.0,0.0)],)
		# 		for index in range(1,len(transformed_quat)):
		# 			# if not transformed_quat(index):
		# 			# 	continue
		# 			quat = transformed_quat[index]
		# 			# quat = (quat[0]*10*KINECT_JOINTS[index], quat[1]*10*KINECT_JOINTS[index])
		# 			quat = (quat[0]*50*KINECT_JOINTS[index], quat[1]*50*KINECT_JOINTS[index])

		# 			# quat = np.array((quat[0]*100*KINECT_JOINTS[index], quat[2]*100*KINECT_JOINTS[index]))
		# 			# print(quat)
		# 			tf_quats.append(quat)
		# 			# tf_quats= np.append(tf_quats, quat)
		# 		# print(len(tf_quats))

		# 		for bones in kinect_bones:
		# 			if len(joint_pos)==12 or len(joint_pos)==17:
		# 				joint_pos.append((0.0, 0.0)) # joint 12/17
		# 				joint_pos.append((0.0, 0.0)) # joint 13/18

		# 				# joint_pos = np.append(joint_pos,(0,0)) # joint 12/17
		# 				# joint_pos = np.append(joint_pos,(0,0)) # joint 13/18

		# 			if len(joint_pos)==21:
		# 				joint_pos.append((0.0, 0.0)) # joint 21

		# 				# joint_pos = np.append(joint_pos,(0,0)) # joint 21

		# 			joint1_index = bones[0]
		# 			joint2_index = bones[1]
		# 			current_pos = joint_pos[joint1_index]
		# 			tf_joint = tf_quats[joint2_index]

		# 			# print(type(tf_joint))
		# 			# print(tf_joint)
		# 			# print(type(current_pos))
		# 			next_pos = (tf_joint[0] + current_pos[0], tf_joint[1] + current_pos[1])
		# 			if bones[1]>4:
		# 				next_pos = (-tf_joint[0] + current_pos[0], -tf_joint[1] +current_pos[1])

		# 			joint_pos.append((next_pos))
		# 			# np.append(joint_pos, next_pos)
		# 			# print(current_pos)
		# 		# print(joint_pos)
		# 		# time.sleep(3)


		# 		idx = 0
		# 		for bones in kinect_bones:
		# 			start = (int(joint_pos[bones[0]][0]),int(joint_pos[bones[0]][1]))
		# 			end = (int(joint_pos[bones[1]][0]),int(joint_pos[bones[1]][1]))

		# 			cv2.line(canvas, start, end, kinect_bones_colour[idx], 8) 
		# 			idx+=1
		# 		# cv2.imshow("skeleton_quaternions",canvas)
		# 		# cv2.waitKey(0)
		# 		# print(len(canvas[0]))
		# 		out.write(canvas.astype('uint8'))

		# 		# print(coor)
		# 		# body_coor.append(coor)
		# 		# row = str(row).split("(")[1:]
		# 		# print(row)
		# 		# time.sleep(2)
		# #         print(f'Column names are {", ".join(row)}')
		# 	line_count += 1
		# out.release()
		# #     else:
		# #         print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
		# #         line_count += 1
		# # print(f'Processed {line_count} lines.')