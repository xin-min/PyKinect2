## read 
## use conda env kinectpy3.6

import csv
import time
import numpy as np         
import cv2
import transforms3d
import os
import math
from kinect_quat_functions import quat2vector, vector2screen
from kinect_quat_functions_nocv import abs2relquat, rel2absquat


# files = ["./state_dict_model_outputlog_change_quat_100.txt"]
files_ref = "./quat_lstm/0303_lstm_punch1_ref.txt"
files_ground = "./quat_lstm/0303_lstm_punch1_ground.txt"
files = ["./quat_lstm/0303_lstm_punch1_model.txt"]

for file in files:
	# print(file_csv[:-4])
	# time.sleep(3)

	# file_csv = "/08-12-22_14-58-38"+".csv" #.csv # in joint folder

	frame_idx = 10
	output_dir = ""
	record = 0
	if record ==1:
		if not (os.path.exists(output_dir)):
			os.makedirs(output_dir) # Create a new directory because it does not exist
		output_name = output_dir+file[:-4]++str(frame_idx)+'.avi'
		# annotated_output_name = output_dir+now+'_annotate

		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		out = cv2.VideoWriter(output_name, fourcc, 5.0, (1920, 1800))

	quats = []
	quat = []
	with open(files_ref) as ref_file:
		for line in ref_file:
			line = line.strip()
			if not line:
				continue
			if line[0]=='t':
				if (quat != []):
					quats.append(quat)      
					# print(quats)
					# time.sleep(2)
				quat = []
				line = line[9:-1] # remove tensor([ and ending comma
			elif line[-2]==']':
				line = line[:-2] # remove ])
			elif line[0]=='d':
				continue
			else:
				line = line[:-1]
			line = line.split(",")
			digits = ['-','0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
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
	quats.append(quat)


	print("done extracting ref quats")

	ref_file.close()
	# print(quats[0][:10])
	ref_quats = quats

	quats = []
	quat = []
	with open(files_ground) as quat_file:
		for line in quat_file:
			line = line.strip()
			# print(line)
			# time.sleep(1)
			if not line:
				continue
			# print(line[0])
			if line[0]=='t':
				if (quat != []):
					quats.append(quat)
					# print(quats)
					# print("appended")
					# print(quats)
					# time.sleep(2)
				# else:
				# 	# print("skip")
				quat = []
				line = line[9:-1] # remove tensor([ and ending comma
				# print(line)
			elif line[0]=='[':
				# if (quat != []):
				quats.append(quat)
				quat = []
			elif line[0]=='d':
				continue
			# else:
			# 	line = line[:-1]
			line = line.split(",")
			digits = ['-','0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
			for l in line:
				if l:
					try:
						quat.append(float(l.strip())-2)
						# print("appended")
					except:
						# print("help")
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
						quat.append(float(l.strip())-2)					
	quats.append(quat)
	# print(len(quats))
	# print(len(quats[0]))

			# print(quat)

	print("done extracting ground quats")

	quat_file.close()

	ground_quats = quats


	# error

	quats = []
	quat = []
	with open(file) as quat_file:
		for line in quat_file:
			line = line.strip()
			# print(line)
			# time.sleep(1)
			if not line:
				continue
			# print(line[0])
			if line[0]=='t':
				if (quat != []):
					quats.append(quat)
					# print(quats)
					# print("appended")
					# print(quats)
					# time.sleep(2)
				# else:
				# 	# print("skip")
				quat = []
				line = line[9:-1] # remove tensor([ and ending comma
				# print(line)
			elif line[0]=='[':
				# if (quat != []):
				quats.append(quat)
				quat = []
			elif line[0]=='d':
				continue
			# else:
			# 	line = line[:-1]
			line = line.split(",")
			digits = ['-','0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
			for l in line:
				if l:
					try:
						quat.append(float(l.strip())-2)
						# print("appended")
					except:
						# print("help")
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
						quat.append(float(l.strip())-2)					
	quats.append(quat)
	# print(len(quats))
	# print(len(quats[0]))

			# print(quat)

	print("done extracting quats")
	# print(quats)

	quat_file.close()

	######################## reference file
	print_counter = 0
	quat_array = []

	for quat_100 in ref_quats:
		counter = 0
		body_quat = []
		for single in quat_100:
			if counter%4==0:
				if counter>0:
					body_quat.append(single_quat)
				single_quat = []
				single_quat.append(single)

			else:
				single_quat.append(single)
			counter +=1
			# print(counter)
		body_quat.append(single_quat)    
		quat_array.append(body_quat)

	counter = 0
	new_quat_array = []

	for q in quat_array:
		for quat in q:
			if counter%25==0:
				if counter>0:
					new_quat_array.append(single_body)
				single_body = []
				single_body.append(quat)

			else:
				single_body.append(quat)
			counter+=1
		new_quat_array.append(single_body)

	ref_quat_array = new_quat_array

	########################### ground quat file

	new_quat_array = []
	quat_array = []

	for quat_100 in ground_quats:
		counter = 0
		body_quat = []
		for single in quat_100:
			if counter%4==0:
				if counter>0:
					body_quat.append(single_quat)
				single_quat = []
				single_quat.append(single)

			else:
				single_quat.append(single)
			counter +=1
			# print(counter)
		body_quat.append(single_quat)
		quat_array.append(body_quat)
	ground_quat_array = quat_array


	########################### quat file

	new_quat_array = []
	quat_array = []

	for quat_100 in quats:
		counter = 0
		body_quat = []
		for single in quat_100:
			if counter%4==0:
				if counter>0:
					body_quat.append(single_quat)
				single_quat = []
				single_quat.append(single)

			else:
				single_quat.append(single)
			counter +=1
			# print(counter)
		body_quat.append(single_quat)
		quat_array.append(body_quat)
	new_quat_array = quat_array
	
	# for i in reversed(range(1,len(ground_quat_array)-1)):
	# 	remove = 0
	# 	for j in range(len(ground_quat_array[0])):
	# 		# for k in range(len(ground_quat_array[0][0])):
	# 			if (abs(ground_quat_array[i][j][k])-abs(ground_quat_array[i-1][j][k]))>0.4:
	# 				if (abs(ground_quat_array[i][j][k])-abs(ground_quat_array[i+1][j][k]))>0.4:
	# 				# print(ground_quat_array[i][j][k])
	# 				# print(ground_quat_array[i-1][j][k])
	# 				# time.sleep(1)
	# 					remove = 1
	# 	if remove ==1:
	# 		print("removed")
	# 		new_quat_array.pop(i)
	# 		ground_quat_array.pop(i)
	# 		ref_quat_array.pop(i)

	ground_quats = []
	ground_quats.append(ref_quat_array[0])
	last_i = 0
	# print(len(new_quat_array))
	for i in range(len(ref_quat_array)):
		if i%frame_idx==0:
			if i+frame_idx<(len(ground_quat_array)):
				abs_quats = rel2absquat(ref_quat_array[i], ground_quat_array[i:i+frame_idx])
				abs_quats = [[[b[0],b[1],b[2],b[3]] for b in a] for a in abs_quats]
				# print(abs_quats)
				for q in abs_quats:
					ground_quats.append(q)
				last_i = i

	if last_i ==0 or not last_i%frame_idx==0:
		abs_quats = rel2absquat(ref_quat_array[last_i], ground_quat_array[last_i:])
		abs_quats = [[[b[0],b[1],b[2],b[3]] for b in a] for a in abs_quats]
		
		for q in abs_quats:
			ground_quats.append(q)

	quats = []
	quats.append(ref_quat_array[0])
	last_i = 0
	# print(len(new_quat_array))
	for i in range(len(ref_quat_array)):
		if i%frame_idx==0:
			if i+frame_idx<(len(new_quat_array)):
				abs_quats = rel2absquat(ref_quat_array[i], new_quat_array[i:i+frame_idx])
				abs_quats = [[[b[0],b[1],b[2],b[3]] for b in a] for a in abs_quats]
				# print(abs_quats)
				for q in abs_quats:
					quats.append(q)
				last_i = i

	if last_i ==0 or not last_i%frame_idx==0:
		abs_quats = rel2absquat(ref_quat_array[last_i], new_quat_array[last_i:])
		abs_quats = [[[b[0],b[1],b[2],b[3]] for b in a] for a in abs_quats]

		for q in abs_quats:
			quats.append(q)




		### chekc for if anny value is >0.4 diff from the prev one, remove all quats of that index from all 3 vector arrays


	counter =0

	vector_arrays = []
	for q in quats:
		vector_arrays.append(quat2vector(q, cameraspace = False))

	vector_arrays_ground = []
	for q in ground_quats:
		vector_arrays_ground.append(quat2vector(q, cameraspace = False))

	vector_arrays_ref = []
	for q in ref_quat_array:
		vector_arrays_ref.append(quat2vector(q, cameraspace = False))



	############# first frame (original)
	original_vector = vector_arrays[0]
	canvas = np.zeros((1800, 1920, 3), np.uint8)
	canvas = vector2screen (original_vector, 50, (800,100, 800), canvas)
	if record ==0:
		cv2.imshow("skeleton_quaternions",canvas)
		cv2.waitKey(0)
	elif record==1:
		out.write(canvas.astype('uint8'))

	############# rest of the frames (estimated and ground truth)
	for i in range(1,len(vector_arrays)):
		vector_array = vector_arrays[i]
		vector_array_ground = vector_arrays_ground[i]
		# vector_array_ground = vector_arrays_ref[i]



		canvas = np.zeros((1800, 1920, 3), np.uint8)
		canvas = vector2screen (vector_array_ground, 50, (800,100, 800), canvas, ground=1)
		canvas = vector2screen (vector_array, 50, (800,100, 800), canvas)
		print(i)

		if record ==0:
			cv2.imshow("skeleton_quaternions",canvas)
			cv2.waitKey(0)
		elif record==1:
			out.write(canvas.astype('uint8'))
	if record==1:
		out.release()

