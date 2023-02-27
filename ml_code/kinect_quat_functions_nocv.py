## read 
## use conda env pytorchpy3.9

# import csv
import time
import numpy as np
# import cv2
# import os
from scipy.spatial.transform import Rotation
# from scipy.stats import linregress
import pandas as pd
# import re
# import torch
import math


# a = [ 1.5398e-03,  7.8785e-01, -2.4952e-02,  6.1535e-01]
# b = [-1.3489e-02,  7.8481e-01, -2.7265e-02,  6.1899e-01]

# r = Rotation.from_quat(a)
# ref_r = Rotation.from_quat(b)

# new_rot = r*ref_r.inv()
# new_quat = new_rot.as_quat()
# print(new_quat[0]+2, new_quat[1]+2, new_quat[2]+2, new_quat[3]+2)

##### function to transform absolute quaternion (x, y, z, w) to relative quaternion (deltaR)
##### deltaR*q0 = q1, deltaR = q1*inv(q0)
##### inputs: q0 (25*4 quat to be used at reference starting point), quat_array (n*25*4 quats to be transformed recursively)
##### output: rel_quat ([n-1]*25*4 relative quat array)

def abs2relquat (q0, quat_array):
	ref_quats = q0
	relative_quats = []
	for quats in quat_array:
		temp_quats = []
		for q in range(25):
			ref_quat = ref_quats[q]
			quat = quats[q]
			try:
				r = Rotation.from_quat(quat)
				ref_r = Rotation.from_quat(ref_quat)

				new_rot = r*ref_r.inv()
				new_quat = new_rot.as_quat()
				temp_quats.append(new_quat)
			except:
				temp_quats.append([0,0,0,0])
		ref_quats = quats
		relative_quats.append(temp_quats)
	return relative_quats

##### function to transform relative quaternion (deltaR) to absolute quaternion q1 (x, y, z, w)
##### deltaR*q0 = q1, deltaR = q1*inv(q0)
##### inputs: q0 (25*4 quat to be used at reference starting point), quat_array (n*25*4 deltaR quats to be transformed recursively)
##### output: abs_quat ([n-1]*25*4 relative quat array)

def rel2absquat (q0, quat_array):
	ref_quats = q0
	absolute_quats = []
	for quats in quat_array:
		temp_quats = []
		for q in range(25):
			ref_quat = ref_quats[q]
			quat = quats[q]
			sq_sum = math.sqrt(quat[0]*quat[0]+quat[1]*quat[1]+quat[2]*quat[2]+quat[3]*quat[3])
			if sq_sum>0:
				quat = [quat[0]/sq_sum, quat[1]/sq_sum, quat[2]/sq_sum, quat[3]/sq_sum]
			try:
				r = Rotation.from_quat(quat)
				ref_r = Rotation.from_quat(ref_quat)
				new_rot = r*ref_r
				new_quat = new_rot.as_quat()
				temp_quats.append(new_quat)
			except:
				# print(ref_quat)
				# print(quat)
				# time.sleep(10)
				temp_quats.append([0,0,0,0])
		ref_quats = temp_quats
		absolute_quats.append(temp_quats)
	return absolute_quats

