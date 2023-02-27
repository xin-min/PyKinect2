import time
f = open("velocity_list.txt", "r")

digit = ['1','2','3','4','5','6','7','8','9','0','-']
lines = f.readlines()
for line in lines:
	line = line.split()
	coordinates = []
	for x in line[1:]:
		while x[0] not in digit:
			x = x[1:]
		while x[-1] not in digit:
			x = x[:-1]
		coordinates.append(float(x))
	for x in coordinates:
		if x>3 or x<-3:
			print(line[0]+str(coordinates))
			time.sleep(2)
			continue

	# print(line[0][:-1])
	# time.sleep(2)
time_diff = 1/20 #20fps
initial_pose = [] #25joints*3XYZcoor
vel_list = [] #25joints*3XYZcoor
new_pose = [] #25joints*3XYZcoor

for i in range(25):
	new_pose.append( [initial_pose[i][0]*vel_list[i][0]*time_diff,
		initial_pose[i][1]*vel_list[i][1]*time_diff,
		initial_pose[i][2]*vel_list[i][2]*time_diff
		])