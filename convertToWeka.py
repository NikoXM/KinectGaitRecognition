#this program calculate the 3D coordinate data into length
# and save the file as the weka format
import os
import numpy as np
import string

def euclidis (joint_1, joint_2):
	dif = (joint_1-joint_2)
	dis = dif*dif
	summary = dis[0:,0] + dis[0:,1] + dis[0:,2]
	return summary**0.5

def listdir_nohidden(path):
	for f in os.listdir(path):
		if not f.startswith('.'):
			yield f

#this function is to caculate mean and std of raw data
#and then filter raw data 
def filter_mean_std(datas):
	means = np.zeros(20)
	std = np.zeros(20)
	# pre-caculate mean and std of every limb
	lens = len(datas[0])

	for i in range(20):
		means[i] = datas[i].sum()/lens
		#std[i] = ((datas[i]*datas[i]).sum()/lens - means[i]**2)
		std[i] = np.std(datas[i])
	#print means[0]
	#print std[0]
	for i in range(20):
		j = 0
		while(j < len(datas[i])):
			if (datas[i][j] < means[i] - 2*std[i]) or (datas[i][j] > means[i] + 2*std[i]):
				for k in range(20):
					datas[k] = np.delete(datas[k],j,0)
			j += 1
	return datas

def initial_file(path):
	fileDistance = open(path,'w')
	fileDistance.write("@relation static-identification\n")
	fileDistance.write("@attribute neck_len numeric\n")
	fileDistance.write("@attribute rshoulder_len numeric\n")
	fileDistance.write("@attribute lshoulder_len numeric\n")
	fileDistance.write("@attribute rarm_len numeric\n")
	fileDistance.write("@attribute larm_len numeric\n")
	fileDistance.write("@attribute rforearm_len numeric\n")
	fileDistance.write("@attribute lforearm_len numeric\n")
	fileDistance.write("@attribute rhand_len numeric\n")
	fileDistance.write("@attribute lhand_len numeric\n")
	fileDistance.write("@attribute upper_spine numeric\n")
	fileDistance.write("@attribute lower_spine numeric\n")
	fileDistance.write("@attribute rhip_len numeric\n")
	fileDistance.write("@attribute lhip_len numeric\n")
	fileDistance.write("@attribute rthigh_len numeric\n")
	fileDistance.write("@attribute lthigh_len numeric\n")
	fileDistance.write("@attribute rcalf_len numeric\n")
	fileDistance.write("@attribute lcalf_len numeric\n")
	fileDistance.write("@attribute rfoot_len numeric\n")
	fileDistance.write("@attribute lfoot_len numeric\n")
	fileDistance.write("@attribute height numeric\n")
	fileDistance.write("@attribute identification numeric\n")
	fileDistance.write("@data\n")
	fileDistance.close()

#this function write data into the weka files
def write_weka_file(datas,length,path,person):
	fileDistance = open(path,'a')
	#next operation is to write all the filtered data into weka files
	'''
	for line in range(0,length):
		for i in range(0,20):
			fileDistance.write(str(datas[i][line])+',')
		fileDistance.write(str(string.atoi(person.replace("Person",''))))
		fileDistance.write('\n')
	fileDistance.close()
	'''
	#from writing every frame of data to writeing mean data
	lens = len(datas[0])
	for i in range(0,20):
		fileDistance.write(str(datas[i].sum()/lens)+',')
	fileDistance.write(person+'\n')
	fileDistance.close()
'''
#take the raw data into weka data
personFilePath = "/Users/niko/Documents/KinectGaitScripts/KinectGaitRawDataset"
#produce the distance file from filtered data
fileDistancePath = "/Users/niko/Documents/KinectGaitScripts/Data/Distance/rawDistance.arff"
'''

#take the filtered data into weka data
personFilePath = "/Users/niko/Documents/KinectGaitScripts/FilteredGaitDataset"
#produce the distance file from raw data
fileDistancePath = "/Users/niko/Documents/KinectGaitScripts/Data/Distance/filteredDistance.arff"

'''
#take the tested data into list
personFilePath = "/Users/niko/Documents/KinectGaitScripts/TestOnlyData"
#produce the distance file from test data
fileDistancePath = "/Users/niko/Documents/KinectGaitScripts/Data/Distance/testDistance.arff"
'''
personFiles = listdir_nohidden(personFilePath)

initial_file(fileDistancePath)

#set the test file into limit
limit = 20
p_limit = 0
#next one sentence is to operate all the files
for person in personFiles:
	if(p_limit == limit):
		break
	person = person.replace("'",'')
	print(person)
	p_limit += 1
	dataFilePath = personFilePath + "/" + person
	dataFiles = listdir_nohidden(dataFilePath)
	for file in dataFiles:
		file = file.replace("'",'')
		#if(cmp(file,".DS_Store") == 0):
		#	continue
		fileABSPath = dataFilePath + "/" + file
		#the next operation is to caculate the length of all part
		fileData = open(fileABSPath,'r')
		data = fileData.readlines()	
		fileData.close()

		points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		
		length = len(data)
		for item in range(0,length/20):
			for seg in range(0,20):
				temp = data[item*20 + seg].split(";")
		#need to append different kind of data into different lists
				point = [string.atof(temp[1]),string.atof(temp[2]),string.atof(temp[3].replace("\n",''))]
				points[seg].append(point)

		head = np.array(points[0])
		shoulder_center = np.array(points[1])
		shoulder_right = np.array(points[2])
		shoulder_left = np.array(points[3])
		elbow_right = np.array(points[4])
		elbow_left = np.array(points[5])
		wrist_right = np.array(points[6])
		wrist_left = np.array(points[7])
		hand_right = np.array(points[8])
		hand_left = np.array(points[9])
		spine = np.array(points[10])
		hip_center = np.array(points[11])
		hip_right = np.array(points[12])
		hip_left = np.array(points[13])
		knee_right = np.array(points[14])
		knee_left = np.array(points[15])
		ankle_right = np.array(points[16])
		ankle_left = np.array(points[17])
		foot_right = np.array(points[18])
		foot_left = np.array(points[19])

		datas = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

		#1.neck
		neck = euclidis(head,shoulder_center)
		datas[0] = euclidis(head,shoulder_center)#.tolist()
		#2.right-shoulder
		right_shoulder = euclidis(shoulder_center,shoulder_right)
		datas[1] = euclidis(shoulder_center,shoulder_right)#.tolist()
		#3.left-shoulder
		left_shoulder = euclidis(shoulder_center,shoulder_left)
		datas[2] = euclidis(shoulder_center,shoulder_left)#.tolist()
		#4.right-arm
		right_arm = euclidis(shoulder_right,elbow_right)
		datas[3] = euclidis(shoulder_right,elbow_right)#.tolist()
		#5.left-arm
		left_arm = euclidis(shoulder_left,elbow_left)
		datas[4] = euclidis(shoulder_left,elbow_left)#.tolist()
		#6.right-front-arm
		right_front_arm = euclidis(elbow_right,wrist_right)
		datas[5] = euclidis(elbow_right,wrist_right)#.tolist()
		#7.left-front-arm
		left_front_arm = euclidis(elbow_left,wrist_left)
		datas[6] = euclidis(elbow_left,wrist_left)#.tolist()
		#8.right-hand
		right_hand = euclidis(wrist_right,hand_right)
		datas[7] = euclidis(wrist_right,hand_right)#.tolist()
		#9.left-hand
		left_hand = euclidis(wrist_left,hand_left)
		datas[8] = euclidis(wrist_left,hand_left)#.tolist()
		#10.upper-spine
		upper_spine = euclidis(shoulder_center,spine)
		datas[9] = euclidis(shoulder_center,spine)#.tolist()
		#11.lower-spine
		lower_spine = euclidis(spine,hip_center)
		datas[10] = euclidis(spine,hip_center)#.tolist()
		#12.right-hip
		right_hip = euclidis(hip_center,hip_right)
		datas[11] = euclidis(hip_center,hip_right)#.tolist()
		#13.left-hip
		left_hip = euclidis(hip_center,hip_left)
		datas[12] = euclidis(hip_center,hip_left)#.tolist()
		#14.right-thigh
		right_thigh = euclidis(hip_right,knee_right)
		datas[13] = euclidis(hip_right,knee_right)#.tolist()
		#15.left-thigh
		left_thigh = euclidis(hip_left,knee_left)
		datas[14] = euclidis(hip_left,knee_left)#.tolist()
		#16.right-leg
		right_leg = euclidis(knee_right,ankle_right)
		datas[15] = euclidis(knee_right,ankle_right)#.tolist()
		#17.left-leg
		left_leg = euclidis(knee_left,ankle_left)
		datas[16] = euclidis(knee_left,ankle_left)#.tolist()
		#18.right-foot
		right_foot = euclidis(ankle_right,foot_right)
		datas[17] = euclidis(ankle_right,foot_right)#.tolist()
		#19.left-foot
		left_foot = euclidis(ankle_left,foot_left)
		datas[18] = euclidis(ankle_left,foot_left)#.tolist()
		#20.height
		height = neck + upper_spine + lower_spine + (right_hip + left_hip)/2 + (right_thigh+left_thigh)/2 + (right_leg+left_leg)/2 + (right_foot+left_foot)/2
		datas[19] = (neck + upper_spine + lower_spine + (right_hip + left_hip)/2 + (right_thigh+left_thigh)/2 + (right_leg+left_leg)/2 + (right_foot+left_foot)/2)#.tolist()
		#print(datas[0])

		#print("the raw size is,",len(datas[0]))
		datas = filter_mean_std(datas)
		#print("the filtered size is",len(datas[0]))
		write_weka_file(datas,len(datas[0]),fileDistancePath,str(string.atoi(person.replace("Person",''))))
		#print(datas[0])


#next operation is to write the recaculated mean and std and then label the idendification


