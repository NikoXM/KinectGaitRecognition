#this program calculate the 3D coordinate data into length
# and save the file as the weka format
import os
import string
import shutil
import numpy as np
import GaitData as gd

limb_descriptors = ['neck_len','rshoulder_len','lshoulder_len','rarm_len','larm_len','rforearm_len','lforearm_len',
					'rhand_len','lhand_len','upper_spine','lower_spine','rhip_len','lhip_len','rthigh_len','lthigh_len',
					'rcalf_len','lcalf_len','rfoot_len','lfoot_len','height']

class StaticAnalyzer:
	def __init__(self,srcTrainPath,dstTrainPath,srcTestPath,dstTestPath):
		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.datas = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.gaitData = gd.GaitData()
		self.srcTrainPath = srcTrainPath
		self.dstTrainPath = dstTrainPath
		self.srcTestPath = srcTestPath
		self.dstTestPath = dstTestPath

		# self.fileDistance = open(wekaPath+"/static.arff",'w')
		#self.fileDistance.write("@relation static-identification\n")
		#self.fileDistance.write("@attribute neck_len numeric\n")
		# for limb in limb_descriptors:
		# 	self.fileDistance.write("@attribute ")
		# 	self.fileDistance.write(limb)
		# 	self.fileDistance.write(" numeric\n")
		# self.fileDistance.write("@attribute identification numeric\n")
		# self.fileDistance.write("@data\n")

		if(os.path.exists(self.dstTrainPath)):
			shutil.rmtree(self.dstTrainPath)
			os.mkdir(self.dstTrainPath)
		else:
			os.mkdir(self.dstTrainPath)
		if(os.path.exists(self.dstTestPath)):
			shutil.rmtree(self.dstTestPath)
			os.mkdir(self.dstTestPath)
		else:
			os.mkdir(self.dstTestPath)

	#def __del__(self):
		# self.fileDistance.close()
		#self.trainData.close()

	def euclidis (self,joint_1, joint_2):
		dif = (joint_1-joint_2)
		dis = dif*dif
		summary = dis[0:,0] + dis[0:,1] + dis[0:,2]
		return summary**0.5

	def listdir_nohidden(self,path):
		for f in os.listdir(path):
			if not f.startswith('.'):
				yield f

	def clear(self):
		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

	def read_data(self,personFilePath):
		person = open(personFilePath)
		personData = person.readlines()
		if len(personData) == 0:
			print "The data of file is empty:"
			print personFilePath
			return
		length = len(personData)
		for item in range(0,length/20):
			for seg in range(0,20):
				temp = personData[item*20 + seg].split(";")
				point = [string.atof(temp[1]),string.atof(temp[2]),string.atof(temp[3].replace("\n",''))]
				self.points[seg].append(point)
		person.close()

	#this function is to caculate mean and std of raw data
	#and then filter raw data and write
	def filter_mean_std(self):
		means = np.zeros(20)
		std = np.zeros(20)
		# pre-caculate mean and std of every limb
		lens = len(self.datas[0])
		for i in range(20):
			means[i] = self.datas[i].sum()/lens
			std[i] = np.std(self.datas[i])
		for i in range(20):
			j = 0
			while(j < len(self.datas[i])):
				if (self.datas[i][j] < means[i] - 2*std[i]) or (self.datas[i][j] > means[i] + 2*std[i]):
					for k in range(20):
						self.datas[k] = np.delete(self.datas[k],j,0)
				j += 1
		for i in range(0,20):
			self.trainData.write(str(self.datas[i].sum()/lens)+',')
			#self.fileDistance.write(str(self.datas[i].sum()/lens)+',')
		self.trainData.write(str(self.gaitData.getId())+'\n')
		#self.fileDistance.write(str(self.gaitData.getId())+'\n')

	def data_process(self):
		self.mode = "Train Static:"
		self.trainData = open(self.dstTrainPath+"/static.txt",'w')
		self.static_analysis(self.srcTrainPath,self.dstTrainPath)
		self.trainData.close()

		self.mode = "Test  Static:"
		self.trainData = open(self.dstTestPath+"/static.txt",'w')
		self.static_analysis(self.srcTestPath,self.dstTestPath)
		self.trainData.close()

	def static_analysis(self,srcPath,dstPath):
		#The first step is to read data from files
		personDirectorsPath = srcPath
		personDirectors = self.listdir_nohidden(personDirectorsPath)
		for personDirector in personDirectors:
			personDirectorPath = personDirectorsPath + '/' + personDirector
			if not os.path.isdir(personDirectorPath):
				continue
			print self.mode,personDirector
			personFiles = self.listdir_nohidden(personDirectorPath)
			#caculate the number of nth file
			writePersonDirectorPath = dstPath +'/' + personDirector + '/'
			if(os.path.exists(writePersonDirectorPath)):
				shutil.rmtree(writePersonDirectorPath)
				os.mkdir(writePersonDirectorPath)
			else:
				os.mkdir(writePersonDirectorPath)

			for personFile in personFiles:
				personFilePath = personDirectorPath + '/' + personFile
				self.clear()
				self.read_data(personFilePath)
				self.gaitData.setId(int(personDirector.replace("Person",'')))
		#The second step is to deal with the data
				self.calculateLength()
				self.filter_mean_std()
		#The third step is to write data into file
				self.write_data(writePersonDirectorPath+personFile)

	#this function write down all data
	def write_data(self,dstPersonFile):
		dstFile = open(dstPersonFile,'w')
		length = len(self.datas[0])
		for line in range(0,length):
			for i in range(0,20):
				dstFile.write(limb_descriptors[i]+',')
				dstFile.write(str(self.datas[i][line])+'\n')
			dstFile.write('\n')
		dstFile.close()

	def calculateLength(self):
		head = np.array(self.points[0])
		shoulder_center = np.array(self.points[1])
		shoulder_right = np.array(self.points[2])
		shoulder_left = np.array(self.points[3])
		elbow_right = np.array(self.points[4])
		elbow_left = np.array(self.points[5])
		wrist_right = np.array(self.points[6])
		wrist_left = np.array(self.points[7])
		hand_right = np.array(self.points[8])
		hand_left = np.array(self.points[9])
		spine = np.array(self.points[10])
		hip_center = np.array(self.points[11])
		hip_right = np.array(self.points[12])
		hip_left = np.array(self.points[13])
		knee_right = np.array(self.points[14])
		knee_left = np.array(self.points[15])
		ankle_right = np.array(self.points[16])
		ankle_left = np.array(self.points[17])
		foot_right = np.array(self.points[18])
		foot_left = np.array(self.points[19])

		#1.neck
		neck = self.euclidis(head,shoulder_center)
		self.datas[0] = self.euclidis(head,shoulder_center)#.tolist()
		#2.right-shoulder
		right_shoulder = self.euclidis(shoulder_center,shoulder_right)
		self.datas[1] = self.euclidis(shoulder_center,shoulder_right)#.tolist()
		#3.left-shoulder
		left_shoulder = self.euclidis(shoulder_center,shoulder_left)
		self.datas[2] = self.euclidis(shoulder_center,shoulder_left)#.tolist()
		#4.right-arm
		right_arm = self.euclidis(shoulder_right,elbow_right)
		self.datas[3] = self.euclidis(shoulder_right,elbow_right)#.tolist()
		#5.left-arm
		left_arm = self.euclidis(shoulder_left,elbow_left)
		self.datas[4] = self.euclidis(shoulder_left,elbow_left)#.tolist()
		#6.right-front-arm
		right_front_arm = self.euclidis(elbow_right,wrist_right)
		self.datas[5] = self.euclidis(elbow_right,wrist_right)#.tolist()
		#7.left-front-arm
		left_front_arm = self.euclidis(elbow_left,wrist_left)
		self.datas[6] = self.euclidis(elbow_left,wrist_left)#.tolist()
		#8.right-hand
		right_hand = self.euclidis(wrist_right,hand_right)
		self.datas[7] = self.euclidis(wrist_right,hand_right)#.tolist()
		#9.left-hand
		left_hand = self.euclidis(wrist_left,hand_left)
		self.datas[8] = self.euclidis(wrist_left,hand_left)#.tolist()
		#10.upper-spine
		upper_spine = self.euclidis(shoulder_center,spine)
		self.datas[9] = self.euclidis(shoulder_center,spine)#.tolist()
		#11.lower-spine
		lower_spine = self.euclidis(spine,hip_center)
		self.datas[10] = self.euclidis(spine,hip_center)#.tolist()
		#12.right-hip
		right_hip = self.euclidis(hip_center,hip_right)
		self.datas[11] = self.euclidis(hip_center,hip_right)#.tolist()
		#13.left-hip
		left_hip = self.euclidis(hip_center,hip_left)
		self.datas[12] = self.euclidis(hip_center,hip_left)#.tolist()
		#14.right-thigh
		right_thigh = self.euclidis(hip_right,knee_right)
		self.datas[13] = self.euclidis(hip_right,knee_right)#.tolist()
		#15.left-thigh
		left_thigh = self.euclidis(hip_left,knee_left)
		self.datas[14] = self.euclidis(hip_left,knee_left)#.tolist()
		#16.right-leg
		right_leg = self.euclidis(knee_right,ankle_right)
		self.datas[15] = self.euclidis(knee_right,ankle_right)#.tolist()
		#17.left-leg
		left_leg = self.euclidis(knee_left,ankle_left)
		self.datas[16] = self.euclidis(knee_left,ankle_left)#.tolist()
		#18.right-foot
		right_foot = self.euclidis(ankle_right,foot_right)
		self.datas[17] = self.euclidis(ankle_right,foot_right)#.tolist()
		#19.left-foot
		left_foot = self.euclidis(ankle_left,foot_left)
		self.datas[18] = self.euclidis(ankle_left,foot_left)#.tolist()
		#20.height
		height = neck + upper_spine + lower_spine + (right_hip + left_hip)/2 + (right_thigh+left_thigh)/2 + (right_leg+left_leg)/2 + (right_foot+left_foot)/2
		self.datas[19] = (neck + upper_spine + lower_spine + (right_hip + left_hip)/2 + (right_thigh+left_thigh)/2 + (right_leg+left_leg)/2 + (right_foot+left_foot)/2)#.tolist()

#The data path contain converted data
srcPath = [
		#0
		"/Users/niko/Documents/KinectGaitScripts/Data/FilteredGaitDataset",
		#1
		"/Users/niko/Documents/KinectGaitScripts/Data/RawGaitDataset",
		#2
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/ConvertedData",
		#3
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/FilteredGaitDataset",
		#4
		"/Users/niko/Documents/KinectGaitScripts/TrainDataset/TrainGaitDataset",
		#5
		"/Users/niko/Documents/KinectGaitScripts/TestDataset/TestGaitDataset"]

dstPath = [
		#0
		"/Users/niko/Documents/KinectGaitScripts/Data/ConvertedDataset",
		#1
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/ConvertedDataset",
		#2
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/WekaDataset",
		#3
		"/Users/niko/Documents/KinectGaitScripts/TrainDataset/TrainStaticDataset",
		#4
		"/Users/niko/Documents/KinectGaitScripts/TestDataset/TestStaticDataset"]

if __name__=="__main__":
	#train data
	sa = StaticAnalyzer(srcPath[4],dstPath[3],srcPath[5],dstPath[4])
	#test data
	#test_sa = StaticAnalyzer(srcPath[5],dstPath[4])
	sa.data_process()
