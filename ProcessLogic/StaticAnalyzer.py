#this program calculate the 3D coordinate data into length
# and save the file as the weka format
import os
import string
import shutil
import numpy as np
import GaitData as gd

limbDescriptors  = ['neck','rshoulder','lshoulder','rarm','larm','rfarm','lfarm',
					'rhand','lhand','uspine','lspine','rhip','lhip','rthigh','lthigh',
					'rcalf','lcalf','rfoot','lfoot','height']

class StaticAnalyzer:
	def __init__(self,path,limbList={}):
		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.datas = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.gaitData = gd.GaitData()
		self.srcTrainPath = path+"/Dataset/TrainDataset/TrainGaitDataset"
		self.dstTrainPath = path+"/Dataset/TrainDataset/TrainStaticDataset"
		self.srcTestPath = path+"/Dataset/TestDataset/TestGaitDataset"
		self.dstTestPath = path+"/Dataset/TestDataset/TestStaticDataset"

		self.listMap = {}
		for l in limbDescriptors:
			self.listMap[l] = 0

		for l in limbList:
			self.listMap[l] = 1
		print self.listMap
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

	def euclidis (self,joint1, joint2):
		dif = (joint1-joint2)
		dis = dif*dif
		summary = dis[0:,0] + dis[0:,1] + dis[0:,2]
		return summary**0.5

	def listdirNohidden(self,path):
		for f in os.listdir(path):
			if not f.startswith('.'):
				yield f

	def clear(self):
		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

	def readData(self,personFilePath):
		dSize = len(limbDescriptors)
		person = open(personFilePath)
		personData = person.readlines()
		if len(personData) == 0:
			print "The data of file is empty:"
			print personFilePath
			return
		length = len(personData)
		for item in range(0,length/dSize):
			for seg in range(0,dSize):
				temp = personData[item*dSize + seg].split(";")
				point = [string.atof(temp[1]),string.atof(temp[2]),string.atof(temp[3].replace("\n",''))]
				self.points[seg].append(point)
		person.close()

	#this function is to caculate mean and std of raw data
	#and then filter raw data and write
	def filterMeanStd(self):
		dSize = len(limbDescriptors)
		means = np.zeros(dSize)
		std = np.zeros(dSize)
		# pre-caculate mean and std of every limb
		lens = len(self.datas[0])
		for i in range(dSize):
			means[i] = self.datas[i].sum()/lens
			std[i] = np.std(self.datas[i])
		for i in range(dSize):
			j = 0
			while(j < len(self.datas[i])):
				if (self.datas[i][j] < means[i] - 2*std[i]) or (self.datas[i][j] > means[i] + 2*std[i]):
					for k in range(dSize):
						self.datas[k] = np.delete(self.datas[k],j,0)
				j += 1
		for i in range(0,dSize):
			self.trainData.write(str(self.datas[i].sum()/lens)+',')
			#self.fileDistance.write(str(self.datas[i].sum()/lens)+',')
		self.trainData.write(str(self.gaitData.getId())+'\n')
		#self.fileDistance.write(str(self.gaitData.getId())+'\n')

	def dataProcess(self):
		self.mode = "Train Static:"
		self.trainData = open(self.dstTrainPath+"/static.txt",'w')
		self.staticAnalysis(self.srcTrainPath,self.dstTrainPath)
		self.trainData.close()

		self.mode = "Test  Static:"
		self.trainData = open(self.dstTestPath+"/static.txt",'w')
		self.staticAnalysis(self.srcTestPath,self.dstTestPath)
		self.trainData.close()

	def staticAnalysis(self,srcPath,dstPath):
		#The first step is to read data from files
		personDirectorsPath = srcPath
		personDirectors = self.listdirNohidden(personDirectorsPath)
		for personDirector in personDirectors:
			personDirectorPath = personDirectorsPath + '/' + personDirector
			if not os.path.isdir(personDirectorPath):
				continue
			print self.mode,personDirector
			personFiles = self.listdirNohidden(personDirectorPath)
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
				self.readData(personFilePath)
				self.gaitData.setId(int(personDirector.replace("Person",'')))
		#The second step is to deal with the data
				self.calculateLength()
				self.filterMeanStd()
		#The third step is to write data into file
				self.writeData(writePersonDirectorPath+personFile)

	#this function write down all data
	def writeData(self,dstPersonFile):
		dstFile = open(dstPersonFile,'w')
		length = len(self.datas[0])
		for line in range(0,length):
			for i in range(0,20):
				if self.listMap[limbDescriptors[i]] == 1:
					dstFile.write(limbDescriptors[i]+',')
					dstFile.write(str(self.datas[i][line])+'\n')
			# dstFile.write('\n')
		dstFile.close()

	def calculateLength(self):
		head = np.array(self.points[0])
		shoulderCenter = np.array(self.points[1])
		shoulderRight = np.array(self.points[2])
		shoulderLeft = np.array(self.points[3])
		elbowRight = np.array(self.points[4])
		elbowLeft = np.array(self.points[5])
		wristRight = np.array(self.points[6])
		wristLeft = np.array(self.points[7])
		handRight = np.array(self.points[8])
		handLeft = np.array(self.points[9])
		spine = np.array(self.points[10])
		hipCenter = np.array(self.points[11])
		hipRight = np.array(self.points[12])
		hipLeft = np.array(self.points[13])
		kneeRight = np.array(self.points[14])
		kneeLeft = np.array(self.points[15])
		ankleRight = np.array(self.points[16])
		ankleLeft = np.array(self.points[17])
		footRight = np.array(self.points[18])
		footLeft = np.array(self.points[19])

		#1.neck
		neck = self.euclidis(head,shoulderCenter)
		self.datas[0] = self.euclidis(head,shoulderCenter)#.tolist()
		#2.right-shoulder
		rightShoulder = self.euclidis(shoulderCenter,shoulderRight)
		self.datas[1] = self.euclidis(shoulderCenter,shoulderRight)#.tolist()
		#3.left-shoulder
		leftShoulder = self.euclidis(shoulderCenter,shoulderLeft)
		self.datas[2] = self.euclidis(shoulderCenter,shoulderLeft)#.tolist()
		#4.right-arm
		rightArm = self.euclidis(shoulderRight,elbowRight)
		self.datas[3] = self.euclidis(shoulderRight,elbowRight)#.tolist()
		#5.left-arm
		leftArm = self.euclidis(shoulderLeft,elbowLeft)
		self.datas[4] = self.euclidis(shoulderLeft,elbowLeft)#.tolist()
		#6.right-front-arm
		rightFrontArm = self.euclidis(elbowRight,wristRight)
		self.datas[5] = self.euclidis(elbowRight,wristRight)#.tolist()
		#7.left-front-arm
		leftFrontArm = self.euclidis(elbowLeft,wristLeft)
		self.datas[6] = self.euclidis(elbowLeft,wristLeft)#.tolist()
		#8.right-hand
		rightHand = self.euclidis(wristRight,handRight)
		self.datas[7] = self.euclidis(wristRight,handRight)#.tolist()
		#9.left-hand
		leftHand = self.euclidis(wristLeft,handLeft)
		self.datas[8] = self.euclidis(wristLeft,handLeft)#.tolist()
		#10.upper-spine
		upperSpine = self.euclidis(shoulderCenter,spine)
		self.datas[9] = self.euclidis(shoulderCenter,spine)#.tolist()
		#11.lower-spine
		lowerSpine = self.euclidis(spine,hipCenter)
		self.datas[10] = self.euclidis(spine,hipCenter)#.tolist()
		#12.right-hip
		rightHip = self.euclidis(hipCenter,hipRight)
		self.datas[11] = self.euclidis(hipCenter,hipRight)#.tolist()
		#13.left-hip
		leftHip = self.euclidis(hipCenter,hipLeft)
		self.datas[12] = self.euclidis(hipCenter,hipLeft)#.tolist()
		#14.right-thigh
		rightThigh = self.euclidis(hipRight,kneeRight)
		self.datas[13] = self.euclidis(hipRight,kneeRight)#.tolist()
		#15.left-thigh
		leftThigh = self.euclidis(hipLeft,kneeLeft)
		self.datas[14] = self.euclidis(hipLeft,kneeLeft)#.tolist()
		#16.right-leg
		rightLeg = self.euclidis(kneeRight,ankleRight)
		self.datas[15] = self.euclidis(kneeRight,ankleRight)#.tolist()
		#17.left-leg
		leftLeg = self.euclidis(kneeLeft,ankleLeft)
		self.datas[16] = self.euclidis(kneeLeft,ankleLeft)#.tolist()
		#18.right-foot
		rightFoot = self.euclidis(ankleRight,footRight)
		self.datas[17] = self.euclidis(ankleRight,footRight)#.tolist()
		#19.left-foot
		leftFoot = self.euclidis(ankleLeft,footLeft)
		self.datas[18] = self.euclidis(ankleLeft,footLeft)#.tolist()
		#20.height
		height = neck + upperSpine + lowerSpine + (rightHip + leftHip)/2 + (rightThigh+leftThigh)/2 + (rightLeg+leftLeg)/2 + (rightFoot+leftFoot)/2
		self.datas[19] = (neck + upperSpine + lowerSpine + (rightHip + leftHip)/2 + (rightThigh+leftThigh)/2 + (rightLeg+leftLeg)/2 + (rightFoot+leftFoot)/2)#.tolist()

if __name__=="__main__":
	#train data
	homedir = os.getcwd()
	sa = StaticAnalyzer(homedir,limbDescriptors)
	sa.dataProcess()
