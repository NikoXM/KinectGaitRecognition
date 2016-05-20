import os
import string
import shutil
import random as rd
import numpy as np
import GaitData as gd

dstPath = ["/Users/niko/Documents/KinectGaitScripts/Data/FilteredGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/FilteredGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/trainGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/testGaitDataset",]
# dstPath = "/Users/niko/Documents/KinectGaitScripts/Data/convertedData/"
srcPath = ["/Users/niko/Documents/KinectGaitScripts/Data/FilteredGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/Data/RawGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/RawGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/FilteredGaitDataset"]

joint_descriptors = ['Head', 'Shoulder-Center', 'Shoulder-Right', 'Shoulder-Left', 'Elbow-Right', 'Elbow-Left', 'Wrist-Right', 'Wrist-Left',
			   'Hand-Right', 'Hand-Left', 'Spine', 'Hip-centro', 'Hip-Right', 'Hip-Left', 'Knee-Right', 'Knee-Left',
			   'Ankle-Right', 'Ankle-Left', 'Foot-Right', 'Foot-Left']

class RandomSelect:
	def __init__(self,srcPath,trainPath,testPath,p):
		self.gaitData = gd.GaitData()
		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.trainPath = trainPath
		self.testPath = testPath
		self.srcPath = srcPath
		self.trainMatric = []
		self.testMatric = []
		self.p = p

	def listdir_nohidden(self,path):
		for f in os.listdir(path):
			if not f.startswith('.'):
				yield f
	def clear(self):
		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

	def writeSparseMatric(self):
		trainPath = self.trainPath + '/' + "y.txt"
		testPath = self.testPath + '/' + "y.txt"

		trainFile = open(trainPath,'w')
		testFile = open(testPath,'w')
		trainLength = len(self.trainMatric)
		testLength = len(self.testMatric)
		s = []
		for i in range(trainLength):
			s.append('0')
		for i in range(trainLength):
			for j in range(self.trainMatric[i]):
				s[i] = '1'
				for k in range(trainLength):
					trainFile.write(str(s[k]))
					trainFile.write(' ')
				trainFile.write('\n')
				s[i] = '0'
		s = []
		for i in range(testLength):
			s.append('0')
		for i in range(testLength):
			for j in range(self.testMatric[i]):
				s[i] = '1'
				for k in range(testLength):
					testFile.write(str(s[k]))
					testFile.write(' ')
				testFile.write('\n')
				s[i] = '0'
		trainFile.close()
		testFile.close()

	def data_process(self):
		personDirectorsPath = self.srcPath
		personDirectors = self.listdir_nohidden(personDirectorsPath)
		for personDirector in personDirectors:
			personDirectorPath = personDirectorsPath + '/' + personDirector
			personFiles = self.listdir_nohidden(personDirectorPath)
			
			trainPoint = 0
			testPoint = 0
			
			tempList = []
			for personFile in personFiles:
				tempList.append(personFile)
			personFiles = tempList
			count = int(round(len(personFiles)*self.p))
			trainFiles = rd.sample(personFiles,count)
			testFiles = list(set(personFiles).difference(set(trainFiles)))
			#train
			writePersonDirectorPath = self.trainPath +'/' + personDirector + '/'
			if(os.path.exists(writePersonDirectorPath)):
				shutil.rmtree(writePersonDirectorPath)
				os.mkdir(writePersonDirectorPath)
			for personFile in trainFiles:
				personFilePath = personDirectorPath + '/' + personFile
				print "training:"+personFilePath
				self.clear()
				self.read_data(personFilePath)
				#self.filter()
				trainPoint += 1
				if(os.path.exists(writePersonDirectorPath)):
					self.write_data(writePersonDirectorPath+personFile)
				else:
					os.mkdir(writePersonDirectorPath)
					self.write_data(writePersonDirectorPath+personFile)
			#test
			writePersonDirectorPath = self.testPath +'/' + personDirector + '/'
			if(os.path.exists(writePersonDirectorPath)):
				shutil.rmtree(writePersonDirectorPath)
				os.mkdir(writePersonDirectorPath)
			for personFile in testFiles:
				personFilePath = personDirectorPath + '/' + personFile
				print "testing:"+personFilePath
				self.clear()
				self.read_data(personFilePath)
				#self.filter()
				writePersonDirectorPath = self.testPath +'/' + personDirector + '/'
				testPoint += 1
				if(os.path.exists(writePersonDirectorPath)):
					self.write_data(writePersonDirectorPath+personFile)
				else:
					os.mkdir(writePersonDirectorPath)
					self.write_data(writePersonDirectorPath+personFile)

			self.trainMatric.append(trainPoint)
			self.testMatric.append(testPoint)
			self.writeSparseMatric()

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

	def write_data(self,dstPersonFile):
		dstFile = open(dstPersonFile,'w')
		points = np.array(self.points)
		length = len(points[0])
		for frame in range(length):
			for limb in range(0,len(points)):
				point = joint_descriptors[limb]+ ";"
				for i in range(2):
					point += str(points[limb][frame][i]) + ";"
				point += str(points[limb][frame][2]) + '\n'
				dstFile.write(point)
		dstFile.close()

if __name__ == '__main__':
	rs = RandomSelect(srcPath[3],trainPath = dstPath[2],testPath = dstPath[3],p=0.7)
	rs.data_process()