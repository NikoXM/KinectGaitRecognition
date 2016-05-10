import os
import numpy as np
import string
import GaitData as gd
import matplotlib.pyplot as plt

joint_descriptors = ['Head', 'Shoulder-Center', 'Shoulder-Right', 'Shoulder-Left', 'Elbow-Right', 'Elbow-Left', 'Wrist-Right', 'Wrist-Left',
			   'Hand-Right', 'Hand-Left', 'Spine', 'Hip-centro', 'Hip-Right', 'Hip-Left', 'Knee-Right', 'Knee-Left',
			   'Ankle-Right', 'Ankle-Left', 'Foot-Right', 'Foot-Left']


dstPath = ["/Users/niko/Documents/KinectGaitScripts/Data/ConvertedData",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/convertedData"]
# dstPath = "/Users/niko/Documents/KinectGaitScripts/Data/convertedData/"
srcPath = ["/Users/niko/Documents/KinectGaitScripts/FilteredGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/RawGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/FilteredGaitDataset"]

class Convertor:
	'This class convert coordinate from raw to newly aligned'
	def __init__(self,srcPath,dstPath):
		self.gaitData = gd.GaitData()

		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.srcPath = srcPath
		self.dstPath = dstPath
		self.is_converted = False

	def data_process(self):
		personDirectorsPath = self.srcPath
		personDirectors = self.listdir_nohidden(personDirectorsPath)
		for personDirector in personDirectors:
			personDirectorPath = personDirectorsPath + '/' + personDirector
			personFiles = self.listdir_nohidden(personDirectorPath)
			for personFile in personFiles:
				personFilePath = personDirectorPath + '/' + personFile
				print personFilePath
				self.read_data(personFilePath)
				#self.convert()
				self.testAngleArange()
				writePersonDirectorPath = self.dstPath +'/' + personDirector + '/'
				if(os.path.exists(writePersonDirectorPath)):
					self.write_data(writePersonDirectorPath+personFile)
				else:
					os.mkdir(writePersonDirectorPath)
					self.write_data(writePersonDirectorPath+personFile)
		self.is_converted = True

	def testSpineStraight(self):
		data = np.array(self.points[self.gaitData.shoulder_center])
		y = data[:,1]
		modulo = np.sqrt(data[:,0]**2+data[:,1]**2+data[:,2]**2)
		cos_theta = y/modulo
		theta = np.arccos(cos_theta)/np.pi
		x = np.arange(len(theta))
		plt.plot(x,theta)
		plt.show()

	def testAngleArange(self):
		shoulder_center = np.array(self.points[self.gaitData.shoulder_center])
		hip_center = np.array(self.points[self.gaitData.hip_center])
		left_knee = np.array(self.points[self.gaitData.knee_left])
		a1 = hip_center - shoulder_center
		a2 = left_knee - hip_center
		angle = self.calculateAngle(a1,a2)
		self.imageShow(angle)

	def calculateAngle(self,vector1,vector2):
		dot_multi = vector1*vector2
		dot_multi = dot_multi[:,0]+dot_multi[:,1]+dot_multi[:,2]
		cos_theta = dot_multi/(self.getModulo(vector1)*self.getModulo(vector2))
		return np.arccos(cos_theta)

	def getModulo(self,vector):
		x = vector[:,0]*vector[:,0]
		y = vector[:,1]*vector[:,1]
		z = vector[:,2]*vector[:,2]
		return np.sqrt(x+y+z)

	#this function convert the original coordinate to new coordinate
	def convert(self):
		data = np.array(self.points)
		center = np.copy(data[self.gaitData.hip_center])
		#the first step is to move the origin from kinect to body
		for i in range(len(data)):
				data[i] = data[i] - center
		#the vertical axis keep unchange
		#the second step is to twist x and z axis
		x_new = data[self.gaitData.shoulder_left] - data[self.gaitData.shoulder_right]
		modulo = np.sqrt(x_new[:,0]**2+x_new[:,1]**2+x_new[:,2]**2)
		cos_theta = (x_new[:,0]/modulo)
		sin_theta = (np.sqrt(1-cos_theta**2))

		x = 0
		z = 2
		for i in range(len(data)):
			temp = data[i]
			data[i][:,x] = cos_theta * temp[:,x] + sin_theta * temp[:,z]
			data[i][:,z] = sin_theta * temp[:,x] + cos_theta * temp[:,z]
		# x = np.arange(len(cos_theta))
		# plt.plot(x,sin_theta)
		# plt.plot(x,cos_theta)
		# plt.show()
		
		self.points = data.tolist()

	def listdir_nohidden(self,path):
		for f in os.listdir(path):
			if not f.startswith('.'):
				yield f

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
			#need to append different kind of data into different lists
					point = [string.atof(temp[1]),string.atof(temp[2]),string.atof(temp[3].replace("\n",''))]
					self.points[seg].append(point)
			#print self.points

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

	def get_data(self):
		if self.is_converted:
			return self.points
		else:
			self.convert()
			return self.points

	def imageShow(self,points):
		self.medfilter(points,1)
		xdata = np.arange(len(points))
		plt.plot(xdata,points)
		plt.show()

	def medfilter(self,angle,size):
		if size%2 == 0:
			print "size must be odd"
			return
		pad = (size-1)/2
		angle_copy = np.copy(angle)
		for i in range(pad,len(angle)-pad-1):
			sum = angle_copy[i]
			for j in range(1,pad+1):
				sum += angle_copy[i-j]+angle_copy[i+j]
			angle[i] = sum/size

if __name__ == '__main__':
	convertor = Convertor(srcPath[2],dstPath[1])
	convertor.data_process()