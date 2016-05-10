#import sys
#sys.path.append("/Users/niko/Documents/KinectGaitScripts/")
import os
import string
import numpy as np
import GaitData as gd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit

dstPath = ["/Users/niko/Documents/KinectGaitScripts/Data/ConvertedData",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/ConvertedData"]
#The data path contain converted data
srcPath = ["/Users/niko/Documents/KinectGaitScripts/FilteredGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/RawGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/ConvertedData"]

def fourier_func_1(x,w,a0,a1,b1):
	return a0 + a1*np.cos(x*w) + b1*np.sin(x*w)
def fourier_func_2(x,w,a0,a1,b1,a2,b2):
	return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w)
def fourier_func_3(x,w,a0,a1,b1,a2,b2,a3,b3):
	return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w) + a3*np.cos(3*x*w) + b3*np.sin(3*x*w)
def fourier_func_4(x,w,a0,a1,b1,a2,b2,a3,b3,a4,b4):
	return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w) + a3*np.cos(3*x*w) + b3*np.sin(3*x*w) + a4*np.cos(4*x*w) + b4*np.sin(4*x*w)
def fourier_func_5(x,w,a0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5):
	return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w) + a3*np.cos(3*x*w) + b3*np.sin(3*x*w) + a4*np.cos(4*x*w) + b4*np.sin(4*x*w) + a5*np.cos(5*x*w) + b5*np.sin(5*x*w)
def fourier_func_6(x,w,a0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6):
	return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w) + a3*np.cos(3*x*w) + b3*np.sin(3*x*w) + a4*np.cos(4*x*w) + b4*np.sin(4*x*w) + a5*np.cos(5*x*w) + b5*np.sin(5*x*w) + a6*np.cos(6*x*w) + b6*np.sin(6*x*w)
def fourier_func_7(x,w,a0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7):
	return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w) + a3*np.cos(3*x*w) + b3*np.sin(3*x*w) + a4*np.cos(4*x*w) + b4*np.sin(4*x*w) + a5*np.cos(5*x*w) + b5*np.sin(5*x*w) + a6*np.cos(6*x*w) + b6*np.sin(6*x*w) + a7*np.cos(7*x*w) + b7*np.sin(7*x*w)

class fourierFitting:
	'this class is to fitting the curve of walk'

	def __init__(self,srcPath,dstPath):
		self.gaitData = gd.GaitData()
		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.srcPath = srcPath
		self.dstPath = dstPath
		self.ndfit = 1

	def listdir_nohidden(self,path):
		for f in os.listdir(path):
			if not f.startswith('.'):
				yield f

	def data_process(self):
		#The first step is to read data from files
		personDirectorsPath = self.srcPath
		personDirectors = self.listdir_nohidden(personDirectorsPath)
		for personDirector in personDirectors:
			personDirectorPath = personDirectorsPath + '/' + personDirector
			personFiles = self.listdir_nohidden(personDirectorPath)
			for personFile in personFiles:
				personFilePath = personDirectorPath + '/' + personFile
				self.read_data(personFilePath)
		#The second step is to deal with the data
				#test_joints = np.array([self.gaitData.knee_right, self.gaitData.knee_left])
				test_joints = np.array([self.gaitData.knee_left])
				self.getJointSwingImage(test_joints)

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

	def set_ndfit(self,nd):
		self.ndfit = nd

	def curve_fitting(self,angle):
		x = np.arange(len(angle))
		if self.ndfit == 1:
			return curve_fit(fourier_func_1,x,angle)
		elif self.ndfit == 2:
			return curve_fit(fourier_func_2,x,angle)
		elif self.ndfit == 3:
			return curve_fit(fourier_func_3,x,angle)
		elif self.ndfit == 4:
			return curve_fit(fourier_func_4,x,angle)
		elif self.ndfit == 5:
			return curve_fit(fourier_func_5,x,angle)
		elif self.ndfit == 6:
			return curve_fit(fourier_func_6,x,angle)
		elif self.ndfit == 7:
			return curve_fit(fourier_func_7,x,angle)
		else:
			print "size must be between 1 and 7"
			return False

	def apply_function(self,xdata,arg):
		if self.ndfit == 1:
			print arg
			return fourier_func_1(xdata,arg[0],arg[1],arg[2],arg[3])

	def getModulo(self,vector):
		x = vector[:,0]**2
		y = vector[:,1]**2
		z = vector[:,2]**2
		return np.sqrt(x+y+z)

	def calculateAngle(self, joint):
		data = np.array(self.points)
		a1 = data[self.gaitData.hip_center] - data[self.gaitData.shoulder_center]
		a2 = data[joint] - data[self.gaitData.hip_center]
		modulo = self.getModulo(a1)*self.getModulo(a2)
		dot_multi = a1*a2
		dot_multi = dot_multi[:,0]+dot_multi[:,1]+dot_multi[:,2]
		cos_theta = (dot_multi)/modulo
		angle = np.arccos(cos_theta)
		return angle

	def getJointSwingImage(self,joints):
		for joint in joints:
			angle = fourier.calculateAngle(joint)
			self.medfilter(angle,21)
			x = np.arange(len(angle))
			plt.plot(x,angle)
			#use fourier funtion to fit the image
			self.set_ndfit(1)
			popt, pcov = self.curve_fitting(angle)
			#apply the model argument to real fourier funtion
			ydata = self.apply_function(x,popt)
			plt.plot(x,ydata)

			#the next operation is only for test
			matlabFilePath = "/Users/niko/Documents/KinectGaitScripts/TestOnlyData/MatlabData/1.txt"
			matlabFile = open(matlabFilePath,'w')
			for i in range(len(angle)-1):
				matlabFile.write(str(angle[i]) + ' ')
			matlabFile.write(str(angle[len(angle)-1]))
			matlabFile.close()
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

if __name__=="__main__":
	fourier = fourierFitting(srcPath[2],dstPath[1])
	fourier.data_process()
	