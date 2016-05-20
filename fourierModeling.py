#import sys
#sys.path.append("/Users/niko/Documents/KinectGaitScripts/")
import os
import string
import numpy as np
import GaitData as gd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit
import random as rd

wekaFilePath = "/Users/niko/Documents/KinectGaitScripts/TestOnlyData/WekaDataset/fourier.arff"

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

def poly_func_2(x,a0,a1,a2):
	return a2*x**2 + a1*x + a0

class fourierFitting:
	'this class is to fitting the curve of walk'

	def __init__(self,srcPath,dstPath,number=1):
		self.gaitData = gd.GaitData()
		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		self.srcPath = srcPath
		self.dstPath = dstPath
		self.ndfit = 1
		self.number = number
		self.wekaFile = open(wekaFilePath,'w')
		self.wekaFile.write("@relation fourier fitting\n")
		for i in range(4):
			self.wekaFile.write("@attribute ")
			self.wekaFile.write(str(i)+" ")
			self.wekaFile.write("numeric\n")
		self.wekaFile.write("@attribute identification numeric\n@data\n")
		self.frequecy = []
		self.N = None
		# self.sparseMatric = []

	def __del__(self):
		self.wekaFile.close()

	# def writeSparseMatric(self):
	# 	path = "/Users/niko/Documents/KinectGaitScripts/TestOnlyData/x.txt"
	# 	file = open(path,'wa')
	# 	length = len(self.sparseMatric)
	# 	s = []
	# 	for i in range(length):
	# 		s.append('0')
	# 	for i in range(length):
	# 		for j in range(self.sparseMatric[i]):
	# 			s[i] = '1'
	# 			for k in range(length):
	# 				file.write(str(s[k]))
	# 				file.write(' ')
	# 			file.write('\n')
	# 			s[i] = 0
	# 	file.close()

	def listdir_nohidden(self,path):
		for f in os.listdir(path):
			if not f.startswith('.'):
				yield f

	def clear(self):
		self.points = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

	def data_process(self):
		#The first step is to read data from files
		personDirectorsPath = self.srcPath
		personDirectors = self.listdir_nohidden(personDirectorsPath)
		for personDirector in personDirectors:
			personDirectorPath = personDirectorsPath + '/' + personDirector
			personFiles = self.listdir_nohidden(personDirectorPath)
			#caculate the number of nth file
			# fPoint = 0
			for personFile in personFiles:
				personFilePath = personDirectorPath + '/' + personFile
				print personFilePath
				self.clear()
				self.read_data(personFilePath)
				self.gaitData.setId(int(personDirector.replace("Person",'')))
				# fPoint += 1
		#The second step is to deal with the data
				#test_joints = np.array([self.gaitData.knee_right, self.gaitData.knee_left])
				angle_joints = np.array([self.gaitData.ankle_left])
				self.drawAngleCurve(angle_joints)

				speed_joints = [self.gaitData.hip_center,self.gaitData.shoulder_center,self.gaitData.head,self.gaitData.spine]
				#self.drawSpeedCurve(speed_joints)
				#self.drawDirection()
			#append the number to sparse matric
			# self.sparseMatric.append(fPoint)

		#self.writeSparseMatric()

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

	def set_ndfit(self,nd):
		self.ndfit = nd

	def curve_fitting(self,x,angle):
		#p0 = np.ones(2*(self.ndfit+1))
		#p0 = [0,0,0,0,0,0.010101584095144]
		# p0 = [0]*(2*(self.ndfit+1) - 1)
		# p0.append(0.01*rd.random())
		# print p0
		# p0 = np.array(p0)
		p0 = 0.01*np.random.normal(size=2*(self.ndfit+1))

		if self.ndfit == 1:
			return curve_fit(fourier_func_1,x,angle,p0,maxfev=5000)
		elif self.ndfit == 2:
			return curve_fit(fourier_func_2,x,angle,p0,maxfev=5000)
		elif self.ndfit == 3:
			return curve_fit(fourier_func_3,x,angle,p0,maxfev=5000)
		elif self.ndfit == 4:
			return curve_fit(fourier_func_4,x,angle,p0,maxfev=5000)
		elif self.ndfit == 5:
			return curve_fit(fourier_func_5,x,angle,p0,maxfev=5000)
		elif self.ndfit == 6:
			return curve_fit(fourier_func_6,x,angle,p0,maxfev=5000)
		elif self.ndfit == 7:
			return curve_fit(fourier_func_7,x,angle,p0,maxfev=5000)
		elif self.ndfit == -1:
			return curve_fit(poly_func_2,x,angle)
		else:
			print "size must be between 1 and 7"
			return False

	def apply_function(self,xdata,arg):
		if self.ndfit == 1:
			return fourier_func_1(xdata,arg[0],arg[1],arg[2],arg[3])
		elif self.ndfit == 2:
			return fourier_func_2(xdata,arg[0],arg[1],arg[2],arg[3],arg[4],arg[5])
		elif self.ndfit == 3:
			return fourier_func_3(xdata,arg[0],arg[1],arg[2],arg[3],arg[4],arg[5],arg[6],arg[7])
		elif self.ndfit == 4:
			return fourier_func_4(xdata,arg[0],arg[1],arg[2],arg[3],arg[4],arg[5],arg[6],arg[7],arg[8],arg[9])
		elif self.ndfit == 5:
			return fourier_func_5(xdata,arg[0],arg[1],arg[2],arg[3],arg[4],arg[5],arg[6],arg[7],arg[8],arg[9],arg[10],arg[11])
		elif self.ndfit == 6:
			return fourier_func_6(xdata,arg[0],arg[1],arg[2],arg[3],arg[4],arg[5],arg[6],arg[7],arg[8],arg[9],arg[10],arg[11],arg[12],arg[13])
		elif self.ndfit == 7:
			return fourier_func_7(xdata,arg[0],arg[1],arg[2],arg[3],arg[4],arg[5],arg[6],arg[7],arg[8],arg[9],arg[10],arg[11],arg[12],arg[13],arg[14],arg[15])
		elif self.ndfit == -1:
			return poly_func_2(xdata,arg[0],arg[1],arg[2])
		else:
			print "size error"

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

#extra periods from the vedio sequence
	def extractPeriods(self,angle):
		sign = np.sign(np.diff(angle))
		ddiff = np.diff(sign)
		#plt.plot(np.arange(len(angle)),angle)
		
		# for i in range(np.size(ddiff)):
		# 	if ddiff[i] == -2:
		# 		plt.annotate('local max', xy=(i, angle[i+1]), xytext=(i, angle[i+1]+0.05),arrowprops=dict(facecolor='black', shrink=0.05),)
		# 	elif ddiff[i] == 2:
		# 		plt.annotate('local min', xy=(i, angle[i+1]), xytext=(i, angle[i+1]-0.05),arrowprops=dict(facecolor='red', shrink=0.05),)

		ddiff = np.diff(np.sign(np.diff(angle)))
		periods = []
		x = np.arange(len(angle))

		temp = []
		xs = []
		i = 0
		#find the first crest
		while i < np.size(ddiff) and ddiff[i] != -2:
			i += 1
		i += 1
		while i < np.size(ddiff):
			temp = []
			temp.append(angle[i])
			xs.append(x[i])
			j = i + 1
			while j < np.size(ddiff) and ddiff[j] != -2:
				temp.append(angle[j])
				j = j+1
			if j >= np.size(ddiff):
				break
			temp.append(angle[j])
			temp.append(angle[j+1])
			i = j+1
			periods.append(temp)
		return periods

	def periodsFilter(self,periods):
		i = 0
		while i < len(periods):
			p_max = max(periods[i])
			p_min = min(periods[i])
			if(abs(periods[i][0]-periods[i][len(periods[i])-1]) >= (9./10.)*(p_max - p_min)):
				del periods[i]
				i -= 1
			i += 1
		#one period have at least 30 frames
		i = 0
		while i < len(periods):
			if(len(periods[i]) < 30):
				del periods[i]
				i -= 1
			i += 1
		return periods
		# print x_diff
		# print y_diff
		# x_index = (x_arr >= (np.mean(x_arr)-2*np.std(x_arr))) * (x_arr <= (np.mean(x_arr)+2*np.std(x_arr)))
		# y_index = (y_arr >= (np.mean(y_arr)-2*np.std(y_arr))) * (y_arr <= (np.mean(y_arr)+2*np.std(y_arr)))
		# print x_index
		# print y_index
		# print len(periods)
		#periods = periods[x_index]
		#periods = periods[y_index]
		#print periods

	def drawAngleCurve(self,joints):
		for joint in joints:
			angle = fourier.calculateAngle(joint)
			self.meanfilter(angle,15)
			periods = self.extractPeriods(angle)
			periods = self.periodsFilter(periods)
#fourier curve fitting
			# for i in range(len(periods)):
			# 	print periods[i]
			# 	p = np.array(periods[i])
			# 	self.set_ndfit(3)
			# 	x = np.arange(len(p))
			# 	popt, pcov = self.curve_fitting(x,p)
			# 	ydata = self.apply_function(x,popt)
			# 	plt.plot(x,p)
			# 	plt.plot(x,ydata)
			self.set_ndfit(2)
			# for p in periods:
			for p in periods:
				x = np.arange(len(p))
				popt,pcov = self.curve_fitting(x,np.array(p))
				ydata = self.apply_function(x,popt)
				# plt.plot(x,np.array(p))
				# plt.plot(x,ydata)
				#write fit parameter(except a0) to classfy
				for i in range(len(popt)):
					if i == 1:
						continue
					self.wekaFile.write(str(popt[i])+' ')
				self.wekaFile.write(str(self.gaitData.getId()))
				self.wekaFile.write('\n')

#fourier transform
			# amplitude, phase = self.fourierTransform(angle)
			# write the amplitude and phase as the parameter of recognition
			# for alp in amplitude:
			# 	self.wekaFile.write(str(alp))
			# 	self.wekaFile.write(str(','))
			# for phs in phase:
			# 	self.wekaFile.write(str(phs))
			# 	self.wekaFile.write(str(','))
#machine learning
			# for i in frequency:
			# 	self.wekaFile.write(str(i)+' ')
			# self.wekaFile.write('\n')
			#self.wekaFile.write()
			#self.wekaFile.write(str(self.gaitData.getId())+'\n')

#only for test
			matlabFilePath = "/Users/niko/Documents/KinectGaitScripts/TestOnlyData/MatlabData/1.txt"
			matlabFile = open(matlabFilePath,'w')
			for i in range(len(angle)-1):
				matlabFile.write(str(angle[i]) + ' ')
			matlabFile.write(str(angle[len(angle)-1]))
			matlabFile.close()
		plt.show()

#the speed doesn't work
	def drawSpeedCurve(self,joints):
		data = np.array(self.points)
		limbSpeeds = []
		for limb in joints:
			temp = []
			for t in range(1,len(data[0])):
				dis = data[limb][t] - data[limb][t-1]
				temp.append(np.sqrt(dis[0]**2+dis[1]**2+dis[2]**2))
			limbSpeeds.append(temp)
		limbSpeedsData = np.array(limbSpeeds)

		size = len(limbSpeedsData[0])
		avg = np.zeros(size)
		for i in range(len(lists)):
			avg += limbSpeedsData[i]
		avg = avg/size
		self.meanfilter(avg,21)
		plt.plot(np.arange(size),avg.tolist())
		plt.show()
#the angle seems not to work so well
	def drawDirection(self):
		data = np.array(self.points)
		x_new = data[self.gaitData.hip_left] - data[self.gaitData.hip_right]
		y_new = data[self.gaitData.shoulder_center] - data[self.gaitData.hip_center]
		z_new = np.cross(x_new,y_new)
		cos_theta = z_new[:,2]/self.getModulo(z_new)
		self.medianfilter(cos_theta,21)
		plt.plot(np.arange(len(cos_theta.tolist())),cos_theta.tolist())
		plt.show()

	def drawPath(self):
		return 0

	def meanfilter(self,array,size):
		if size%2 == 0:
			print "size must be odd"
			return
		pad = (size-1)/2
		array_copy = np.copy(array)
		for i in range(pad,len(array)-pad-1):
			sum = array_copy[i]
			for j in range(1,pad+1):
				sum += array_copy[i-j]+array_copy[i+j]
			array[i] = sum/size

	def medianfilter(self,array,size):
		if size%2 == 0:
			print "size must be odd"
			return
		pad = (size-1)/2
		for i in range(pad,len(array)-pad-1):
			temp = np.sort(array[i-pad:i+pad+1])
			print temp
			array[i] = temp[pad]

	def nextpow2(self,n):
		mf = np.log2(n)
		mi = np.ceil(mf)
		return int(2**mi)

	def fourierTransform(self,angle):
		Fs = 256
		m = np.shape(angle)
		m = 2048
		#plt.plot(np.arange(m), angle)
		N = self.nextpow2(m)

		#the real amplitude is the result of devided by N and multiplied  by 2.
		# the N bigger, accuracy is better
		Y = np.fft.fft(angle,N)/N*2
		#frequecy
		f = Fs/N*np.linspace(0,1,N-1)
		#amplitude
		A = abs(Y)
		#phase
		P = np.angle(Y)
		#***********show the image**************
		# plt.plot(np.arange(N/2),A[0:N/2])
		# plt.show()
		if self.N != None and self.N > 2048:
			print "not same\n"
		self.N = N
		return A,P

dstPath = ["/Users/niko/Documents/KinectGaitScripts/Data/ConvertedData",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/ConvertedData"]
#The data path contain converted data
srcPath = ["/Users/niko/Documents/KinectGaitScripts/Data/FilteredGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/Data/RawGaitDataset",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/ConvertedData",
		"/Users/niko/Documents/KinectGaitScripts/TestOnlyData/FilteredGaitDataset"]

if __name__=="__main__":
	fourier = fourierFitting(srcPath[3],dstPath[1])
	fourier.data_process()
	