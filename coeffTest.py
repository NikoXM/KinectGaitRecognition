import numpy as np
import matplotlib.pyplot as plt

filePath = "/Users/niko/Documents/KinectGaitScripts/TestOnlyData/WekaDataset/fourier.arff"
lines = open(filePath,'r').readlines()

def func(x,a1,a2,a3,a4,a5,a6,a7):
	return a7*x**7 + a6*x**6 + a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x

for line in lines:
	if(line[0] == '@'):
		continue
	line = line.split(',')
	x = np.arange(100)
	y = func(x,float(line[0]),float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]))
	print float(line[0])
	print float(line[1])
	print float(line[2])
	print float(line[3])
	print float(line[4])
	print float(line[5])
	print float(line[6])
	plt.plot(x,y)
	plt.show()
