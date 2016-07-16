# import sys
# sys.path.append("/Users/niko/Documents/KinectGaitScripts/")
import os
import string
import shutil
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import platform
from scipy.integrate import trapz
from scipy.optimize import curve_fit
from GaitData import GaitData

def fourierFunction1(x, w, a0, a1, b1):
    return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w)


def fourierFunction2(x, w, a0, a1, b1, a2, b2):
    return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w) + a2 * np.cos(2 * x * w) + b2 * np.sin(2 * x * w)


def fourierFunction3(x, w, a0, a1, b1, a2, b2, a3, b3):
    return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w) + a2 * np.cos(2 * x * w) + b2 * np.sin(2 * x * w) + a3 * np.cos(
        3 * x * w) + b3 * np.sin(3 * x * w)


def fourierFunction4(x, w, a0, a1, b1, a2, b2, a3, b3, a4, b4):
    return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w) + a2 * np.cos(2 * x * w) + b2 * np.sin(2 * x * w) + a3 * np.cos(
        3 * x * w) + b3 * np.sin(3 * x * w) + a4 * np.cos(4 * x * w) + b4 * np.sin(4 * x * w)


def fourierFunction5(x, w, a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5):
    return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w) + a2 * np.cos(2 * x * w) + b2 * np.sin(2 * x * w) + a3 * np.cos(
        3 * x * w) + b3 * np.sin(3 * x * w) + a4 * np.cos(4 * x * w) + b4 * np.sin(4 * x * w) + a5 * np.cos(
        5 * x * w) + b5 * np.sin(5 * x * w)


def fourierFunction6(x, w, a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6):
    return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w) + a2 * np.cos(2 * x * w) + b2 * np.sin(2 * x * w) + a3 * np.cos(
        3 * x * w) + b3 * np.sin(3 * x * w) + a4 * np.cos(4 * x * w) + b4 * np.sin(4 * x * w) + a5 * np.cos(
        5 * x * w) + b5 * np.sin(5 * x * w) + a6 * np.cos(6 * x * w) + b6 * np.sin(6 * x * w)


def fourierFunction7(x, w, a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7):
    return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w) + a2 * np.cos(2 * x * w) + b2 * np.sin(2 * x * w) + a3 * np.cos(
        3 * x * w) + b3 * np.sin(3 * x * w) + a4 * np.cos(4 * x * w) + b4 * np.sin(4 * x * w) + a5 * np.cos(
        5 * x * w) + b5 * np.sin(5 * x * w) + a6 * np.cos(6 * x * w) + b6 * np.sin(6 * x * w) + a7 * np.cos(
        7 * x * w) + b7 * np.sin(7 * x * w)

def polyFunction(x, a0, a1, a2, a3, a4, a5, a6, a7):
    return a7 * x ** 7 + a6 * x ** 6 + a5 * x ** 5 + a4 * x ** 4 + a3 * x ** 3 + a2 * x ** 2 + a1 * x + a0

class DynamicAnalyzer:
    'this class is to fitting the curve of walk'

    def __init__(self,path,angleList={}):
        self.gaitData = GaitData()
        self.points = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        self.srcTrainPath = path + "/Dataset/TrainDataset/TrainGaitDataset"
        self.dstTrainPath = path + "/Dataset/TrainDataset/TrainDynamicDataset"
        self.srcTestPath = path + "/Dataset/TestDataset/TestGaitDataset"
        self.dstTestPath = path + "/Dataset/TestDataset/TestDynamicDataset"
        self.ndfit = 1
        # self.wekaFile = open(wekaFilePath, 'w')
        # self.wekaFile.write("@relation fourier_fitting\n")
        # self.trainData = open(self.dstPath + '/dynamic.txt', 'w')
        # for i in range(8):
        #     self.wekaFile.write("@attribute ")
        #     self.wekaFile.write(str(i) + " ")
        #     self.wekaFile.write("numeric\n")
        # self.wekaFile.write("@attribute identification numeric\n@data\n")
        self.frequecy = []
        self.N = None
        self.anglesMap = {
                "srkrar":[self.gaitData.shoulderRight,self.gaitData.kneeRight,self.gaitData.ankleRight],
                "srklal":[self.gaitData.shoulderRight,self.gaitData.kneeLeft,self.gaitData.ankleLeft],
                "slkrar":[self.gaitData.shoulderLeft,self.gaitData.kneeRight,self.gaitData.ankleRight],
                "slklal":[self.gaitData.shoulderLeft,self.gaitData.kneeLeft,self.gaitData.ankleLeft],
                "hrklal":[self.gaitData.hipRight,self.gaitData.kneeLeft,self.gaitData.ankleLeft],
                "hlkrar":[self.gaitData.hipLeft,self.gaitData.kneeRight,self.gaitData.ankleRight],
                "krhlal":[self.gaitData.kneeRight,self.gaitData.hipLeft,self.gaitData.ankleLeft],
                "klhrar":[self.gaitData.kneeLeft,self.gaitData.hipRight,self.gaitData.ankleRight],
                "arhlkl":[self.gaitData.ankleRight,self.gaitData.hipLeft,self.gaitData.kneeLeft],
                "alhrkr":[self.gaitData.ankleLeft,self.gaitData.hipRight,self.gaitData.kneeRight],
                "hcsckl":[self.gaitData.hipCenter,self.gaitData.shoulderCenter,self.gaitData.kneeLeft],
                "hcsckr":[self.gaitData.hipCenter,self.gaitData.shoulderCenter,self.gaitData.kneeRight],
                }

        self.angleList = angleList

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

    # self.sparseMatric = []

    #def __del__(self):
        # self.wekaFile.close()
        #self.trainData.close()

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

    def listdirNohidden(self, path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    def clear(self):
        self.points = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    def dataProcess(self):
        self.mode = "Train Dynamic:"
        self.dynamicAnalysis(self.srcTrainPath,self.dstTrainPath)

        self.mode = "Test  Dynamic:"
        self.dynamicAnalysis(self.srcTestPath,self.dstTestPath)

    def dynamicAnalysis(self,srcPath,dstPath):
        # The first step is to read data from files
        personDirectorsPath = srcPath
        personDirectors = self.listdirNohidden(personDirectorsPath)
        for personDirector in personDirectors:
            personDirectorPath = personDirectorsPath + '/' + personDirector
            if not os.path.isdir(personDirectorPath):
				continue
            print self.mode,personDirector
            personFiles = self.listdirNohidden(personDirectorPath)
            # caculate the number of nth file
            # fPoint = 0
            for personFile in personFiles:
                personFilePath = personDirectorPath + '/' + personFile
                self.clear()
                self.read_data(personFilePath)
                self.gaitData.setId(int(personDirector.replace("Person", '')))
                # fPoint += 1
                # The second step is to deal with the data
                # test_joints = np.array([self.gaitData.knee_right, self.gaitData.knee_left])
                self.drawAngleCurve(dstPath)

                # speed_joints = [self.gaitData.hip_center, self.gaitData.shoulder_center, self.gaitData.head,
                #                 self.gaitData.spine]
            # self.drawSpeedCurve(speed_joints)
            # self.drawDirection()
            # append the number to sparse matric
            # self.sparseMatric.append(fPoint)

            # self.writeSparseMatric()

    def read_data(self, personFilePath):
        person = open(personFilePath)
        personData = person.readlines()
        if len(personData) == 0:
            print "The data of file is empty:"
            print personFilePath
            return
        length = len(personData)
        for item in range(0, length / 20):
            for seg in range(0, 20):
                temp = personData[item * 20 + seg].split(";")
                point = [string.atof(temp[1]), string.atof(temp[2]), string.atof(temp[3].replace("\n", ''))]
                self.points[seg].append(point)
        person.close()

    def setNdfit(self, nd):
        self.ndfit = nd

    def curveFitting(self, x, angle):
        # p0 = np.ones(2*(self.ndfit+1))
        # p0 = [0,0,0,0,0,0.010101584095144]
        # p0 = [0]*(2*(self.ndfit+1) - 1)
        # p0.append(0.01*rd.random())
        # print p0
        # p0 = np.array(p0)
        p0 = 0.01 * np.random.normal(size=2 * (self.ndfit + 1))
        if self.ndfit == 1:
            return curve_fit(fourierFunction1, x, angle, p0, maxfev=5000)
        elif self.ndfit == 2:
            return curve_fit(fourierFunction2, x, angle, p0, maxfev=5000)
        elif self.ndfit == 3:
            return curve_fit(fourierFunction3, x, angle, p0, maxfev=5000)
        elif self.ndfit == 4:
            return curve_fit(fourierFunction4, x, angle, p0, maxfev=5000)
        elif self.ndfit == 5:
            return curve_fit(fourierFunction5, x, angle, p0, maxfev=5000)
        elif self.ndfit == 6:
            return curve_fit(fourierFunction6, x, angle, p0, maxfev=5000)
        elif self.ndfit == 7:
            return curve_fit(fourierFunction7, x, angle, p0, maxfev=5000)
        elif self.ndfit == -1:
            return curve_fit(polyFunction, x, angle)
        else:
            print "size must be between 1 and 7"
            return False

    def applyFunction(self, xdata, arg):
        if self.ndfit == 1:
            return fourierFunction1(xdata, arg[0], arg[1], arg[2], arg[3])
        elif self.ndfit == 2:
            return fourierFunction2(xdata, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5])
        elif self.ndfit == 3:
            return fourierFunction3(xdata, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7])
        elif self.ndfit == 4:
            return fourierFunction4(xdata, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9])
        elif self.ndfit == 5:
            return fourierFunction5(xdata, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9],
                                  arg[10], arg[11])
        elif self.ndfit == 6:
            return fourierFunction6(xdata, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9],
                                  arg[10], arg[11], arg[12], arg[13])
        elif self.ndfit == 7:
            return fourierFunction7(xdata, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9],
                                  arg[10], arg[11], arg[12], arg[13], arg[14], arg[15])
        elif self.ndfit == -1:
            return polyFunction(xdata, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7])
        else:
            print "size error"

    def getModulo(self, vector):
        x = vector[:, 0] ** 2
        y = vector[:, 1] ** 2
        z = vector[:, 2] ** 2
        return np.sqrt(x + y + z)

    def calculateAngle(self, joint):
        data = np.array(self.points)
        a1 = data[joint[1]] - data[joint[0]]
        a2 = data[joint[2]] - data[joint[0]]
        modulo = self.getModulo(a1) * self.getModulo(a2)
        dotMulti = a1 * a2
        dotMulti = dotMulti[:, 0] + dotMulti[:, 1] + dotMulti[:, 2]
        cosTheta = (dotMulti) / modulo
        angle = np.arccos(cosTheta)
        return angle

    # extra periods from the vedio sequence
    def extractPeriods(self, angle):
        sign = np.sign(np.diff(angle))
        ddiff = np.diff(sign)
        # plt.plot(np.arange(len(angle)),angle)

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
        # find the first crest
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
                j = j + 1
            if j >= np.size(ddiff):
                break
            temp.append(angle[j])
            temp.append(angle[j + 1])
            i = j + 1
            periods.append(temp)
        return periods

    def periodsFilter(self, periods):
        i = 0
        while i < len(periods):
            pMax = max(periods[i])
            pMin = min(periods[i])
            if (abs(periods[i][0] - periods[i][len(periods[i]) - 1]) >= (9. / 10.) * (pMax - pMin)):
                del periods[i]
                i -= 1
            i += 1
        # one period have at least 30 frames
        i = 0
        while i < len(periods):
            if ((len(periods[i]) < 30) or (len(periods[i]) > 40)):
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
    # periods = periods[x_index]
    # periods = periods[y_index]
    # print periods

    def drawAngleCurve(self,dstPath):
        for m in self.angleList:
            angle = self.calculateAngle(self.anglesMap[m])
            self.meanFilter(angle,17)
            periods = self.extractPeriods(angle)
            periods = self.periodsFilter(periods)
            # fourier curve fitting
            # for i in range(len(periods)):
            # 	print periods[i]
            # 	p = np.array(periods[i])
            # 	self.set_ndfit(3)
            # 	x = np.arange(len(p))
            # 	popt, pcov = self.curve_fitting(x,p)
            # 	ydata = self.apply_function(x,popt)
            # 	plt.plot(x,p)
            # 	plt.plot(x,ydata)
            if len(periods) == 0:
                trainData = open(dstPath+'/'+str(m)+".txt",'a')
                for i in range(7):
                    trainData.write("0.,")
                trainData.write(str(self.gaitData.getId()) + '\n')
                trainData.close()
                continue
            self.setNdfit(-1)
            # p = periods[0]
            # calciulate average
            minimun = 100
            for i in range(len(periods)):
                if len(periods[i]) < minimun:
                    minimun = len(periods[i])
                    index = i
            period = []
            for i in range(minimun):
                avg = 0
                for p in periods:
                    avg += p[i]
                period.append(avg / len(periods))
            #print period
            x = np.arange(len(period))

            popt, pcov = self.curveFitting(x, np.array(period))
            # ydata = self.apply_function(x,popt)
            # # plt.plot(x,np.array(p))
            # plt.plot(x,ydata)
            # plt.plot(x,period)

            # write fit parameter(except a0) to classfy
            trainData = open(dstPath+'/'+str(m)+".txt",'a')
            for i in range(1, len(popt)):
                # self.wekaFile.write(str(popt[i]) + ',')
                trainData.write(str(popt[i]) + ',')
            # self.wekaFile.write(str(self.gaitData.getId()) + '\n')
            trainData.write(str(self.gaitData.getId()) + '\n')
            trainData.close()
            # fourier transform
            # amplitude, phase = self.fourierTransform(angle)
            # write the amplitude and phase as the parameter of recognition
            # for alp in amplitude:
            # 	self.wekaFile.write(str(alp))
            # 	self.wekaFile.write(str(','))
            # for phs in phase:
            # 	self.wekaFile.write(str(phs))
            # 	self.wekaFile.write(str(','))
            # machine learning
            # for i in frequency:
            # 	self.wekaFile.write(str(i)+' ')
            # self.wekaFile.write('\n')
            # self.wekaFile.write()
            # self.wekaFile.write(str(self.gaitData.getId())+'\n')

            # only for test
            # matlabFilePath = "/Users/niko/Documents/KinectGaitScripts/TestOnlyData/MatlabDataset/1.txt"
            # matlabFile = open(matlabFilePath, 'w')
            # for i in range(len(angle) - 1):
            #     matlabFile.write(str(angle[i]) + ' ')
            # matlabFile.write(str(angle[len(angle) - 1]))
            # matlabFile.close()
        #plt.show()

    # the speed doesn't work
    def drawSpeedCurve(self, joints):
        data = np.array(self.points)
        limbSpeeds = []
        for limb in joints:
            temp = []
            for t in range(1, len(data[0])):
                dis = data[limb][t] - data[limb][t - 1]
                temp.append(np.sqrt(dis[0] ** 2 + dis[1] ** 2 + dis[2] ** 2))
            limbSpeeds.append(temp)
        limbSpeedsData = np.array(limbSpeeds)

        size = len(limbSpeedsData[0])
        avg = np.zeros(size)
        for i in range(len(lists)):
            avg += limbSpeedsData[i]
        avg = avg / size
        self.meanFilter(avg, 21)
        #plt.plot(np.arange(size), avg.tolist())
        #plt.show()

    # the angle seems not to work so well
    def drawDirection(self):
        data = np.array(self.points)
        xNew = data[self.gaitData.hipLeft] - data[self.gaitData.hipRight]
        yNew = data[self.gaitData.shoulderCenter] - data[self.gaitData.hipCenter]
        zNew = np.cross(xNew, yNew)
        cosTheta = zNew[:, 2] / self.getModulo(zNew)
        self.medianFilter(cosTheta, 21)
        #plt.plot(np.arange(len(cos_theta.tolist())), cos_theta.tolist())
        #plt.show()

    def drawPath(self):
        return 0

    def meanFilter(self, array, size):
        if size % 2 == 0:
            print "size must be odd"
            return
        pad = (size - 1) / 2
        arrayCopy = np.copy(array)
        for i in range(pad, len(array) - pad - 1):
            sum = arrayCopy[i]
            for j in range(1, pad + 1):
                sum += arrayCopy[i - j] + arrayCopy[i + j]
            array[i] = sum / size

    def medianFilter(self, array, size):
        if size % 2 == 0:
            print "size must be odd"
            return
        pad = (size - 1) / 2
        for i in range(pad, len(array) - pad - 1):
            temp = np.sort(array[i - pad:i + pad + 1])
            #print temp
            array[i] = temp[pad]

    def nextpow2(self, n):
        mf = np.log2(n)
        mi = np.ceil(mf)
        return int(2 ** mi)

    def fourierTransform(self, angle):
        Fs = 256
        m = np.shape(angle)
        m = 2048
        # plt.plot(np.arange(m), angle)
        N = self.nextpow2(m)

        # the real amplitude is the result of devided by N and multiplied  by 2.
        # the N bigger, accuracy is better
        Y = np.fft.fft(angle, N) / N * 2
        # frequecy
        f = Fs / N * np.linspace(0, 1, N - 1)
        # amplitude
        A = abs(Y)
        # phase
        P = np.angle(Y)
        # ***********show the image**************
        # plt.plot(np.arange(N/2),A[0:N/2])
        # plt.show()
        if self.N != None and self.N > 2048:
            print "not same\n"
        self.N = N
        return A, P

if __name__ == "__main__":
    # train
    lists = ['srkrar','srklal']
    import sys
    homedir = sys.path[0]
    p = platform.platform().split('-')[0]
    if p == "Windows":
        seperator = "\\"
    elif p == "Darwin":
        seperator = "/"
    i = homedir.rfind(seperator)
    homedir = homedir[0:i]
    d = DynamicAnalyzer(homedir,lists)
    d.dataProcess()
