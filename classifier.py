import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import manhattan_distances
import warnings
warnings.filterwarnings("ignore")

srcPath = ["/Users/niko/Documents/KinectGaitScripts/TrainDataset/trainStaticDataset",
			"/Users/niko/Documents/KinectGaitScripts/TestDataset/testStaticDataset",
			"/Users/niko/Documents/KinectGaitScripts/TrainDataset/trainDynamicDataset",
			"/Users/niko/Documents/KinectGaitScripts/TestDataset/testDynamicDataset"]

def poly_func(x, a1, a2, a3, a4, a5, a6, a7):
    return a7 * x ** 7 + a6 * x ** 6 + a5 * x ** 5 + a4 * x ** 4 + a3 * x ** 3 + a2 * x ** 2 + a1 * x

class Classifier:
	def __init__(self):
		self.trainStaticPath = srcPath[0]
		self.testStaticPath = srcPath[1]
		self.trainDynamicPath = srcPath[2]
		self.testDynamicPath = srcPath[3]
		self.trainStaticData = []
		self.testStaticData = []
		self.trainDynamicData = []
		self.testDynamicData = []
		self.trainStaticList = []
		self.trainDynamicList = []
		self.testStaticList = []
		self.testDynamicList= []
		self.testList = []
		self.static_result = []
		self.dynamic_result = []

	def data_process(self):
		self.static_classify()
		self.dynamic_classify()
		self.show_result()

	def clear(self):
		self.testData = []

	def listdir_nohidden(self,path):
		for f in os.listdir(path):
			if not f.startswith('.'):
				yield f

	def read_static_data(self):
		trainPath = self.trainStaticPath
		testPath = self.testStaticPath
		trainDataFile = open(trainPath+"/static.txt")
		testDataFile = open(testPath+"/static.txt")

		trainData = trainDataFile.readlines()
		testData = testDataFile.readlines()
		self.trainStaticList = []
		for d in trainData:
			d = d.split(',')
			self.trainStaticList.append(d[len(d)-1].replace('\n',''))
			del d[len(d)-1]
			temp = []
			for i in range(len(d)):
				temp.append(float(d[i]))
			self.trainStaticData.append(temp)
		for d in testData:
			d = d.split(',')
			self.testStaticList.append(d[len(d)-1].replace('\n',''))
			del d[len(d)-1]
			temp = []
			for i in range(len(d)):
				temp.append(float(d[i]))
			self.testStaticData.append(temp)
		trainDataFile.close()
		testDataFile.close()

	def read_dynamic_data(self):
		trainPath = self.trainDynamicPath
		testPath = self.testDynamicPath
		trainDataFiles = self.listdir_nohidden(trainPath)
		testDataFiles = self.listdir_nohidden(testPath)

		trainFiles = []
		for file in trainDataFiles:
			trainFiles.append(file)
		sorted(trainFiles,key = lambda x: int(x.replace(".txt",'')))
		testFiles = []
		for file in testDataFiles:
			testFiles.append(file)
		sorted(testFiles,key = lambda x: int(x.replace(".txt",'')))

		# temp = open(self.trainDynamicPath + '/' + trainFiles[0])
		# length = len(temp.readlines())
		# temp.close()

		# listVector = []
		# for i in range(len(trainFiles)):
		# 	listVector.append([])
		# listVectors = []
		# for i in range(length):
		# 	listVectors.append(listVector)
		trainDynamicData = []
		testDynamicData = []
		self.trainDynamicList = []

		for i in range(len(trainFiles)):
			trainDataFile = open(self.trainDynamicPath + '/' + trainFiles[i])
		 	testDataFile = open(self.testDynamicPath + '/' + testFiles[i])
			trainData = trainDataFile.readlines()
		 	testData = testDataFile.readlines()
		 	trainLists = []
			for d in trainData:
				d = d.split(',')
				if i == 0:
					self.trainDynamicList.append(d[len(d)-1].replace('\n',''))
				del d[len(d)-1]
				temp = []
				for j in d:
					temp.append(float(j))
				trainLists.append(temp)
			trainDynamicData.append(trainLists)

			testLists = []
			for d in testData:
				d = d.split(',')
				if i == 0:
					self.testDynamicList.append(d[len(d)-1].replace('\n',''))
				del d[len(d)-1]
				temp = []
				for i in range(len(d)):
					temp.append(float(d[i]))
				testLists.append(temp)
			testDynamicData.append(testLists)

			testDataFile.close()
			trainDataFile.close()
		# for col in range(len(trainDynamicData[0])):
		# 	temp = []
		# 	for row in range(len(trainDynamicData)):
		# 		print row, col
		# 		temp.append(trainDynamicData[row][col])
		# 	self.trainDynamicData.append(temp)
		# print self.trainDynamicData
		# for col in range(len(testDynamicData[0])):
		# 	temp = []
		# 	for row in range(len(testDynamicData[col])):
		# 		temp.append(testDynamicData[row][col])
		# 	self.testDynamicData.append(temp)
		#self.trainDynamicData = trainDynamicData
		#self.testDynamicData = testDynamicData
		self.trainDynamicData = [[r[col] for r in trainDynamicData] for col in range(len(trainDynamicData[0]))] 
		self.testDynamicData = [[r[col] for r in testDynamicData] for col in range(len(testDynamicData[0]))]
		# print self.testDynamicData

	def static_classify(self):
		self.read_static_data()
		neigh = NearestNeighbors(n_neighbors = 2)
		trainData = self.trainStaticData
		testData = self.testStaticData
		neigh.fit(trainData)
		result = neigh.kneighbors(testData,return_distance=False)
		self.static_result = []
		for r in result:
			count_list = {}
			for item in r:
				i = self.trainStaticList[item]
				if count_list.has_key(i):
					count_list[i] = count_list[i] + 1
				else:
					count_list[i] = 1
			maximun = 0
			for i in count_list:
				if count_list[i] > maximun:
					maximun = count_list[i]
					index = i
			self.static_result.append(index)
		#print self.static_result

	def dynamic_classify(self):
		self.read_dynamic_data()
		trainData = np.array(self.trainDynamicData)
		trainData = trainData.reshape(trainData.shape[0],-1)

		testData = np.array(self.testDynamicData)
		testData = testData.reshape(testData.shape[0], -1)
		# print testData

		# trainData = np.zeros([6,35])
		# testData = np.array([np.zeros([35])])

		#trainData = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]])
		#testData = np.array([[1,0]])
		#neigh = NearestNeighbors(n_neighbors=2,metric = df.dtw)
		#neigh.fit(trainData)
		#result = neigh.kneighbors(testData,return_distance = False)
		#result = []
		result = self.knn_dynamic(trainData,testData)
		self.dynamic_result = []
		print result
		print len(result)
		for r in result:
			count_list = {}
			for item in r:
				i = self.trainDynamicList[item]
				if count_list.has_key(i):
					count_list[i] = count_list[i] + 1
				else:
					count_list[i] = 1
			maximun = 0
			for i in count_list:
				if count_list[i] > maximun:
					maximun = count_list[i]
					index = i
			self.dynamic_result.append(index)
		#print len(self.result)
		# print self.dynamic_result

	def knn_dynamic(self,trainData,testData):
		x = np.arange(30)
		print len(trainData)
		print len(testData)
		for testPsn in testData:
			dis = 0
			for trainPsn in trainData:
				dis += self.dtw(testPsn,trainPsn)
			print dis
		return []
	# 	for test in testData:
	# 		dis_list = []
	# 		dis = 0
	# 		for train in trainData:

	def show_result(self):
		# print self.trainList
		idlist = []
		count = 0
		for i in range(len(self.dynamic_result)):
			if self.dynamic_result[i] == self.testDynamicList[i]:
				count += 1
		print "dynamic success:"
		print count
		count = 0
		for i in range(len(self.static_result)):
			if self.static_result[i] == self.testStaticList[i]:
				count += 1
		print "static success:"
		print count

	def dtw(self,xs,ys,count=[0]):
		if(len(xs) == 10):
			return 0
		print count[0]
		count[0] += 1
		xs = xs.reshape(-1,7)
		ys = ys.reshape(-1,7)
		dist = 0
		for i in range(len(xs)):
			xdata = np.arange(30)
			ytrainData = poly_func(xdata,xs[i][0],xs[i][1],xs[i][2],xs[i][3],xs[i][4],xs[i][5],xs[i][6])
			ytestData = poly_func(xdata,ys[i][0],ys[i][1],ys[i][2],ys[i][3],ys[i][4],ys[i][5],ys[i][6])
			dist += self.dtw_single(ytrainData,ytestData)
		#print "calciulateing..."
		return dist

	def dtw_single(self,x, y):
		# print x.shape
		# print y.shape

		dist = manhattan_distances
	 	r, c = len(x), len(y)
		D0 = np.zeros((r + 1, c + 1))
		D0[0, 1:] = np.inf
		D0[1:, 0] = np.inf
		D1 = D0[1:, 1:]
		for i in range(r):
			for j in range(c):
				D1[i, j] = dist([x[i]], [y[j]])
		C = D1.copy()
		for i in range(r):
			for j in range(c):
				D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
		if len(x)==1:
			path = np.zeros(len(y)), range(len(y))
		elif len(y) == 1:
			path = range(len(x)), np.zeros(len(x))
		else:
			path = self._traceback(D0)
		#return D1[-1, -1] / sum(D1.shape), C, D1, path
		return D1[-1, -1] / sum(D1.shape)

	def _traceback(self,D):
	    i, j = np.array(D.shape) - 2
	    p, q = [i], [j]
	    while ((i > 0) or (j > 0)):
	        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
	        if (tb == 0):
	            i -= 1
	            j -= 1
	        elif (tb == 1):
	            i -= 1
	        else: # (tb == 2):
	            j -= 1
	        p.insert(0, i)
	        q.insert(0, j)
	    return np.array(p), np.array(q)

if __name__ == "__main__":
	classifier = Classifier()
	classifier.data_process()