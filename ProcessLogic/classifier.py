import os
import numpy as np
import warnings

def polyFunction(x, a1, a2, a3, a4, a5, a6, a7):
    return a7 * x ** 7 + a6 * x ** 6 + a5 * x ** 5 + a4 * x ** 4 + a3 * x ** 3 + a2 * x ** 2 + a1 * x

class Classifier:
	def __init__(self,path="/Users/niko/Documents/KinectGaitRecognition"):
		self.trainStaticPath = path+"/Dataset/TrainDataset/TrainStaticDataset"
		self.testStaticPath = path+"/Dataset/TestDataset/TestStaticDataset"
		self.trainDynamicPath = path+"/Dataset/TrainDataset/TrainDynamicDataset"
		self.testDynamicPath = path+"/Dataset/TestDataset/TestDynamicDataset"
		self.trainStaticData = []
		self.testStaticData = []
		self.trainDynamicData = []
		self.testDynamicData = []
		self.trainStaticList = []
		self.trainDynamicList = []
		self.testStaticList = []
		self.testDynamicList= []
		self.testList = []
		self.staticResult = []
		self.dynamicResult = []
		self.fusionResult = []

		self.staticDistance = []
		self.dynamicDistance = []

	def dataProcess(self):
		self.staticClassify()
		self.dynamicClassify()
		self.fusionClassify()
		self.showResult()

	def clear(self):
		self.testData = []

	def listdirNohidden(self,path):
		for f in os.listdir(path):
			if not f.startswith('.'):
				yield f

	def readStaticData(self):
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

	def staticClassify(self):
		self.readStaticData()
		#neigh = NearestNeighbors(n_neighbors = 2)
		trainData = self.trainStaticData
		testData = self.testStaticData
		#neigh.fit(trainData)
		#result = neigh.kneighbors(testData,return_distance=False)
		result = self.knnStatic(trainData,testData,2)
		self.staticResult = []
		for r in result:
			countList = {}
			for item in r:
				i = self.trainStaticList[item]
				if countList.has_key(i):
					countList[i] = countList[i] + 1
				else:
					countList[i] = 1
			maximun = 0
			for i in countList:
				if countList[i] > maximun:
					maximun = countList[i]
					index = i
			self.staticResult.append(index)
		# print "static result:", self.staticResult
		count = 0
		for i in range(len(self.staticResult)):
			if self.staticResult[i] == self.testStaticList[i]:
				count += 1
		return count,float(count)/len(self.testStaticList),len(self.testStaticList),self.staticResult
		#print self.staticResult
	def knnStatic(self,trainData,testData,n):
		result = []
		for testPsn in range(len(testData)):
			disList = []
			mapper = {}
			for trainPsn in range(len(trainData)):
				dis = self.euclidian(testData[testPsn],trainData[trainPsn])
				mapper[trainPsn] = dis
				disList.append(dis)
			disList = sorted(disList)
			
			temp = []
			for i in disList[:n]:
				for j in mapper:
					if mapper[j] == i:
						temp.append(j)
			result.append(temp)
		return result

	def euclidian(self,xs,ys):
		if len(xs) != len(ys):
			print "dimension error"
			return 0
		x = np.array(xs)
		y = np.array(ys)
		dim = len(xs)
		dif = x-y
		for i in range(dim):
			dis = np.sqrt(sum((x-y)**2))
		return dis

	def readDynamicData(self):
		trainPath = self.trainDynamicPath
		testPath = self.testDynamicPath
		trainDataFiles = self.listdirNohidden(trainPath)
		testDataFiles = self.listdirNohidden(testPath)

		trainFiles = []
		for files in trainDataFiles:
			trainFiles.append(files)
		# sorted(trainFiles,key = lambda x: int(x.replace(".txt",'')))
		testFiles = []
		for files in testDataFiles:
			testFiles.append(files)
		# sorted(testFiles,key = lambda x: int(x.replace(".txt",'')))

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
				for j in range(len(d)):
					temp.append(float(d[j]))
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

	def dynamicClassify(self):
		self.readDynamicData()
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
		result = self.knnDynamic(trainData,testData,2)
		self.dynamicResult = []
		
		for r in result:
			countList = {}
			for item in r:
				i = self.trainDynamicList[item]
				if countList.has_key(i):
					countList[i] = countList[i] + 1
				else:
					countList[i] = 1
			maximun = 0
			for i in countList:
				if countList[i] > maximun:
					maximun = countList[i]
					index = i
			self.dynamicResult.append(index)
		#print len(self.result)
		# print self.dynamicResult
		# print "dynamic result:",self.dynamicResult
		count = 0
		for i in range(len(self.dynamicResult)):
			if self.dynamicResult[i] == self.testDynamicList[i]:
				count += 1
		return count,float(count)/len(self.testDynamicList),len(self.testDynamicList),self.dynamicResult

	def knnDynamic(self,trainData,testData,n):
		result = []
		for testPsn in range(len(testData)):
			disList = []
			mapper = {}
			for trainPsn in range(len(trainData)):
				dis = self.dtw(testData[testPsn],trainData[trainPsn])
				mapper[trainPsn] = dis
				disList.append(dis)
			disList = sorted(disList)
			
			temp = []
			for i in disList[:n]:
				for j in mapper:
					if mapper[j] == i:
						temp.append(j)
			result.append(temp)
		return result

	def fusionClassify(self):
		self.readStaticData()
		self.readDynamicData()
		#neigh = NearestNeighbors(n_neighbors = 2)
		trainStaticData = self.trainStaticData
		testStaticData = self.testStaticData
		trainDynamicData = np.array(self.trainDynamicData)
		trainDynamicData = trainDynamicData.reshape(trainDynamicData.shape[0],-1)
		testDynamicData = np.array(self.testDynamicData)
		testDynamicData = testDynamicData.reshape(testDynamicData.shape[0], -1)
		#neigh.fit(trainData)
		#result = neigh.kneighbors(testData,return_distance=False)
		result = self.knnFusion(trainDynamicData,testDynamicData,trainStaticData,testStaticData,2)
		self.fusionResult = []
		for r in result:
			countList = {}
			for item in r:
				i = self.trainStaticList[item]
				if countList.has_key(i):
					countList[i] = countList[i] + 1
				else:
					countList[i] = 1
			maximun = 0
			for i in countList:
				if countList[i] > maximun:
					maximun = countList[i]
					index = i
			self.fusionResult.append(index)
		# print "fusion result:", self.fusionResult
		count = 0
		for i in range(len(self.fusionResult)):
			if self.fusionResult[i] == self.testStaticList[i]:
				count += 1
		return count,float(count)/len(self.testStaticList),len(self.testStaticList),self.fusionResult

	def knnFusion(self,trainDynamicData,testDynamicData,trainStaticData,testStaticData,n):
		result = []
		trainNum = len(trainDynamicData)
		testNum = len(testDynamicData)

		dynamicList = []
		staticList = []
		for testPsn in range(testNum):
			dList = []
			sList = []
			for trainPsn in range(trainNum):
				dynamicDistance = self.dtw(testDynamicData[testPsn],trainDynamicData[trainPsn])
				staticDistance = self.euclidian(testStaticData[testPsn],trainStaticData[trainPsn])
				# mapper[trainPsn] = dis
				dList.append(dynamicDistance)
				sList.append(staticDistance)
			dynamicList.append(dList)
			staticList.append(sList)
		dMax = max([max(i) for i in dynamicList])
		dMin = min([min(i) for i in dynamicList])
		sMax = max([max(i) for i in staticList])
		sMin = min([min(i) for i in staticList])

		dData = np.array(dynamicList)
		sData = np.array(staticList)
		dData = dData/(dMax-dMin)
		sData = sData/(sMax-sMin)

		dis = (dData + sData).tolist()

		for psn in dis:
			mapper = {}
			tempList = psn
			for i in range(len(psn)):
				mapper[i] = psn[i]
			tempList = sorted(tempList)
			tempBuffer = []
			for i in tempList[:n]:
				for j in mapper:
					if mapper[j] == i:
						tempBuffer.append(j)
			result.append(tempBuffer)
		return result

	def showResult(self):
#dynamic success number:
		count = 0
		for i in range(len(self.dynamicResult)):
			if self.dynamicResult[i] == self.testDynamicList[i]:
				count += 1
		print "dynamic success:"
		print count
#static success number:
		count = 0
		for i in range(len(self.staticResult)):
			if self.staticResult[i] == self.testStaticList[i]:
				count += 1
		print "static success:"
		print count
#fusion success number:
		count = 0
		for i in range(len(self.fusionResult)):
			if self.fusionResult[i] == self.testStaticList[i]:
				count += 1
		print "fusion success:"
		print count

	def dtw(self,xs,ys,count=[0]):
		# print "dtw:",count[0]
		count[0] += 1
		xs = xs.reshape(-1,7)
		ys = ys.reshape(-1,7)
		dist = 0
		for i in range(len(xs)):
			xdata = np.arange(30)
			ytrainData = polyFunction(xdata,xs[i][0],xs[i][1],xs[i][2],xs[i][3],xs[i][4],xs[i][5],xs[i][6])
			ytestData = polyFunction(xdata,ys[i][0],ys[i][1],ys[i][2],ys[i][3],ys[i][4],ys[i][5],ys[i][6])
			dist += self.dtwSingle(ytrainData,ytestData)
		#print "calciulateing..."
		return dist

	def dtwSingle(self,x, y):
		# print x.shape
		# print y.shape

		#dist = manhattan_distances
	 	r, c = len(x), len(y)
		D0 = np.zeros((r + 1, c + 1))
		D0[0, 1:] = np.inf
		D0[1:, 0] = np.inf
		D1 = D0[1:, 1:]
		for i in range(r):
			for j in range(c):
				#print "x[i]:",x[i],"y[i]",y[i]
				D1[i, j] = abs(x[i] - y[j])
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
	classifier.dataProcess()