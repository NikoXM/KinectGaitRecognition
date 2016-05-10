import os
'''
for i in range(2,165):
    p = "/Users/niko/Documents/KinectGaitScripts/FilteredGaitDataset/Person"+"{:0>3d}".format(i)
    os.mkdir(p)
'''
#this operation is to write path of raw dataset
'''
rawDataListFile = open("/Users/niko/Documents/KinectGaitScripts/Data/files.list",'w')
path = "/Users/niko/Documents/KinectGaitScripts/KinectGaitRawDataset/Person"
for i in range(1,165):
	file = "{:0>3d}".format(i)
	path_file = path + file
	files = os.listdir(path_file)
	for name in files:
		sname = name.replace("'",'')
		path_file_name = path_file +'/'+ sname + '\n'
		rawDataListFile.write(path_file_name)

rawDataListFile.close()
'''
#this operation is to write path of dictionary of raw dataset
dictDataListFile = open("/Users/niko/Documents/KinectGaitScripts/Data/dict_4_group1.list",'w')
dictDataFilesPath = "/Users/niko/Documents/KinectGaitScripts/Data/Dictionaries"
dictDataFiles = os.listdir(dictDataFilesPath)
for file in dictDataFiles:
	file_name = file.replace("'",'')
	if(cmp(file_name,".DS_Store") == 0):
		continue
	dictDataFile = dictDataFilesPath + '/' + file_name + '\n'
	dictDataListFile.write(dictDataFile)
dictDataListFile.close()