import os
import shutil

homdir = os.getcwd()
trainGaitPath = homdir+"\\TrainDataset\\TrainGaitDataset"
if (os.path.exists(trainGaitPath)):
    print "1"
    shutil.rmtree(trainGaitPath)
    os.mkdir(trainGaitPath)
else:
    print "2"
    os.mkdir(trainGaitPath)
filterFilePath = homdir+"\\FilteredGaitDataset"
files = os.listdir(filterFilePath)
for f in files:
    fpath = filterFilePath + "\\"+f
    print fpath
    dstGaitPath = trainGaitPath+"\\"+f
    shutil.copytree(fpath,dstGaitPath)