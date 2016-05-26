import convertCoordinate as cc
import armaFilter as af
import dynamicAnalysis as da
import staticAnalysis as sa
import classifier as classifier
import randomSelect as rs
import os

srcPath = ["/RawGaitDataset",
			"/FilteredGaitDataset",
			"/TrainDataset/TrainGaitDataset",
			"/TestDataset/TestGaitDataset"]

dstPath = ["/FilteredGaitDataset",
			"/TrainDataset/TrainGaitDataset",
			"/TestDataset/TestGaitDataset",
			"/TrainDataset/TrainDynamicDataset",
			"/TestDataset/TestDynamicDataset",
			"/TrainDataset/TrainStaticDataset",
			"/TestDataset/TestStaticDataset"]
if __name__ == "__main__":
	homedir = os.getcwd()
	#filt = af.Filter(homedir+srcPath[0],homedir+dstPath[0])
	#filt.data_process()
	#select = rs.RandomSelect(homedir+srcPath[1],homedir+dstPath[1],homedir+dstPath[2])
	#select.data_process()
	dy = da.dynamicAnalysis(homedir+srcPath[2],homedir+dstPath[3],homedir+srcPath[3],homedir+dstPath[4])
	dy.data_process()