# -*- coding: utf-8 -*-

import pickle
import sys
import os
import numpy as np

def arma(past_values, future_values, ai):
	values= []
	for v in future_values:
		values.append(v)
		
	for v in past_values:
		values.append(v)
	
	avgmean = 0
	
	for v in values:		
		avgmean = float(ai * v) + avgmean
	
	return avgmean
#

def arma3D(past_values, future_values, ai):
	values= []
	
	for v in future_values:
		values.append(v)
	for v in past_values:
		values.append(v)
	
	avgmeanX = 0	
	avgmeanY = 0
	avgmeanZ = 0
	
	for x,y,z in values:
		avgmeanX = float(ai * x) + avgmeanX
		avgmeanY = float(ai * y) + avgmeanY
		avgmeanZ = float(ai * z) + avgmeanZ
	
	return [avgmeanX, avgmeanY, avgmeanZ]
		
#

if(len(sys.argv) < 2):
	print("*******************************************************************************************************")
	print("* Usage: apply_arma_filter.py [dictionaries_files_list_file] [number_of_frames(1 N value both for past and future)]*")
	print("*******************************************************************************************************")
	sys.exit(0)	


print("**********************************************************************************")
print("* ARMA FILTER                                        by Virginia O. Andersson    *")
print("**********************************************************************************")

#nome_arquivo = sys.argv[1]

#dictionaies list name example: dictionaries_Nvalue_group1.list
#dictionaries_10_grupo0.list

group_files =  sys.argv[1].split("_")
group_files[2] = group_files[2].replace(".list",'')

past_values = []
future_values = []

frame = []
p = []

descriptors = ['Head', 'Shoulder-Center', 'Shoulder-Right', 'Shoulder-Left', 'Elbow-Right', 'Elbow-Left', 'Wrist-Right', 'Wrist-Left',
			   'Hand-Right', 'Hand-Left', 'Spine', 'Hip-centro', 'Hip-Right', 'Hip-Left', 'Knee-Right', 'Knee-Left',
			   'Ankle-Righ', 'Ankle-Left', 'Foot-Right', 'Foot-Left']
filtered = {}

#N value = past and future number of frames to look backwards and aftwards
value = int(sys.argv[2])
ai = 1/float(2*value)

#Open file with dictionary list
#Create a dictionaries' files list

print "Reading 3D points dictionaries:"
#if os.name == 'posix':
#	filelst = open("Data/dictionaries.list", 'r')
#if os.name == 'nt':

filelst = open(sys.argv[1], 'r')
dict_files = filelst.readlines()

for lines in dict_files:
	lines = lines.replace('\n','')

	fr = open(lines, 'rb')
	dictionary = pickle.load(fr)
	name_of_file = lines.split('_')
	individual = name_of_file[1]

	print name_of_file, individual

	#fs = open("/Users/niko/Documents/KinectGaitScripts/FilteredGaitDataset/Person001/"+name_of_file[2].replace(".pkl",'')+".txt",'w')

	filteredDataPath = "/Users/niko/Documents/KinectGaitScripts/FilteredGaitDataset/"
	filteredDataPersonPath = filteredDataPath + individual + '/' + name_of_file[2].replace(".pkl",".txt")
	filteredFile = open(filteredDataPersonPath,'w')

	for key in sorted(range(value+1, len(dictionary.keys()) - value + 1)):
		for segment in range(0,20):
			for i in range(value,0,-1):
				past_values.append([float(dictionary[key-i][segment][1]), float(dictionary[key-i][segment][2]), float(dictionary[key-i][segment][3])])
			for i in range(1,value+1,1):
				future_values.append([float(dictionary[key+i][segment][1]), float(dictionary[key+i][segment][2]), float(dictionary[key+i][segment][3])])

			c = arma3D(past_values, future_values, ai)

			past_values =[]
			future_values =[]

			p.append(descriptors[segment])
			[p.append(str(i)) for i in c]
			ps = str(p)
			ps = ps.replace("[",'')
			ps = ps.replace("]",'')+"\n"
			ps = ps.replace("'",'')
			ps = ps.replace(",",';')
			ps = ps.replace(" ",'')

			filteredFile.write(ps)
			#frame.append(p)
			filtered[key] = frame
			p = []
		frame = []
	
	filteredFile.close()
	filtered.clear()

fr.close()
	
