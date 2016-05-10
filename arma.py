# -*- coding: utf-8 -*-
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

#Uses ffmpeg to generate the movie! Install ffmpeg to use this script.

#
def arma(past_values, future_values, ai):
	values = []
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
	values = []
	
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
	print("**********************************************************************************")
	print("* Usage: arma.py [dictionary_file_name] [number_of_frames(past and future)]      *")
	print("**********************************************************************************")
	sys.exit(0)	


print("**********************************************************************************")
print("* ARMA FILTER                                        by Virginia O. Andersson    *")
print("**********************************************************************************")

file_name = sys.argv[1]
values = int(sys.argv[2])


fr = open(file_name, 'rb')
dictionary = pickle.load(fr)

names = file_name.split('_')
print names	

future_values = []
past_values = []

print dictionary.keys()

p = []

# Attaching 3D axis to the figure
fig1 = plt.figure()
ax = p3.Axes3D(fig1)

ax.set_xlim3d([-1, 1])
ax.set_xlabel('X')

ax.set_zlim3d([-1, 1])
ax.set_zlabel('Y')

ax.set_ylim3d([0, 3.5])
ax.set_ylabel('Z')

ax.set_title('Subject')

ims = []

ai = 1/float(2*values)

for key in sorted(range(values+1, len(dictionary.keys())-values)):
	for segment in range(0,20):
		for i in range(values,0,-1):
			past_values.append([float(dictionary[key-i][segment][1]), float(dictionary[key-i][segment][2]), float(dictionary[key-i][segment][3])])
		for i in range(1,values+1,1):
			future_values.append([float(dictionary[key+i][segment][1]), float(dictionary[key+i][segment][2]), float(dictionary[key+i][segment][3])])
		
		c = arma3D(past_values, future_values, ai)

		past_values =[]
		future_values =[]
		
		p.append(c)

	f1 = ax.plot([p[0][0], p[1][0], p[2][0], p[4][0], p[6][0], p[8][0], p[6][0], p[4][0], p[2][0], p[1][0], p[3][0], p[5][0], p[7][0], p[9][0], p[7][0], p[5][0], p[3][0], p[1][0], p[10][0], p[11][0], p[12][0], p[14][0], p[16][0], p[18][0], p[16][0], p[14][0], p[12][0], p[10][0], p[11][0], p[13][0],p[15][0],p[17][0], p[19][0]],
	[p[0][2], p[1][2], p[2][2], p[4][2], p[6][2], p[8][2], p[6][2], p[4][2], p[2][2], p[1][2], p[3][2], p[5][2], p[7][2], p[9][2], p[7][2], p[5][2], p[3][2], p[1][2], p[10][2], p[11][2], p[12][2], p[14][2], p[16][2], p[18][2], p[16][2], p[14][2], p[12][2], p[10][2], p[11][2], p[13][2],p[15][2],p[17][2], p[19][2]],
	[p[0][1], p[1][1], p[2][1], p[4][1], p[6][1], p[8][1], p[6][1], p[4][1], p[2][1], p[1][1], p[3][1], p[5][1], p[7][1], p[9][1], p[7][1], p[5][1], p[3][1], p[1][1], p[10][1], p[11][1], p[12][1], p[14][1], p[16][1], p[18][1], p[16][1], p[14][1], p[12][1], p[10][1], p[11][1], p[13][1],p[15][1],p[17][1], p[19][1]], 'bo-')

	ims.append(f1)
	
	p = []
	

#Creating the Animation object
ani = animation.ArtistAnimation(fig1, ims, interval=50, blit=True,repeat_delay=1000)

ani.save('dynamic_images.mp4')

plt.show()

fr.close()

