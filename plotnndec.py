import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
cxaxis=[]
cyaxis=[]
czaxis=[]

epochs=[1000,2000,3000,4000,5000]
bsize=[100,200,300,400,500]
oepochs=[100,500,1000,2000,2500]
obsize=[5,10,30,50,100]
mainarr=[]
with open('inpnndec.txt') as f:
	lines=f.readlines()
	length=len(lines)
	for i in range(length):
		row=lines[i]
		cols=row.split(',')
		for j in range(len(cols)):
			mainarr.append([bsize[i],epochs[j],float(cols[j].strip())])
			cxaxis.append(bsize[i])
			cyaxis.append(epochs[j])
			czaxis.append(float(cols[j].strip())/100)
xaxis=np.array(cxaxis)
yaxis=np.array(cyaxis)
hist, xedges, yedges = np.histogram2d(xaxis, yaxis,bins=4)
xpos, ypos = np.meshgrid(xedges-50, yedges-500)
xpos = xaxis.flatten('F')-50
ypos = yaxis.flatten('F')-500
zpos= np.zeros_like(xpos)+0.60
print xpos.shape
print ypos.shape
print zpos.shape
# Construct arrays with the dimensions for the 16 bars.
dx = 100 * np.ones_like(zpos)
dy = 1000 * np.ones_like(zpos)
dz = np.array(czaxis).flatten()-0.60
colors = ['r','g','b']
surf = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='y', zsort='average')
ax.set_xlim3d(50, 600)
ax.set_ylim3d(500,6000)
ax.set_zlim3d(0.60,0.90)

plt.show()
