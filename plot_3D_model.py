#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

fileName = sys.argv[1]

data = []
with open(fileName, 'r') as inFile:
	for line in inFile:
		numbers = line.split(",")
		data.append( (int(numbers[0]), int(numbers[1]), float(numbers[2])) )

X = [a[0] for a in data]
Y = [a[1] for a in data]
Z = [a[2] for a in data]

xmin = min(X)
xmax = max(X)
ymin = min(Y)
ymax = max(Y)
zmin = min(Z)
zmax = max(Z)

fig = plt.figure(figsize=[12.5, 12.5], dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=21, azim=36)

ax.set_xlim3d(xmin, xmax)
ax.set_ylim3d(ymin, ymax)
ax.set_zlim3d(zmin, zmax)

surf = ax.plot_trisurf(X, Y, Z, cmap=cm.Blues, linewidth=0.2)

plt.show()
