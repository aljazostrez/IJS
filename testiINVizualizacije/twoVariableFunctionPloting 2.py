# from numpy import exp,arange
# from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

# # the function that I'm going to plot
# def z_func(x,y):
#  return (1-(x**2+y**3))*exp(-(x**2+y**2)/2)
 
# x = arange(-3.0,3.0,0.1)
# y = arange(-3.0,3.0,0.1)
# X,Y = meshgrid(x, y) # grid of point
# Z = z_func(X, Y) # evaluation of the function on the grid

# im = imshow(Z,cmap=cm.RdBu) # drawing the function
# # adding the Contour lines with labels
# cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
# clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
# colorbar(im) # adding the colobar on the right
# # latex fashion title
# title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
# show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

#Parameters to set
mu_x = 0
variance_x = 10

mu_y = 0
variance_y = 15

#Create grid and multivariate normal
x = np.linspace(-10,10,500)
y = np.linspace(-10,10,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 10], [10, variance_y]])

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('p(x, y)', fontsize=15, labelpad=10)
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.set_zticklabels(np.arange(-10,10,2))
plt.show()