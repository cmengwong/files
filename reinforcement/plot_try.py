# this function is going to plot the 2D - array
# for me to visualize the error of the laplace_equation of 
# every iteration.

# after doing this funciton, my next goal is to 
# transform then into fourier transformation
# and get the feature out of this data.


# from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(X, Y, Z):
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# Make data.
	X_m, Y_m = np.meshgrid(X, Y)

	# Plot the surface.
	surf = ax.plot_surface(X_m, Y_m, Z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)

	# Customize the z axis.
	# ax.set_zlim(-1.01, 1.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

if __name__ == '__main__':
	X = np.arange(-5, 5, 0.25)
	Y = np.arange(-5, 5, 0.25)
	Z = np.sin(np.sqrt(X**2 + Y**2))
	plot_3d(X, Y, Z)