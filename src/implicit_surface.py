# coding: utf8

import numpy as np

class ImplicitSurface(object):
	def __init__(self):
		pass

	def potential_evaluator(self):
		def potential(x,y,z):
			points = np.transpose(np.array([x,y,z]),range(1,np.array(x).ndim+1)+[0])
			return np.zeros_like(points[...,0])
		return potential

	def surface_mesh(self, grid_size=128, grid_resolution=1, potential_value=0.5):
		from skimage.measure import marching_cubes

		grid_limit = grid_size*grid_resolution/2
		x,y,z = np.mgrid[-grid_limit:grid_limit:grid_resolution,-grid_limit:grid_limit:grid_resolution,-grid_limit:grid_limit:grid_resolution]

		potential_field = self.potential_evaluator()(x,y,z)
		surface_points, surface_triangles = marching_cubes(potential_field,potential_value)
		surface_points = np.array(surface_points)*grid_resolution - grid_limit

		return surface_points, surface_triangles


class SphericalImplicitSurface(ImplicitSurface):
	def __init__(self, center=[0,0,0], radius=1.):
		super(ImplicitSurface, self).__init__()
		# Instantiate attributes of the class to store the center and 
		# necessary parameters (e.g. radius for Stolte).

	def potential_evaluator(self):
		def potential(x,y,z):
			# Compute the distance of points described by x,y,z arrays
			# to the center point (HINT : use np.linalg.norm).
			# Apply the potential function of your choice to the distances
			# for instance p(d) = (1/R^8)*(d^2-R^2)^4 for Stolte
			# Return the array of potential values
		return potential

