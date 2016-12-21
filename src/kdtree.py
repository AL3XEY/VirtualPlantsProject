# coding: utf8
import openalea.plantgl.all as pgl

class KDNode:
	def __init__(self, pivot = None, axis = None, left_child = None, right_child = None):
		self.pivot	   = pivot
		self.axis		= axis
		self.left_child  = left_child
		self.right_child = right_child


def print_kdtree(kdtree, depth = 0):
	if isinstance(kdtree, KDNode):
		print ('	'*depth) + 'Node :', kdtree.axis,  kdtree.pivot
		print_kdtree(kdtree.left_child, depth+1)
		print_kdtree(kdtree.right_child, depth+1)
	else:
		assert (type(kdtree) == list) or (isinstance(kdtree,np.ndarray))
		print ('	'*depth) + 'Leaf :', kdtree


def create_kdtree(point_list, minbucketsize=3, depth=0):
	import numpy as np

	if len(point_list) >= 2*minbucketsize:
		dim = depth%3
		sorted_points = np.array(point_list)[np.argsort(np.array(point_list)[:,dim])]

		kd_node = KDNode()
		kd_node.pivot = list(sorted_points[len(sorted_points)/2])
		kd_node.axis = dim
		kd_node.left_child = create_kdtree(sorted_points[:len(sorted_points)/2],minbucketsize,depth+1)
		kd_node.right_child = create_kdtree(sorted_points[len(sorted_points)/2 + 1:],minbucketsize,depth+1)

		return kd_node
	else:
		return [list(p) for p in point_list]


def generate_random_point(size=[1,1,1], distribution='uniform'):
	from random import uniform, gauss
	if distribution == 'uniform':
		return pgl.Vector3(uniform(-size[0],size[0]), uniform(-size[1],size[1]), uniform(-size[2],size[2])) 
	elif distribution == 'gaussian':
		return pgl.Vector3(gauss(0,size[0]/3.), gauss(0,size[1]/3.), gauss(0,size[1]/3.)) 

 
def generate_random_pointlist(size=[1,1,1], nb = 100, distribution='uniform'):
	return [generate_random_point(size, distribution=distribution) for i in xrange(nb)]


def brute_force_closest(point, pointlist):
	""" Find the closest points of 'point' in 'pointlist' using a brute force approach """
	import sys
	pid, d = -1, sys.maxint
	for i, p in enumerate(pointlist):
		nd = pgl.norm(point-p) 
		if nd < d:
			d = nd
			pid = i
	return pointlist[pid]


def view_kdtree(kdtree, bbox=[[-1., 1.],[-1., 1.],[-1., 1.]], radius=0.05):
	import numpy as np

	scene = pgl.Scene()

	sphere = pgl.Sphere(radius,slices=16,stacks=16)

	silver = pgl.Material(ambient=(49,49,49),diffuse=3.,specular=(129,129,129),shininess=0.4)
	gold = pgl.Material(ambient=(63,50,18),diffuse=3.,specular=(160,141,93),shininess=0.4)

	if isinstance(kdtree, KDNode):
		dim = kdtree.axis
		plane_bbox = [b for i,b in enumerate(bbox) if i != dim]
		plane_points = []
		plane_points += [np.insert([plane_bbox[0][0],plane_bbox[1][0]],dim,kdtree.pivot[dim])]
		plane_points += [np.insert([plane_bbox[0][0],plane_bbox[1][1]],dim,kdtree.pivot[dim])]
		plane_points += [np.insert([plane_bbox[0][1],plane_bbox[1][1]],dim,kdtree.pivot[dim])]
		plane_points += [np.insert([plane_bbox[0][1],plane_bbox[1][0]],dim,kdtree.pivot[dim])]

		left_bbox = np.copy(bbox).astype(float)
		right_bbox = np.copy(bbox).astype(float)
		left_bbox[dim,1] = kdtree.pivot[dim]
		right_bbox[dim,0] = kdtree.pivot[dim]

		scene += pgl.Shape(pgl.Translated(kdtree.pivot,sphere),gold)
		scene += view_kdtree(kdtree.left_child, bbox=left_bbox, radius=radius)
		scene += view_kdtree(kdtree.right_child, bbox=right_bbox, radius=radius)
		# scene += pgl.Shape(pgl.Polyline(plane_points+[plane_points[0]],width=2),pgl.Material(ambient=(0,0,0)))
		scene += pgl.Shape(pgl.Polyline(plane_points+[plane_points[0]],width=2),pgl.Material(ambient=tuple(list(np.insert([0,0],dim,255)))))
		scene += pgl.Shape(pgl.FaceSet(plane_points,[range(4)]),pgl.Material(ambient=(0,0,0),transparency=0.6))

	else:
		assert (type(kdtree) == list) or (isinstance(kdtree,np.ndarray))
		for p in kdtree:
			scene += pgl.Shape(pgl.Translated(p,sphere),silver)

	return scene


def kdtree_closest(point, kdtree, depth=0):

	if isinstance(kdtree, KDNode):
		dim = kdtree.axis

		if point[dim]<kdtree.pivot[dim]:
			point_kdtree, opposite_kdtree = kdtree.left_child, kdtree.right_child
		else:
			point_kdtree, opposite_kdtree = kdtree.right_child, kdtree.left_child

		best_candidate = kdtree_closest(point, point_kdtree)
		best_distance = pgl.norm(point-best_candidate)

		if best_distance > abs(point[dim]-kdtree.pivot[dim]):
			
			if best_distance > pgl.norm(point-kdtree.pivot):
				best_candidate = kdtree.pivot
				best_distance = pgl.norm(point-best_candidate)

			opp_candidate = kdtree_closest(point, opposite_kdtree)
			opp_distance = pgl.norm(point-opp_candidate)
			if best_distance > opp_distance:
				best_candidate = opp_candidate
	else:
		best_candidate = brute_force_closest(point, kdtree)

	return best_candidate


def test_kdtree(create_kdtree_func, closestpoint_func, nbtest=100, nbpoints=1000, size=[1,1,1], minbucketsize=2):
	import time

	points = generate_random_pointlist(nb = nbpoints, size=size, distribution='uniform')
	mkdtree = create_kdtree_func(points, minbucketsize)
	#pgl.Viewer.display(view_kdtree(mkdtree, radius=0.03, bbox=[[-float(s),float(s)] for s in size]))
	kdtime, bftime = 0,0
	for i in xrange(nbtest):
		testpoint = generate_random_point()
		t = time.time()
		kpoint = closestpoint_func(testpoint, mkdtree)
		kdtime += time.time()-t
		t = time.time()
		bfpoint = brute_force_closest(testpoint, points)
		bftime += time.time()-t
		if kpoint != bfpoint: 
			raise ValueError('Invalid closest point')
	print 'Comparative execution time : KD-Tree [', kdtime,'], BruteForce [', bftime,']'

	return kdtime, bftime


def plot_execution_time(nbpoints_min=10, nbpoints_max=1000):
	import matplotlib.pyplot as plt

	kd_times = []
	bf_times = []
	nb_points = range(nbpoints_min,nbpoints_max,10)

	for n in nb_points:
		kdtime, bftime = test_kdtree(create_kdtree,kdtree_closest,nbpoints=n)
		kd_times += [kdtime]
		bf_times += [bftime]
	
	plt.figure("Execution Time")
	plt.plot(nb_points,kd_times,color='r',label="KD-Tree")
	plt.plot(nb_points,bf_times,color='b',label="Brute Force")
	plt.legend()
	plt.show()





