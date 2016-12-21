# coding: utf8
import numpy as np
import scipy.ndimage as nd
from scipy.cluster.vq import vq
 
from gmap import GMap
from array_dict import array_dict

def gmap_laplacian_smoothing(gmap, coef=0.5):
	""" Compute the new position of all elements of degree 0 in the
	GMap by moving them towards the isobarycenter of their neighbors :
	pos(i)* <- pos(i) + coef * sum_{j in N(i)} (1/valence(i))*(pos(j) - pos(i))
	"""


def gmap_gaussian_smoothing(gmap, coef=0.5, gaussian_sigma=None):
	"""
	Compute the new position of all elements of degree 0 in the
	GMap by moving them towards the a weighted barycenter of their neighbors 
	where the weights are a gaussian function of the edge lengths:
	pos(i)* <- pos(i) + coef * sum_{j in N(i)} (Gij/sum_k Gik)*(pos(j) - pos(i))
	Gij = e^{-lij^2 / sigma^2}
	"""


def gmap_taubin_smoothing(gmap, coef_pos=0.33, coef_neg=0.34, gaussian_sigma=None):
	"""
	Compute the new position of all elements of degree 0 in the
	GMap by applying two opposite Gaussian smoothing iterations
	"""
	gmap_gaussian_smoothing(gmap, coef=coef_pos, gaussian_sigma=gaussian_sigma)
	gmap_gaussian_smoothing(gmap, coef=-coef_neg, gaussian_sigma=gaussian_sigma)


def gmap_add_uniform_noise(gmap, coef=0.01):
	characteristic_distance = np.linalg.norm(np.std([gmap.get_position(v) for v in gmap.elements(0)],axis=0))
	for v in gmap.elements(0):
		gmap.set_position(v,gmap.get_position(v) + (1.-2.*np.random.rand(3))*coef*characteristic_distance)


def triangular_gmap_split_edge(gmap, dart):
	"""
	Perform a topological split operation on a triangular GMap
	by adding a new vertex in the middle of the edge and linking
	it to the opposite vertices of the adjacent triangles.
	"""
	# Compute the position of the edge center
	# Split the edge and get the new vertex dart
	# Update the position of the new vertex to the edge center
	# Split the face(s) incident to the new vertex


def triangular_gmap_flip_edge(gmap, dart):
	"""
	Perform a topological flip operation on a triangular GMap
	by modifying the alpha_1 relationships of the darts impacted :
	6 (reciprocal) relationships to update + make sure that 
	position dictionary is not impacted
	"""

	# Compute a dictionary of the new alpha_1 relationships of :
	# dart, alpha_0(dart), alpha_2(dart), alpha_0(alpha_2(dart)),
	# alpha_1(dart), alpha_1(alpha_0(dart)) 

	# Make sure that no dart in the orbit 1 of dart is a key of
	# the positions dictionary, otherwise transfer the position
	# to another embedding dart

	# Assert that the new alpha_1 is still without fixed points

	# Set the alphas of the GMap to their new values 
	# (not forgetting the reciprocal alpha_1)

	# Return the list of all darts  whose valence will be
	# impacted by the topological change
	

def triangular_gmap_dual_collapse_edge(gmap, dart):
	"""
	Perform a topological collapse operation on a triangular GMap by 
	removing an edge and two vertices in the dual structure, checking 
	that all the topological constraints are respected.
	The resulting vertex is placed at the center of the original edge.
	"""
	
	# Store the position of EACH DART in a dictionary   
	# Store the position of the edge center   

	try:
		# Compute the dual topology of the structure
		# Assert that the dual edge (same dart id) has two vertices
			
		# Store all the darts belonging to the vertices of the dual edge
		# Store all the darts belonging to the faces of the dual edge

		# Make sure none of the vertex darts of the dual edge belongs to 
		# a triangular face (dual of a vertex with valence 3)

		# Remove the dual edge from the dual 
		# Eliminate the removed darts from the face and vertex dart lists

		# For each vertex (0-cell) represented in the vertex dart list:
			# Remove the vertex from the dual
			# Elimininate the removed darts from the face dart list

		# Update the primal alphas with the new alphas of the dual's dual
		# Set back the position of each primal vertex to its dart position
		# Erase the position of the dual face dart list if any  
		# Set the postion of one representant in this list to the edge center

		return True
	except AssertionError:
		print "Impossible to collapse edge "+str(dart)
		return False 



def gmap_edge_split_optimization(gmap, maximal_length=1.0):
	"""
	Perform one iteration of edge split optimization:
	Rank the GMap edges by length and iterativelty split 
	those whose length exceeds maximal_length
	"""

	vertex_positions = array_dict([gmap.get_position(v) for v in gmap.darts()],gmap.darts())
	vertex_valence = array_dict(np.array(map(len,[gmap.orbit(v,[1,2]) for v in gmap.darts()]))/2,gmap.darts())
	edge_vertices = np.array([(e,gmap.alpha(0,e)) for e in gmap.elements(1)])
	edge_lengths = array_dict(np.linalg.norm(vertex_positions.values(edge_vertices[:,1]) - vertex_positions.values(edge_vertices[:,0]),axis=1),keys=gmap.elements(1))

	sorted_edge_length_edges = np.array(gmap.elements(1))[np.argsort(-edge_lengths.values(gmap.elements(1)))]
	sorted_edge_length_edges = sorted_edge_length_edges[edge_lengths.values(sorted_edge_length_edges)>maximal_length]
	
	n_splits = 0
	print "--> Splitting edges"
	for e in sorted_edge_length_edges:
		triangular_gmap_split_edge(gmap,e)
		n_splits += 1
	print "<-- Splitting edges (",n_splits," edges split)"

	return n_splits


def gmap_edge_flip_optimization(gmap, target_neighborhood=6):
	"""
	Perform one iteration of edge flip optimization:
	Identify the GMap edges that can be flipped and 
	compute the neighborhood error variation induced by
	their flip. Rank them along this variation and 
	perform allowed edge flips for edges with a negative
	variation.
	"""

	vertex_positions = array_dict([gmap.get_position(v) for v in gmap.darts()],gmap.darts())
	vertex_valence = array_dict(np.array(map(len,[gmap.orbit(v,[1,2]) for v in gmap.darts()]))/2,gmap.darts())

	edge_vertices = np.array([(e,gmap.alpha(0,e)) for e in gmap.elements(1)])
	edge_lengths = array_dict(np.linalg.norm(vertex_positions.values(edge_vertices[:,1]) - vertex_positions.values(edge_vertices[:,0]),axis=1),keys=gmap.elements(1))
	edge_flipped_vertices = np.array([[gmap.alpha(0,gmap.alpha(1,e)),gmap.alpha(0,gmap.alpha(1,gmap.alpha(2,e)))] for e in gmap.elements(1)])

	flippable_edges = np.array(gmap.elements(1))[edge_flipped_vertices[:,0] != edge_flipped_vertices[:,1]]
	
	flippable_edge_vertices = edge_vertices[edge_flipped_vertices[:,0] != edge_flipped_vertices[:,1]]
	flippable_edge_flipped_vertices = np.array([ e for e in edge_flipped_vertices[edge_flipped_vertices[:,0] != edge_flipped_vertices[:,1]]])

	flippable_edge_triangle_vertices = np.array([[np.concatenate([e,[v]]) for v in f] for (e,f) in zip(flippable_edge_vertices,flippable_edge_flipped_vertices)])
	flippable_edge_flipped_triangle_vertices = np.array([[np.concatenate([f,[v]]) for v in e] for (e,f) in zip(flippable_edge_vertices,flippable_edge_flipped_vertices)])

	from gmap_tools import triangle_geometric_features
	flippable_edge_triangle_areas = np.concatenate([triangle_geometric_features(flippable_edge_triangle_vertices[:,e],vertex_positions,features=['area']) for e in [0,1]],axis=1)
	flippable_edge_flipped_triangle_areas = np.concatenate([triangle_geometric_features(flippable_edge_flipped_triangle_vertices[:,e],vertex_positions,features=['area']) for e in [0,1]],axis=1)
		   
	average_area = np.nanmean(flippable_edge_triangle_areas)
	flippable_edge_flipped_triangle_areas[np.isnan(flippable_edge_flipped_triangle_areas)] = 100.
	wrong_edges = np.where(np.abs(flippable_edge_triangle_areas.sum(axis=1)-flippable_edge_flipped_triangle_areas.sum(axis=1)) > average_area/10.)

	flippable_edges = np.delete(flippable_edges,wrong_edges,0)
	flippable_edge_vertices = np.delete(flippable_edge_vertices,wrong_edges,0)
	flippable_edge_triangle_vertices = np.delete(flippable_edge_triangle_vertices,wrong_edges,0)
	flippable_edge_flipped_vertices = np.delete(flippable_edge_flipped_vertices,wrong_edges,0)
	flippable_edge_flipped_triangle_vertices = np.delete(flippable_edge_flipped_triangle_vertices,wrong_edges,0)
	flippable_edge_triangle_areas = np.delete(flippable_edge_triangle_areas,wrong_edges,0)
	flippable_edge_flipped_triangle_areas =  np.delete(flippable_edge_flipped_triangle_areas,wrong_edges,0)
				
	flippable_edge_neighborhood_error = np.power(vertex_valence.values(flippable_edge_vertices)-target_neighborhood,2.0).sum(axis=1)
	flippable_edge_neighborhood_error += np.power(vertex_valence.values(flippable_edge_flipped_vertices)-target_neighborhood,2.0).sum(axis=1)
	flippable_edge_neighborhood_flipped_error = np.power(vertex_valence.values(flippable_edge_vertices)-1-target_neighborhood,2.0).sum(axis=1)
	flippable_edge_neighborhood_flipped_error += np.power(vertex_valence.values(flippable_edge_flipped_vertices)+1-target_neighborhood,2.0).sum(axis=1)

	n_flips = 0 
	if len(flippable_edges)>0:

		flippable_edge_energy_variation = array_dict(flippable_edge_neighborhood_flipped_error-flippable_edge_neighborhood_error,flippable_edges)

		flippable_edge_sorted_energy_variation_edges = flippable_edges[np.argsort(flippable_edge_energy_variation.values(flippable_edges))]
		flippable_edge_sorted_energy_variation_edges = flippable_edge_sorted_energy_variation_edges[flippable_edge_energy_variation.values(flippable_edge_sorted_energy_variation_edges)<0] 

		modified_darts = set() 
		print "--> Flipping edges"

		for e in flippable_edge_sorted_energy_variation_edges:

			flippable_edge = (len(modified_darts.intersection(set(gmap.orbit(e,[1,2])))) == 0)
			flippable_edge = flippable_edge and (len(modified_darts.intersection(set(gmap.orbit(gmap.alpha(0,e),[1,2])))) == 0)
			flippable_edge = flippable_edge and (len(modified_darts.intersection(set(gmap.orbit(gmap.alpha(0,gmap.alpha(1,e)),[1,2])))) == 0)
			flippable_edge = flippable_edge and (len(modified_darts.intersection(set(gmap.orbit(gmap.alpha(0,gmap.alpha(1,gmap.alpha(2,e))),[1,2])))) == 0)

			if flippable_edge:
				n_e = len(gmap.elements(1))
				mod = triangular_gmap_flip_edge(gmap,e)
				modified_darts = modified_darts.union(set(mod))
				n_flips += 1
		print "<-- Flipping edges (",n_flips," edges flipped)"

	return n_flips


def gmap_edge_collapse_optimization(gmap, target_triangles=100, max_error=None, iterations_max=5):
	"""
	Perform several iterations of edge collapse optimization:
	Compute the error associated with the collapse of each
	GMap edge. Then at each iteration, sort the edges by 
	ascending error, and collapse those that have a length >
	minimal_length and an error < max_error, until the GMap
	has target_triangles faces, or iterations_max iterations
	have been performed.
	"""
	vertex_positions = array_dict([gmap.get_position(v) for v in gmap.darts()],gmap.darts())
	vertex_valence = array_dict(np.array(map(len,[gmap.orbit(v,[1,2]) for v in gmap.darts()]))/2,gmap.darts())

	edge_vertices = np.array([(e,gmap.alpha(0,e)) for e in gmap.elements(1)])
	edge_lengths = array_dict(np.linalg.norm(vertex_positions.values(edge_vertices[:,1]) - vertex_positions.values(edge_vertices[:,0]),axis=1),keys=gmap.elements(1))
	
	if target_triangles is None:
		target_triangles = len(gmap.elements(2))/4
	if minimal_length is None:
		minimal_length = np.percentile(edge_lengths.values(),20)
	if max_error is None:
		max_error = np.power(minimal_length,2)


	triangle_vertices = np.array([gmap.incident_cells(f,2,0) for f in gmap.elements(2)])
	vertices_positions = np.array([[gmap.get_position(v) for v in t] for t in triangle_vertices])
	normal_vectors = np.cross(vertices_positions[:,1]-vertices_positions[:,0],vertices_positions[:,2]-vertices_positions[:,0])
	normal_norms = np.linalg.norm(normal_vectors,axis=1)
	normal_vectors = normal_vectors/normal_norms[:,np.newaxis]
	plane_d = -np.einsum('...ij,...ij->...i',normal_vectors,vertices_positions[:,0])
	triangle_planes = np.concatenate([normal_vectors,plane_d[:,np.newaxis]],axis=1)

	triangle_plane_quadrics = array_dict(np.einsum('...i,...j->...ij',triangle_planes,triangle_planes),gmap.elements(2))
	vertex_triangles = array_dict([[gmap.get_embedding_dart(d,triangle_plane_quadrics,2) for d in gmap.incident_cells(v,0,2)] for v in gmap.elements(0)],keys=gmap.elements(0))

	iteration = 0
	while iteration<iterations_max and len(gmap.elements(2))>target_triangles:
		edge_vertices = np.array([[gmap.get_embedding_dart(d,vertex_triangles,0) for d in gmap.incident_cells(e,1,0)] for e in gmap.elements(1)])
		edge_lengths = array_dict(np.linalg.norm(vertex_positions.values(edge_vertices[:,1]) - vertex_positions.values(edge_vertices[:,0]),axis=1),keys=gmap.elements(1))

		edge_middles = np.array([gmap.element_center(e,1) for e in gmap.elements(1)])
		edge_middles_homogeneous = np.concatenate([edge_middles,np.ones((len(gmap.elements(1)),1))],axis=1)
		
		edge_vertex_faces = np.array([np.concatenate(vertex_triangles.values(v)) for v in edge_vertices])
		edge_vertex_face_quadrics = np.array([triangle_plane_quadrics.values(t) for t in edge_vertex_faces])
		edge_vertex_face_middles = np.array([ [e for t in e_v_t] for e,e_v_t in zip(edge_middles_homogeneous,edge_vertex_faces)])
		edge_quadrics_errors = np.array([np.abs(np.einsum('...ij,...ij->...i',m,np.einsum('...ij,...j->...i',q,m))).sum() for q,m in zip(edge_vertex_face_quadrics,edge_vertex_face_middles)])

		edge_quadrics_errors = array_dict(edge_quadrics_errors,gmap.elements(1))

		permutated_edges = np.array(gmap.elements(1))
		np.random.shuffle(permutated_edges)

		# sorted_quadrics_errors_edges = np.array(gmap.elements(1))[np.argsort(edge_quadrics_errors)]
		sorted_quadrics_errors_edges = permutated_edges[np.argsort(edge_quadrics_errors.values(permutated_edges))]
		# sorted_quadrics_errors_edges = sorted_quadrics_errors_edges[np.sort(edge_quadrics_errors) < np.percentile(edge_quadrics_errors,5)]
		sorted_quadrics_errors_edges = sorted_quadrics_errors_edges[edge_quadrics_errors.values(sorted_quadrics_errors_edges) < max_error]
		sorted_quadrics_errors_edges = sorted_quadrics_errors_edges[edge_lengths.values(sorted_quadrics_errors_edges) < minimal_length]
		sorted_quadrics_errors_edges = list(sorted_quadrics_errors_edges)


		# sorted_length_edges = permutated_edges[np.argsort(edge_lengths.values(permutated_edges))]
		# sorted_length_edges = sorted_length_edges[edge_quadrics_errors.values(sorted_length_edges) < max_error]
		# sorted_length_edges = list(sorted_length_edges)

		print "--> Collapsing edges"
		modified_edges = set()
		n_collapses = 0 
		while len(sorted_quadrics_errors_edges)>0 and len(gmap.elements(2))>target_triangles:
		# while len(sorted_length_edges)>0 and len(gmap.elements(2))>target_triangles:
			e = sorted_quadrics_errors_edges.pop(0)
			# e = sorted_length_edges.pop(0)
			if not e in modified_edges and len(gmap.elements(2))>target_triangles:
				#print [gmap.orbit(d,[0,2]) for d in gmap.adjacent_cells(e,1)]
				#print np.unique(np.concatenate([gmap.orbit(d,[0,2]) for d in gmap.adjacent_cells(e,1)]))
				modified_edges = modified_edges.union(set(np.unique(np.concatenate([gmap.orbit(d,[0,2]) for d in gmap.adjacent_cells(e,1)])))).union({e})
				collapsed = triangular_gmap_dual_collapse_edge(gmap,e)
				n_collapses += collapsed
				if collapsed:
					print "  --> Collapsed edge ",e," [",len(gmap.elements(2))," Faces]"
		print "<-- Collapsing edges (",n_collapses," edges collapsed)"

		iteration += 1

	return n_collapses
