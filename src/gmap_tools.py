# coding: utf8
import numpy as np
from scipy import ndimage as nd
from scipy.cluster.vq import vq

from array_dict import array_dict

from gmap import GMap
from basicshapes import topopolygon


def array_unique(array,return_index=False):
	_,unique_rows = np.unique(np.ascontiguousarray(array).view(np.dtype((np.void,array.dtype.itemsize * array.shape[1]))),return_index=True)
	if return_index:
		return array[unique_rows],unique_rows
	else:
		return array[unique_rows]

def read_ply_mesh(filename):
	import csv
	import re

	property_types = {}
	property_types['int'] = 'int'
	property_types['int32'] = 'int'
	property_types['uint'] = 'int'
	property_types['uint8'] = 'int'
	property_types['uchar'] = 'int'
	property_types['float'] = 'float'
	property_types['float32'] = 'float'
	property_types['list'] = 'list'

	ply_file = open(filename,'rb')

	assert "ply" in ply_file.next()
	assert "ascii" in ply_file.next()

	n_wisps = {}
	properties = {}
	properties_types = {}
	properties_list_types = {}
	element_name = ""
	property_name = ""
	elements = []

	line = ply_file.next()
	while not 'end_header' in line:
		
		if re.split(' ',line)[0] == 'element':
			element_name = re.split(' ',line)[1]
			elements.append(element_name)
			n_wisps[element_name] = int(re.split(' ',line)[2])
			properties[element_name] = []
			properties_types[element_name] = {}
			properties_list_types[element_name] = {}
			
		if re.split(' ',line)[0] == 'property':
			property_name = re.split(' ',line)[-1][:-1]
			properties[element_name].append(property_name)
			properties_types[element_name][property_name] = re.split(' ',line)[1]
			if properties_types[element_name][property_name] == 'list':
				list_type = re.split(' ',line)[-2]
				properties_list_types[element_name][property_name] = list_type
		
		line = ply_file.next()

	print n_wisps
	print properties

	element_properties = {}

	for element_name in elements:
		element_properties[element_name] = {}
		for wid in xrange(n_wisps[element_name]):
			line = ply_file.next()
			line_props = {}
			prop_index = 0
			for prop in properties[element_name]:
				prop_type = properties_types[element_name][prop]
				if property_types[prop_type] == 'float':
					line_props[prop] = float(re.split(' ',line)[prop_index])
					prop_index += 1
				elif property_types[prop_type] == 'int':
					line_props[prop] = int(re.split(' ',line)[prop_index])
					prop_index += 1
				elif property_types[prop_type] == 'list':
					list_length = int(re.split(' ',line)[prop_index])
					prop_index += 1
					list_type =  properties_list_types[element_name][prop]
					if property_types[list_type] == 'float':
						line_props[prop] = [float(p) for p in re.split(' ',line)[prop_index:prop_index+list_length]]
					elif property_types[list_type] == 'int':
						line_props[prop] = [int(p) for p in re.split(' ',line)[prop_index:prop_index+list_length]]
					prop_index += list_length
			element_properties[element_name][wid] = line_props
	ply_file.close()
		
	point_positions = {}
	for pid in xrange(n_wisps['vertex']):
		point_positions[pid] = np.array([element_properties['vertex'][pid][dim] for dim in ['x','y','z']])

	face_vertices = {}
	for fid in xrange(n_wisps['face']):
		if element_properties['face'][fid].has_key('vertex_index'):
			face_vertices[fid] = element_properties['face'][fid]['vertex_index']
		elif element_properties['face'][fid].has_key('vertex_indices'):
			face_vertices[fid] = element_properties['face'][fid]['vertex_indices']

	print len(point_positions)," Points, ", len(face_vertices), " Faces"
	#raw_input()

	unique_points = array_unique(np.array(point_positions.values()))
	point_matching = array_dict(vq(np.array(point_positions.values()),unique_points)[0],point_positions.keys())

	print len(unique_points)," Unique Points"
	#raw_input()

	faces = np.array(face_vertices.values())
	if faces.ndim == 2:
		triangular = True
		triangles = np.sort(point_matching.values(faces))
		unique_triangles = array_unique(triangles)
		triangle_matching = array_dict(vq(triangles,unique_triangles)[0],face_vertices.keys())
	else:
		triangular = False
		unique_triangles = point_matching.values(faces)
		triangle_matching = array_dict(face_vertices.keys(),face_vertices.keys())
		
	print len(unique_triangles)," Unique Faces"

	return unique_points, unique_triangles

def gmap_from_triangular_mesh(points, triangles, center=False):
	gmap = GMap()
 
	triangle_edges = np.sort(np.concatenate([np.transpose([v,list(v[1:])+[v[0]]]) for v in triangles]))
 
	edges = array_unique(triangle_edges)
	triangle_edges = (vq(triangle_edges,edges)[0]).reshape((len(triangles),3))
 
	triangle_edge_vertices = np.concatenate([np.transpose([v,list(v[1:])+[v[0]]]) for v in triangles])
	triangle_edge_orientation = (triangle_edge_vertices[:,1]>triangle_edge_vertices[:,0]).reshape((len(triangles),3))
 
	triangle_darts = {}
	for fid,t in enumerate(triangles):
		gmap, tri = topopolygon(3,gmap)
		triangle_darts[fid] = tri
 
	for eid,e in enumerate(edges):
		fids_to_sew, darts_to_sew = np.where(triangle_edges==eid)
		orientations_to_sew = triangle_edge_orientation[(fids_to_sew, darts_to_sew)]
		if len(orientations_to_sew) > 1:
			gmap.sew_dart(2,triangle_darts[fids_to_sew[0]][2*darts_to_sew[0]+(1-orientations_to_sew[0])],triangle_darts[fids_to_sew[1]][2*darts_to_sew[1]+(1-orientations_to_sew[1])])
 
	mesh_center = np.mean(points,axis=0) if center else np.zeros(3)
	for fid,t in enumerate(triangles):
		for i,p in enumerate(t):
			gmap.set_position(triangle_darts[fid][2*i], points[p]-mesh_center)
 
	return gmap


def mesh_display(points, triangles, color=[190,205,205], transparency=0, add=False):
	import openalea.plantgl.all as pgl
	if color is None:
		color = [np.random.randint(0,255) for i in xrange(3)] 
	mat = pgl.Material(tuple(color), diffuse=0.25, transparency=transparency)
	s = pgl.Scene()
	s += pgl.Shape(pgl.FaceSet(points, triangles), mat)
	if add : 
		pgl.Viewer.add(s)
	else : 
		pgl.Viewer.display(s)

def triangle_geometric_features(triangles,positions,features=['area','max_distance']):
	triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])
	triangle_edges = triangles[...,triangle_edge_list]
	triangle_edge_lengths = np.linalg.norm(positions.values(triangle_edges[...,1]) - positions.values(triangle_edges[...,0]),axis=2)
	
	triangle_features={}

	triangle_features['edge_lengths'] = triangle_edge_lengths

	triangle_features['perimeter'] = triangle_edge_lengths.sum(axis=1)
	triangle_features['area'] = np.sqrt((triangle_features['perimeter']/2.0)*(triangle_features['perimeter']/2.0-triangle_edge_lengths[...,0])*(triangle_features['perimeter']/2.0-triangle_edge_lengths[...,1])*(triangle_features['perimeter']/2.0-triangle_edge_lengths[...,2]))
	triangle_features['eccentricity'] = 1. - (12.0*np.sqrt(3)*triangle_features['area'])/np.power(triangle_features['perimeter'],2.0)

	if ('max_distance' in features) or ('min_distance' in features):
		triangle_edge_lengths = np.sort(triangle_edge_lengths)
		triangle_features['max_distance'] = triangle_edge_lengths[:,-1]
		triangle_features['min_distance'] = triangle_edge_lengths[:,0]

	if 'sinus' in features or 'sinus_eccentricity' in features:
		triangle_features['sinus'] = np.zeros_like(triangle_edge_lengths,np.float16)
		triangle_features['sinus'][:,0] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[...,1]**2+triangle_edge_lengths[...,2]**2-triangle_edge_lengths[:,0]**2,2.0)/np.power(2.0*triangle_edge_lengths[...,1]*triangle_edge_lengths[...,2],2.0),np.float16))
		triangle_features['sinus'][:,1] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[...,2]**2+triangle_edge_lengths[...,0]**2-triangle_edge_lengths[:,1]**2,2.0)/np.power(2.0*triangle_edge_lengths[...,2]*triangle_edge_lengths[...,0],2.0),np.float16))
		triangle_features['sinus'][:,2] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[...,0]**2+triangle_edge_lengths[...,1]**2-triangle_edge_lengths[:,2]**2,2.0)/np.power(2.0*triangle_edge_lengths[...,0]*triangle_edge_lengths[...,1],2.0),np.float16))
	
	if 'cosinus' in features:
		triangle_features['cosinus'] = np.zeros_like(triangle_edge_lengths,np.float16)
		triangle_features['cosinus'][:,0] = (triangle_edge_lengths[...,1]**2+triangle_edge_lengths[...,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[...,1]*triangle_edge_lengths[...,2])
		triangle_features['cosinus'][:,1] = (triangle_edge_lengths[...,2]**2+triangle_edge_lengths[...,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[...,2]*triangle_edge_lengths[...,0])
		triangle_features['cosinus'][:,2] = (triangle_edge_lengths[...,0]**2+triangle_edge_lengths[...,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[...,0]*triangle_edge_lengths[...,1])

	if 'sinus_eccentricity' in features:
		triangle_features['sinus_eccentricity'] = 1.0 - (2.0*triangle_features['sinus'].sum(axis=1))/(3*np.sqrt(3))

	return np.concatenate([triangle_features[f][:,np.newaxis] for f in features],axis=1)
	
def gmap_triangular_quality(gmap):
	n_triangles = len(gmap.elements(2))
	positions = array_dict([gmap.get_position(v) for v in gmap.elements(0)],keys=gmap.elements(0))
	triangles = np.array([[gmap.get_embedding_dart(v,positions,0) for v in gmap.incident_cells(t,2,0)] for t in gmap.elements(2)])

	if (triangles.ndim != 2) or (triangles.shape[1] != 3):
		face_valences = np.array(map(len,triangles))
		print dict(zip(np.unique(face_valences),nd.sum(np.ones_like(face_valences),face_valences,np.unique(face_valences))))
		print "Non triangular GMap! Impossible to compute triangle quality!"

	triangle_eccentricities = triangle_geometric_features(triangles,positions,features=['sinus_eccentricity'])[:,0]
	vertex_valences = np.array(map(len,[gmap.incident_cells(v,0,1) for v in gmap.elements(0)]))

	quality = {}
	quality['eccentricity'] = np.mean(triangle_eccentricities)
	quality['valence'] = np.mean(vertex_valences)

	print n_triangles,"Triangles  [ eccentricity :",np.round(quality['eccentricity'],4),", valence :",np.round(quality['valence'],4),"]"

	return quality
