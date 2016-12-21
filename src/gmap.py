# coding: utf8

import numpy as np

class GMap:
	def __init__(self, degree=2):
		""" 
		Constructor 
		"""

		self.maxid = 0
		self.alphas = { 0 : {}, 1 : {}, 2 : {} }
		self.positions = {}

	def darts(self): 
		""" 
		Return a list of id representing the darts of the structure 
		"""
		return self.alphas[0].keys()

	def alpha(self, degree, dart):
		""" Return the application of the alpha_deg on dart """
		return self.alphas[degree][dart]

	def alpha_composed(self, list_of_alpha_value, dart):
		""" 
		Return the application of a composition of alphas on dart 
		"""
		if list_of_alpha_value == []:
			return dart
		else:
			return self.alpha_composed(list_of_alpha_value[:-1],self.alpha(list_of_alpha_value[-1],dart))

	def is_free(self, degree, dart):
		""" 
		Test if dart is free for alpha_degree (if it is a fixed point) 
		"""
		return self.alpha(degree,dart) == dart

	def add_dart(self, dart=None):
		""" 
		Create a new dart and return its id. 
		Set its alpha_i to itself (fixed points) 
		"""
		if dart is None:
			dart = self.maxid
		assert not dart in self.darts()
		for degree in self.alphas.keys():
			self.alphas[degree][dart] = dart
		self.maxid =  np.max(self.darts())+1
		return dart

	def remove_dart(self, dart):
		"""
		Remove a dart from the structure by removing it from
		the alphas dictionaries (keys and values) and transferring
		its position to another embedding dart if necessary.
		"""
		
		# If dart is a key of the positions dictionary:
			# Store the position
			# Remove the dictionary item
			# Find another dart in the vertex orbit (if any)
			# Assign it the position
	   
		# For each value i of alpha:
			# Set alpha_i(alpha_i(dart)) to itself
			# Remove the alphas[i] dictionary item


	def is_valid(self):
		""" 
		Test the validity of the structure. 
		Check if there is pending dart for alpha_0 and alpha_1 (fixed point) 
		"""
		valid = True
		valid = valid and np.all([self.alpha(0,d)!=d for d in self.darts()])
		valid = valid and np.all([self.alpha_composed([0,0],d)==d for d in self.darts()])
		valid = valid and np.all([self.alpha(1,d)!=d for d in self.darts()])
		valid = valid and np.all([self.alpha_composed([1,1],d)==d for d in self.darts()])
		return valid

	def link_darts(self,degree, dart1, dart2): 
		""" 
		Link the two darts with a relation alpha_degree
		"""
		assert self.is_free(degree,dart1) and self.is_free(degree,dart2)
		self.alphas[degree][dart1] = dart2
		self.alphas[degree][dart2] = dart1

	def print_alphas(self):
		""" 
		Print for each dart, the value of the different alpha applications.
		""" 
		try:
			from colorama import Style, Fore
		except:
			print "Try to install colorama (pip install colorama) for a better-looking display!"
			for d in self.darts():
				print d," | ",self.alpha(0,d),self.alpha(1,d),self.alpha(2,d) # , self.get_position(d)
		else:
			print "d	 α0  α1  α2"
			for d in self.darts():
				print d," | ",Fore.MAGENTA+str(self.alpha(0,d))," ",Fore.GREEN+str(self.alpha(1,d))," ",Fore.BLUE+str(self.alpha(2,d))," ",Style.RESET_ALL 


	def orbit(self, dart, list_of_alpha_value):
		""" 
		Return the orbit of dart using a list of alpha relation.
		Example of use : gmap.orbit(0,[0,1]).
		In Python, you can use the set structure to process only once all darts of the orbit.  
		"""
		orbit = []
		marked = set([])
		toprocess = [dart]

		while len(toprocess)>0:
			d = toprocess.pop(0)
			if not d in marked:
				orbit.append(d)
				marked.add(d)
				for degree in list_of_alpha_value:
					toprocess.append(self.alpha(degree,d))

		return orbit


	def orderedorbit(self, dart, list_of_alpha_value):
		"""
		Return the ordered orbit of dart using a list of alpha relations by applying
		repeatingly the alpha relations of the list to dart.
		Example of use. gmap.orderedorbit(0,[0,1]).
		Warning: No fixed point for the given alpha should be contained.
		"""
		orbit = []
		current_dart = dart
		current_alpha_index = 0
		n_alpha = len(list_of_alpha_value)
		while (current_dart != dart) or orbit==[]:
			orbit.append(current_dart)
			current_alpha = list_of_alpha_value[current_alpha_index]
			current_dart = self.alpha(current_alpha,current_dart)
			current_alpha_index = (current_alpha_index+1) % n_alpha
		return orbit


	def sew_dart(self, degree, dart1, dart2, merge_attribute = True):
		"""
		Sew two elements of degree 'degree' that start at dart1 and dart2.
		Determine first the orbits of dart to sew and heck if they are compatible.
		Sew pairs of corresponding darts, and if they have different embedding 
		positions, merge them. 
		"""
		if degree == 1:
			self.link_darts(1, dart1, dart2)
		else:
			alpha_list = [0]
			orbit1 = self.orbit(dart1, alpha_list)
			orbit2 = self.orbit(dart2, alpha_list)
			if len(orbit1) != len(orbit2):
				raise ValueError('Incompatible orbits', orbit1, orbit2)
			for d1,d2 in zip(orbit1, orbit2):
				self.link_darts(degree, d1, d2)
				if merge_attribute:
					d1e = self.get_embedding_dart(d1, self.positions)
					d2e = self.get_embedding_dart(d2, self.positions)
					if d1e in self.positions and d2e in self.positions:
							pos = (self.positions[d1e] + self.positions[d2e]) / 2.
							del self.positions[d2e]
							self.positions[d1e] = pos



	def elements(self, degree):
		""" 
		Return one dart per element of degree. For this, consider all darts as initial set S. 
		Take the first dart d, remove from the set all darts of the orbit starting from d and 
		corresponding to element of degree degree. Take then next element from set S and do the 
		same until S is empty. 
		Return all darts d that were used. """
		
		elements = []
		darts = set(self.darts())

		list_of_alpha_value = range(3)
		list_of_alpha_value.remove(degree)

		while len(darts) > 0:
			dart = darts.pop()
			elementi = self.orbit(dart, list_of_alpha_value)
			darts -= set(elementi)
			elements.append(dart)

		return elements


	def incident_cells(self, dart, degree, incidentdegree):
		"""
		Return all the element of degree incidentdegree
		that are incident to the element dart of degree degree.
		(Typically all edges around a point)
		For this iterate over all the dart of the orbit of (dart, degree).
		For each dart d of this orbit, get all the darts coresponding
		to the orbit of the element (d, incidentdegree) and remove them
		from the original set.
		"""
		results = []

		alphas = range(3)
		alphas.remove(degree) 

		incidentalphas = range(3)
		incidentalphas.remove(incidentdegree) 

		marked = set()

		for d in self.orbit(dart, alphas):
			if not d in marked:
				results.append(d)
				marked |= set(self.orbit(d, incidentalphas))

		return results


	def adjacent_cells(self, dart, degree):
		""" 
		Return all the elements of degree degree
		that are adjacent to the element dart with respect
		to the alpha relation of degree degree.
		(Typically all points sharing an edge with a point)
		For this iterate over all the dart of the orbit of (dart, degree).
		For each dart d of this orbit, get its neighbor n (alpha degree)
		and remove its orbit (n, degree) from the set of darts
		to consider.
		See function incident_cells for inspiration.
		"""

		results = []

		alphas = range(3)
		alphas.remove(degree) 

		marked = set()

		for d in self.orbit(dart, alphas):
			n = self.alpha(degree,d)
			if not n in marked:
				results.append(n)
				marked |= set(self.orbit(n, alphas))

		return results

   
	def is_removable_cell(self, dart, degree):
		"""
		Check whether the cell of a given degree represented by
		the specified dart can be safely removed from the object.
		"""
		
		# Should return True if:
		# - degree is 1 or
		# - degree is 0 and all d in the vertex orbit verify:
		#	alpha_1(alpha_2(d)) = alpha_2(alpha_1(d))  
	
	
	def remove_cell(self, dart, degree):
		"""
		Remove all the darts forming the cell of a given degree
		represented bythe specified dart, if the removal can
		be performed.
		"""
		
		# Verify that the cell can be removed

		# Re-link the opposite darts of the cell orbit
		# For each dart d of the cell orbit:
			# Compute n = alpha_degree(d)
			# If n is not in the orbit:
				# Compute n' = alpha_degree(alpha_degree+1(n))
				# While n' is in the orbit:
					# n' <- alpha_degree(alpha_degree+1(n'))
				# Link n and n' by alpha_degree
				
		 # Remove all darts in the cell orbit
		 # Return the list of removed darts


	def insert_edge(self, dart):
		""" 
		Insert an edge at the point represented by dart.
		Return a dart corresponding to the dandling edge end.
		"""

		dart1 = self.alpha(1, dart)
		newdarts = [self.add_dart() for i in xrange(4)]
		
		self.link_darts(0, newdarts[0], newdarts[1])
		self.link_darts(0, newdarts[3], newdarts[2])
		
		self.link_darts(2, newdarts[0], newdarts[3])
		self.link_darts(2, newdarts[1], newdarts[2])

		self.alphas[1][dart] = newdarts[0]
		self.alphas[1][newdarts[0]] = dart

		self.alphas[1][dart1] = newdarts[3]
		self.alphas[1][newdarts[3]] = dart1

		return newdarts[1]


	def split_face(self, dart1, dart2=None):
		"""
		Split face by inserting an edge between dart1 and dart2 
		"""

		if dart2 is None:
			dart2 = self.alpha_composed([0,1,0],dart1)

		dedge = self.insert_edge(dart1)

		dart2a1 = self.alpha(1,dart2)
		dedgea2 = self.alpha(2, dedge)

		self.alphas[1][dart2] = dedge
		self.alphas[1][dedge] = dart2

		self.alphas[1][dart2a1] = dedgea2
		self.alphas[1][dedgea2] = dart2a1


	def split_edge(self, dart):
		""" 
		Operator to split an edge. 
		Return a dart corresponding to the new points
		"""
		orbit1 = self.orbit(dart,[2])
		orbit2 = self.orbit(self.alpha(0,dart),[2])

		newdart1 = [self.add_dart() for i in orbit1]
		newdart2 = [self.add_dart() for i in orbit2]
		
		for d, nd in zip(orbit1+orbit2, newdart1+newdart2):
			self.alphas[0][d] = nd
			self.alphas[0][nd] = d
		
		for nd1, nd2 in zip(newdart1, newdart2):
			self.link_darts(1, nd1, nd2)
		
		for nd in newdart1+newdart2:
			if self.is_free(2, nd) and not self.is_free(2, self.alpha(0, nd)):
				self.link_darts(2,nd, self.alpha(0,self.alpha(2,self.alpha(0,nd))))

		return newdart1[0]


	def dual(self, geometry=True):
		"""
		Compute the dual of the object.
		Create a new GMap object with the same darts but reversed alphas.
		Update positions of the dual 0-cells as the centers of the 2-cells
		"""
		
		dual_gmap = GMap()
		for alpha_value in [0,1,2]:
			dual_gmap.alphas[alpha_value] = self.alphas[2-alpha_value]

		if geometry:
			for face_dart in self.elements(2):
				dual_gmap.set_position(face_dart,self.element_center(face_dart,2))

		return dual_gmap	  


	def get_embedding_dart(self, dart, propertydict, degree=0):
		""" 
		Check if a dart of the orbit representing the vertex has already been 
		associated with a value in propertydict. If yes, return this dart, else
		return the dart passed as argument.
		"""
		alphas = range(3)
		alphas.remove(degree)
		for d in self.orbit(dart, alphas):
			if propertydict.has_key(d):
				return d
		return dart


	def get_position(self, dart):
		"""
		Retrieve the coordinates associated to the vertex <alpha_1, alpha_2>(dart) 
		"""
		return self.positions.get(self.get_embedding_dart(dart,self.positions,0))


	def set_position(self, dart, position) :
		"""
		Associate coordinates with the vertex <alpha_1,alpha_2>(dart)
		"""
		self.positions[self.get_embedding_dart(dart,self.positions)] = np.array(position)
	

	def display(self, color=[190,205,205], add=False, transparency=0):
		from openalea.plantgl.all import Scene, Shape, Material, FaceSet, Viewer
		from random import randint
		s = Scene()
		for facedart in self.elements(2):
			lastdart = facedart
			positions = []
			for dart in self.orderedorbit(facedart,[0,1]):
				if self.alpha(0, dart) != lastdart:
					positions.append(self.get_position(dart))
					lastdart = dart
				if color is None:
					mat = Material((randint(0,255),randint(0,255),randint(0,255)),transparency=transparency)
				else:
					mat = Material(tuple(color),diffuse=0.25,transparency=transparency)
				s.add(Shape(FaceSet(positions, [range(len(positions))]) , mat, facedart ))
		if add : 
			Viewer.add(s)
		else : 
			Viewer.display(s)


	def element_center(self, dart, degree):
		list_of_alpha_value = range(3)
		list_of_alpha_value.remove(degree)

		element_positions = [self.get_position(d) for d in self.orbit(dart,list_of_alpha_value)]
		return np.mean(element_positions,axis=0)


	def dart_display(self, radius=0.1, coef=0.8, add=False):
		import openalea.plantgl.all as pgl

		sphere = pgl.Sphere(radius,slices=16,stacks=16)
		coal = pgl.Material(ambient=(8,10,13),diffuse=3.,specular=(89,89,89),shininess=0.3)
		purple = pgl.Material(ambient=(72,28,72),diffuse=2.,specular=(89,89,89),shininess=0.3)
		green = pgl.Material(ambient=(0,88,9),diffuse=2.,specular=(89,89,89),shininess=0.3)
		blue = pgl.Material(ambient=(9,0,88),diffuse=2.,specular=(89,89,89),shininess=0.3)

		font = pgl.Font(size=10)

		s = pgl.Scene()

		dart_points = {}

		for dart in self.darts():
			dart_point = self.get_position(dart)
			dart_face_center = self.element_center(dart,2)
			dart_edge_center = self.element_center(dart,1)

			dart_face_point = dart_face_center + coef*(dart_point-dart_face_center)
			dart_face_edge_center = dart_face_center + coef*(dart_edge_center-dart_face_center)

			dart_edge_point = dart_face_edge_center + coef*(dart_face_point-dart_face_edge_center)
			dart_middle_edge_point = dart_face_edge_center + 0.33*(dart_edge_point-dart_face_edge_center)

			dart_points[dart] = [dart_edge_point,dart_middle_edge_point]

			s += pgl.Shape(pgl.Translated(dart_points[dart][0],sphere),coal)
			# s += pgl.Shape(pgl.Translated(np.mean(dart_points[dart],axis=0), pgl.Text(str(dart),fontstyle=font)), coal, id=dart)
			s += pgl.Shape(pgl.Polyline(dart_points[dart],width=2),coal)

		for dart in self.darts():
			alpha_0_points = []
			alpha_0_points += [dart_points[dart][1]]
			alpha_0_points += [dart_points[self.alpha(0,dart)][1]]
			s += pgl.Shape(pgl.Polyline(alpha_0_points,width=5),purple)

			alpha_1_points = []
			alpha_1_points += [0.66*dart_points[dart][0] + 0.33*dart_points[dart][1]]
			alpha_1_points += [0.66*dart_points[self.alpha(1,dart)][0] + 0.33*dart_points[self.alpha(1,dart)][1]]
			s += pgl.Shape(pgl.Polyline(alpha_1_points,width=5),green)

			alpha_2_points = []
			alpha_2_points += [0.33*dart_points[dart][0] + 0.66*dart_points[dart][1]]
			alpha_2_points += [0.33*dart_points[self.alpha(2,dart)][0] + 0.66*dart_points[self.alpha(2,dart)][1]]
			s += pgl.Shape(pgl.Polyline(alpha_2_points,width=5),blue)

		if add : 
			pgl.Viewer.add(s)
		else : 
			pgl.Viewer.display(s)

