# coding: utf8
from openalea import *
from gmap import *
from gmap_optimization import *
from gmap_tools import *
from array_dict import *
from basicshapes import *

def add_square(gmap):
	darts = [gmap.add_dart() for i in xrange(8)]
	for i in xrange(4):
		gmap.link_darts(0, darts[2*i], darts[2*i+1])
	for i in xrange(4):
		gmap.link_darts(1, darts[2*i+1], darts[(2*i+2) % 8])
	return darts
 
def square():
	gmap = GMap()
	add_square(gmap)
	return gmap

def cube(xsize = 5, ysize  = 5 , zsize = 5):
	g = GMap()
	squares = [add_square(g) for i in xrange(6)]
 
	# sew top square to lateral squares
	g.sew_dart(2, squares[0][0], squares[1][1] )
	g.sew_dart(2, squares[0][2], squares[4][1] )
	g.sew_dart(2, squares[0][4], squares[3][1] )
	g.sew_dart(2, squares[0][6], squares[2][1] )
 
	# sew bottom square to lateral squares
	g.sew_dart(2, squares[5][0], squares[1][5] )
	g.sew_dart(2, squares[5][2], squares[2][5] )
	g.sew_dart(2, squares[5][4], squares[3][5] )
	g.sew_dart(2, squares[5][6], squares[4][5] )
 
	# sew lateral squares between each other
	g.sew_dart(2, squares[1][2], squares[2][7] )
	g.sew_dart(2, squares[2][2], squares[3][7] )
	g.sew_dart(2, squares[3][2], squares[4][7] )
	g.sew_dart(2, squares[4][2], squares[1][7] )
 
	for darti, position in zip([0,2,4,6],[ [xsize, ysize, zsize], [xsize, -ysize, zsize] , [-xsize, -ysize, zsize], [-xsize, ysize, zsize]]):
		dart = squares[0][darti]
		g.set_position(dart, position)
	 
	for darti, position in zip([0,2,4,6],[ [xsize, -ysize, -zsize], [xsize, ysize, -zsize] , [-xsize, +ysize, -zsize], [-xsize, -ysize, -zsize]]):
		dart = squares[5][darti]
		g.set_position(dart, position)
 
	return g
 
 
def holeshape(xsize = 5, ysize = 5, zsize = 5, internalratio = 0.5):
	assert 0 &lt; internalratio &lt; 1
 
	g = GMap()
	squares = [add_square(g) for i in xrange(16)]
 
	# sew upper squares between each other
	g.sew_dart(2, squares[0][2], squares[1][1] )
	g.sew_dart(2, squares[1][4], squares[2][3] )
	g.sew_dart(2, squares[2][6], squares[3][5] )
	g.sew_dart(2, squares[3][0], squares[0][7] )
 
	# sew upper squares with external lateral
	g.sew_dart(2, squares[0][0], squares[8][1] )
	g.sew_dart(2, squares[1][2], squares[9][1] )
	g.sew_dart(2, squares[2][4], squares[10][1] )
	g.sew_dart(2, squares[3][6], squares[11][1] )
 
	# # sew upper squares with internal lateral
	g.sew_dart(2, squares[0][5], squares[12][0] )
	g.sew_dart(2, squares[1][7], squares[13][0] )
	g.sew_dart(2, squares[2][1], squares[14][0] )
	g.sew_dart(2, squares[3][3], squares[15][0] )
 
	# sew lower squares between each other
	g.sew_dart(2, squares[4][6], squares[5][1] )
	g.sew_dart(2, squares[5][4], squares[6][7] )
	g.sew_dart(2, squares[6][2], squares[7][5] )
	g.sew_dart(2, squares[7][0], squares[4][3] )
 
	# sew lower squares with external lateral
	g.sew_dart(2, squares[4][0], squares[8][5] )
	g.sew_dart(2, squares[5][6], squares[9][5] )
	g.sew_dart(2, squares[6][4], squares[10][5] )
	g.sew_dart(2, squares[7][2], squares[11][5] )
 
	# sew lower squares with internal lateral
	g.sew_dart(2, squares[4][5], squares[12][4] )
	g.sew_dart(2, squares[5][3], squares[13][4] )
	g.sew_dart(2, squares[6][1], squares[14][4] )
	g.sew_dart(2, squares[7][7], squares[15][4] )
 
	# sew external lateral squares between each other
	g.sew_dart(2, squares[8][7], squares[9][2] )
	g.sew_dart(2, squares[9][7], squares[10][2] )
	g.sew_dart(2, squares[10][7], squares[11][2] )
	g.sew_dart(2, squares[11][7], squares[8][2] )
 
	# sew internal lateral squares between each other
	g.sew_dart(2, squares[12][2], squares[13][7] )
	g.sew_dart(2, squares[13][2], squares[14][7] )
	g.sew_dart(2, squares[14][2], squares[15][7] )
	g.sew_dart(2, squares[15][2], squares[12][7] )
 
	pos = { 
			(0,0) : [xsize,  ysize,  zsize] ,
			(1,2) : [xsize,  -ysize, zsize] ,
			(2,4) : [-xsize, -ysize, zsize] ,
			(3,6) : [-xsize, ysize,  zsize] ,
 
			(0,5) : [xsize*internalratio,  ysize*internalratio,  zsize] ,
			(1,7) : [xsize*internalratio,  -ysize*internalratio, zsize] ,
			(2,1) : [-xsize*internalratio, -ysize*internalratio, zsize] ,
			(3,3) : [-xsize*internalratio, ysize*internalratio,  zsize] ,
 
			(4,1) : [xsize,  ysize,  -zsize] ,
			(5,7) : [xsize,  -ysize, -zsize] ,
			(6,5) : [-xsize, -ysize, -zsize] ,
			(7,3) : [-xsize, ysize,  -zsize] ,
 
			(4,4) : [xsize*internalratio,  ysize*internalratio,  -zsize] ,
			(5,2) : [xsize*internalratio,  -ysize*internalratio, -zsize] ,
			(6,0) : [-xsize*internalratio, -ysize*internalratio, -zsize] ,
			(7,6) : [-xsize*internalratio, ysize*internalratio,  -zsize] ,
		  }
 
	for darti, position in pos.items():
		sqid, dartid = darti
		dart = squares[sqid][dartid]
		g.set_position(dart, position)
		
	return g
 
#TODO
g = square()
g.display()
#mesh_display(points, triangles)
