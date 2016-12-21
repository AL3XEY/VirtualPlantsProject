# coding: utf8
class Skeleton(object):
	def __init__(self):
		self.maxid = 0
		self.segment_points = {}
		self.segment_sizes = {}

	def segments(self):
		return self.segment_points.keys()
