#!/usr/local/bin/python3
#
# decision_tree_classifier.py : a image classifier
#
# Submitted by : Paventhan Vivekanandan, username : pvivekan
#
#

import sys
import numpy as np
import math
from queue import *
import collections


class DTree(object):
	def __init__(self, label, parameter_index):
		self.child = {}
		self.label = label
		self.tag = None
		self.child_labels = None
		self.index = parameter_index
		self.data = None
		self.pos = None
		self.left_c = None
		self.right_c = None
		self.parent = None
		self.isLeaf = False
		
def print_tree_preorder(root):
	if root is None:
		return
	
	if root.isLeaf == False:
		print (str(root.index) + " ", end='')
	else:
		print (str(root.tag) + " ", end='')
	
	if root.left_c is not None:
		print_tree_preorder(root.left_c)
		
	if root.right_c is not None:
		print_tree_preorder(root.right_c)
	
		
def print_tree(root, depth):	
	q = Queue(maxsize=0)
	q.put(root)
	level = 0		

	print("{}\n".format(root.index))
	
	while level <= depth-1 and q.empty() == False:
		qc = Queue(maxsize=0)
		
		while q.empty() == False:
			dtree_p = q.get()
			
			if (dtree_p.left_c is not None):
				if dtree_p.left_c.isLeaf == False:
					print (str(dtree_p.left_c.index) + " ", end='')
					qc.put(dtree_p.left_c)
				else:
					print(str(dtree_p.left_c.tag) + " ", end='')

			if (dtree_p.right_c is not None):
				if dtree_p.right_c.isLeaf == False:
					print (str(dtree_p.right_c.index) + " ", end='')
					qc.put(dtree_p.right_c)
				else:
					print(str(dtree_p.right_c.tag) + " ", end='')
			
			print("   ", end='')
		print("\n")
		q = qc
		level = level + 1
		
	while q.empty() == False:
		
		dtree_l = q.get()
		print(dtree_l.tag + "- ", end='')

def read_tree(filename):
	fh = open(filename, 'r')

	tree_d = {}
	data = fh.readline()
	
	values = data.split()
	
	root = DTree(None, int(values[0]))
	root.isLeaf = False	
	tree_d[int(values[0])] = root
	
	for line in fh:
		values = line.split()
		parent = tree_d[int(values[1])]
		index_tag = int(values[2])
		label = int(values[3])
		dtree_c = None
		
		if values[0] == "N":
			dtree_c = DTree(label, index_tag)
			dtree_c.isLeaf = False
			#dtree_c.label = label
		elif values[0] == "L":
			dtree_c = DTree(label, None)
			dtree_c.tag = index_tag
			dtree_c.isLeaf = True
			#dtree_c.label = label
	
		if dtree_c is not None:
			dtree_c.parent = parent
			
			if dtree_c.label == 0:
				parent.left_c = dtree_c
			else:
				parent.right_c = dtree_c

			if dtree_c.isLeaf == False: 
				tree_d[dtree_c.index] = dtree_c
				
	return root

def read_data(filename):
	fh = open(filename, 'r')
	data = []
	pos = []
	img = []

	for line in fh:
		values = line.split()
		img.append(values[0])
		pos.append(values[1])
		#k = 2
		#pixels = []
		#for p in range(0, 192):
		#	curr_pixel = []
		#	for rgb in range(0, 3):
		#		curr_pixel.append(values[k])
		#		k = k+1
		#	pixels.append(curr_pixel)
		
		data.append(values[2:])

	return (data, pos, img)

def write_data(filename, root, depth):
	fh = open(filename, 'w')
	
	fh.write(str(root.index) + "\n")
	
	q = Queue(maxsize=0)
	q.put(root)
	level = 0

	while level <= depth-1 and q.empty() == False:
		qc = Queue(maxsize=0)
		
		while q.empty() == False:
			dtree_p = q.get()
	
			for i in range(0, len(dtree_p.child)):
				dtree_c = dtree_p.child[i]
				
				if dtree_c.isLeaf == False:
					fh.write("N " + str(dtree_c.parent.index) + " " + str(dtree_c.index) + " " +  str(dtree_c.label) + "\n")
					
					qc.put(dtree_c)
				else:
					fh.write("L " + str(dtree_c.parent.index) + " " + str(dtree_c.tag) + " " + str(dtree_c.label) + "\n")

		q = qc
		level = level + 1
				
# 	tags = ""
# 	while q.empty() == False:
# 		dtree_l = q.get()
# 		tags = tags + str(dtree_l.tag) + "-"
# 		
# 	fh.write(tags)
	
	fh.close()

def construct_decision_tree(data, pos, depth):
	print("Decision Tree depth -> {}".format(depth))

	skip_pm = [] #[2,23,170,191]
	#data = data.reshape(-1, 1)
	(min_entropy_p, labels) = calculate_disorder(data, pos, skip_pm)
	skip_pm.append(min_entropy_p)

	dtree = DTree(0, min_entropy_p)	
	#data_new = np.delete(data_new, min_entropy_p, 1)
	dtree.data = np.copy(data)
	dtree.pos = np.copy(pos)
	dtree.child_labels = labels
	dtree.child = dtree.child.fromkeys(labels)
	dtree.isLeaf = False
	
	q = Queue(maxsize=0)
	q.put(dtree)
	#q.put("$") # $ indicates end of level
	level = 0
	
	print("Decision Tree Root -> {}".format(min_entropy_p))
	
	while level <= depth-1 and q.empty() == False:

		qc = Queue(maxsize=0)
		
		while q.empty() == False:
			dtree_p = q.get()
			
			for i in range(0, len(dtree_p.child)):
				data_s = dtree_p.data[np.where(dtree_p.child_labels == i)[0]]
				pos_s = dtree_p.pos[np.where(dtree_p.child_labels == i)[0]]
				
				cluster_label = i
				if (len(data_s) < 2) or \
				   ((collections.Counter(pos_s)[np.bincount(pos_s).argmax()]) == len(pos_s)):
					if (len(data_s) != 0):
						dtree_c = DTree(cluster_label, None)
						dtree_c.tag = np.bincount(pos_s).argmax()
						dtree_c.isLeaf = True
						dtree_c.parent = dtree_p
						#dtree_c.label = cluster_label
						dtree_p.child[cluster_label] = dtree_c
						print("parent -> {}, level -> {}, leaf -> {}".format(dtree_p.index, level, dtree_c.tag))
				else:
					#data_s = data_s.reshape(-1, 1)
					(n_min_entropy_p, n_labels) = calculate_disorder(data_s, pos_s, skip_pm)
					
					if n_min_entropy_p == -1:
						dtree_c = DTree(cluster_label, None)
						dtree_c.tag = np.bincount(pos_s).argmax()
						dtree_c.isLeaf = True
						dtree_c.parent = dtree_p
						#dtree_c.label = cluster_label
						dtree_p.child[cluster_label] = dtree_c
						continue
					
					skip_pm.append(n_min_entropy_p)
					dtree_c = DTree(cluster_label, n_min_entropy_p)
					dtree_c.data = data_s #np.delete(data_s, n_min_entropy_p, 1)
					dtree_c.pos = pos_s
					dtree_c.child_labels = n_labels
					dtree_c.isLeaf = False
					#dtree_c.label = cluster_label
					dtree_c.child = dtree_c.child.fromkeys(n_labels)
					dtree_c.parent = dtree_p
					dtree_p.child[cluster_label] = dtree_c
					
					
					print("parent -> {}, level -> {}, node -> {}".format(dtree_p.index, level, n_min_entropy_p))
					qc.put(dtree_c)
		
		q = qc
		level = level + 1
		
	while q.empty() == False:
		dtree_l = q.get()
		dtree_l.isLeaf = True
		dtree_l.tag = np.bincount(dtree_l.pos).argmax()
		print("parent -> {}, level -> {}, leaf -> {}".format(dtree_l.parent.index, level, dtree_l.tag))
		
	return dtree

def classify_data(x):
	if x >= 127:
		return 1
	else:
		return 0
	

def calculate_disorder(data, pos, skip_pm):
	
	min_disorder = sys.maxsize
	min_disorder_p = -1
	min_disorder_l = None
	#kmeans = KMeans(n_clusters=nm_clusters, n_init=10) #, n_init=3)
	pos_class = [0, 90, 180, 270]
	
	for i in range(0, len(data[0])):
		if i not in skip_pm:
			C = data[:,i]
			#C = C.reshape(-1, 1)
			
			#kmeans_m = kmeans.fit(C)
			cluster_labels = np.array(list(map(lambda x : classify_data(int(x)), C))) #kmeans_m.labels_
			#cluster_centroids = kmeans_m.cluster_centers_
			label_map = list(zip(cluster_labels, pos))
			
			sumD = 0
			n_t = len(cluster_labels)
			for j in [0,1]:
				n_j = np.count_nonzero(cluster_labels == j)
				c_r = n_j/n_t
				
				in_sum = 0
				for k in range(0, len(pos_class)):
					n_j_c = len(list(filter(lambda x: (x[0] == j) and (x[1] == pos_class[k]), label_map)))
					
					if n_j_c != 0 and n_j != 0:
						in_sum = in_sum + (- (n_j_c/n_j) * math.log (n_j_c/n_j, 2))
				
				sumD = sumD + (c_r * in_sum)
				
			if min_disorder > sumD:
				min_disorder = sumD
				min_disorder_p = i
				min_disorder_l = cluster_labels
	
	return (min_disorder_p, min_disorder_l)

def classify(tree, data, label, img, filename):
	root = tree
	tag = -1
	c = 0
	k = 0
	
	fh = open(filename, 'w')
	
	correctly_classified = 0
	
	for j in range(0, len(data)):
			
		while root.isLeaf == False:
				
			p = data[j][root.index]
				
			if int(p) >= 127:
				root = root.right_c
			else:
				root = root.left_c
				
		tag = int(root.tag)
			
		if tag != -1:
			k = k + 1
		
		if int(label[j]) == tag:
			correctly_classified = correctly_classified + 1
		else: # tag != -1:
			c = c + 1
	
		root = tree
		fh.write(img[j] + " " + str(tag) + "\n")
	
	print("Correctly classified: {} out of {}".format(correctly_classified, k))
	print("Incorrectly classified: {} out of {}".format(c, k))
	
	classification_accuracy = (correctly_classified/k) * 100
	
	print("Classification accuracy: {}%".format(round(classification_accuracy, 2)))
	
	fh.close()


def tree_classifier(run_type, data_file, model_file):
	depth = 8

	if run_type == "train":	
		(data, pos, img) = read_data(data_file)
		pixel_map = np.array(data) #.reshape(-1, 1)
		
		pos = list(map(lambda x : int(x), pos))
		pos = np.array(pos)
	
		dtree = construct_decision_tree(pixel_map, pos, depth)
		write_data(model_file, dtree, depth)
	else:
		root = read_tree(model_file)
		
		print("########### DECISION TREE [LEVEL-ORDER TRAVERSAL] ###########\n")
		print_tree(root, depth)
		print("--------------------------------------------------------------------------")
		print("INORDER TRAVERSAL: ", end='')
		print_tree_preorder(root)
		print("\n--------------------------------------------------------------------------\n")
		
		(data, pos, img) = read_data(data_file)
		pixel_map = np.array(data)
	
		classify(root, data, pos, img, "output.txt")
		
		