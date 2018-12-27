import numpy as np 
import argparse

import pdb
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time

''' Check the valid the distance between instances, raise exception if the distance exceeds the
    thresh =  2*(2**0.5)*R
'''
def distcheck(a,b,R):
	l = len(a)
	thresh = 2*(2**0.5)*R
	distance = cdist(a,b,'euclidean') 
	dist_list = list(distance[np.triu_indices(l,2)])  #w cdlist return matrix type

	
	if any(i < thresh for i in dist_list):
		raise Exception('Minimum separation violated')	
	#distance.append(dist)
	return np.asarray(distance)

# def main(args):
def main(robots, goals, R):
	# goals = args.goals
	# robots = args.robots
	# goal_locs = np.random.randn(goals,2)*10
	# start_locs = np.random.randn(robots,2)*10
	# traj,unassigned_goals = captbasic(start_locs,goal_locs,0.01)
	# dataN = 10
	# distanceMatrices = np.zeros((dataN, robots*goals))
	# assignmentMatrices = np.zeros((dataN, goals),dtype=int)
	dataN = 1 # the number of the generated matrix
	generateData(robots,goals, R, dataN)
	
	

def captbasic(start,goal,R):
	
	startdistmat = distcheck(start,start,R)
	goaldistmat = distcheck(goal,goal,R)
	dist = distcheck(start,goal,R)
	start = time.time()

	matrix = linear_sum_assignment(dist)
	end = time.time()
	print(end - start)

	pdb.set_trace()
	#add check for distance between start locations
	
	#add check for distance between goal locations

	#compute a distance matrix between all start and goal locations

	#do hungarian optimality assignment
	
"""
This function verifies whether 
the generated points satisfy the contrainsts.
"""
def verifyData(start, goal, R):
	
	# Use a try/exception to find proper locations
	try:
		# Check for distance between start locations
		startdistmat = distcheck(start,start,R)
		# Check for distance between goal locations
		goaldistmat = distcheck(goal,goal,R)
		# Check for distance between start and goal locations
		distance = distcheck(start, goal, R)

		# Get the assignment
		matrix = linear_sum_assignment(distance)

		# if the locations are valid, return the distance matrix and assignment matrix
		return distance, matrix
	except:
		# If the data is not valid, return 0
		return [],[]

"""
Generate specific number of training data.
And write the data into "distanceMatrices.csv" 
and "assignmentMatrices.csv". 
"""
def generateData(robots, goals, R, dataN):
	distanceMatrices = np.zeros((dataN, robots*goals))
	assignmentMatrices = np.zeros((dataN, goals),dtype=int)
	
	while dataN > 0:
		goal_locs = np.random.randn(goals,3)*10
		start_locs = np.random.randn(robots,3)*10

		# sort the sequence of robots according to their distance to the origin
		start_locs[:,2] = np.sqrt(start_locs[:,0]**2+start_locs[:,1]**2) ##
		start_locs = start_locs[np.argsort(start_locs[:,2]),:]   ##

		distance, assignment = verifyData(start_locs[:,:2], goal_locs[:,:2], R)
		if distance != []:
			distanceMatrices[dataN-1, :] = distance.flatten()
			assignmentMatrices[dataN-1, :] = np.array(assignment[1]).astype(dtype=int)
			dataN -= 1
	# with open('distanceMatrices.csv','ab') as f:
	with open('distanceMatrices.csv','ab') as f:
		np.savetxt(f, 
				   distanceMatrices,
				   newline='\n',
				   fmt='%f')
	# with open('assignmentMatrices.csv','ab') as f:
	with open('assignmentMatrices.csv','ab') as f:
		np.savetxt(f, 
				   assignmentMatrices,
				   newline='\n',
				   fmt='%d')



if __name__=='__main__':

	#parser = argparse.ArgumentParser(description='CAPT')

	# parser.add_argument('--goals',type=int,default=1000,help='number of goals')
	# parser.add_argument('--robots',type=int,default=1000,help='number of robots')

	#args = parser.parse_args()

	# main(args)
	start = time.time()
	main(16,16, 0.01)
	end = time.time()
	print(end - start)
	print(5)

	