import seaborn
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import pandas
#from loadTrainingData import loadTrainingData

def computeDistance(distanceMat, assignmentMat):
    """
    (INPUT) distanceMat: NxMxM matrix, storing the distance matrix
    (INPUT) assignmentMat: NxM matrix, storing the resulting assignment matrix
    (OUTPUT) totalDistance: A scaler, stroing the summation of distance of all assignments
    """

    # Initialization
    nSample, nGoal= assignmentMat.shape
    totalDistances = 0

    for n in range(nSample):
        assignment = assignmentMat[n].cpu().numpy()
        distance = distanceMat[n,].cpu().numpy()
        Traj = distance[range(nGoal), assignment]

        totalDistances += np.sum(Traj)
        # Find the maximum cost along the trajectory
        max_dist = np.max(Traj)

        u, counts = np.unique(assignment, return_counts=True)
        #print(assignment)
        #print(u,counts)
        # Find the duplicated assignment and find the least distance
        for assign, count in zip(u, counts):
            if count > 1:
                # Find the duplicated assignment
                dup_goals = np.where(assignment==assign)[0]
                #print("dup_goals: ",dup_goals)
                # Find the minimum distance among these 
                # duplicated assignments
                #print("two goals: " ,Traj[dup_goals])
                min_goal_dist = np.min(Traj[dup_goals])
                # Replace other duplicated assignments' distance 
                # with the maximun distance
                totalDistances -= np.sum(Traj[dup_goals])
                totalDistances += (count)*max_dist
    return totalDistances