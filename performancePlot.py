import seaborn
import numpy as np
import seaborn as sns
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

def plotDistance(iterations, optimalDistance, totalDistances):
    """
    (INPUT) iterations: (N,) numpy array of iteration numbers
    (INPUT) optimalDistance: the total distances computed 
                             using hungarian algorithm 
    (INPUT) totalDistances: (N,) numpy aaray of totalDistances 
                            for corresponding iterations
    """

    x = iterations
    y = np.ones_like(x) * optimalDistance
    z = totalDistances 
    data = {'x':x.astype(int), 
            'Optimal Distance':y, 
            'Predicted Distance':z}
    data = pandas.DataFrame(data)
    ax = sns.lineplot(data=data,hue='coherence',style='choice')
    # axes = ax.
    plt.ylim(0,53)
    plt.show()



if __name__ == '__main__':
    # distanceMat, assignmentMat, _assignment_onehot = loadTrainingData(r'distanceMatricesTest.csv',
                                                                    #   r'predictedAssignmentMatrices.csv')
    # distanceMat = np.sum(distanceMat, axis=-1)
    # totalDistances = computeDistance(distanceMat, assignmentMat)
    # print(totalDistances)
    # distanceMatrices = np.loadtxt(r'distanceMatrices.csv',dtype=float)
    # assignment = np.loadtxt('assignmentMatrices.csv',dtype=int)
                        
    x = [i for i in range(10,110,10)]
    b = [5270784.088698037,
    5264513.920349974,
 5267257.39134099,
 5253386.147280009,
 5272599.498772029,
 5262018.43587501,
 5262365.122317011,
 5269033.400811051,
 5255349.365995008,
 5248668.1135279415]
    a = 4558860.143774011
    # c = 4558860.143774011
    d = [4682115.487961023,
        4663917.711252073,
        4663350.176667067,
        4654199.963472059,
        4671670.329962055,
        4667617.328513025,
        4659446.424542045,
        4651658.220157067,
        4656313.687618041,
        4646471.983179057,
        4648829.53113705]
    data = {'x':x, 
            'Optimal test Distance':a, 
            'Predicted test Distance':b}
            # 'Optimal train Distance':c,
            # 'Predicted train Distance':d}
    data = pandas.DataFrame(data)
    ax = sns.lineplot(data=data,
                      hue='coherence',
                      style='choice')
    plt.show()
