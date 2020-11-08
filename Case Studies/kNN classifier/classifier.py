import random
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt



p1 = np.array([1,1])
p2 = np.array([4,4])

def distance(p1, p2):
    """return the most common element in votes  """
    d = np.sqrt(np.sum(np.power(p2 - p1, 2)))
    return d 

def majority_vote(votes):
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
        
    max_count = max(vote_counts.values())
        
    winners = []
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)
    return random.choice(winners) #chooses randomly if there is a tie


def majority_vote_short(votes):
    """return the most common element in votes  """
    mode, count = ss.mstats.mode(votes)
    return mode #chooses randomly if there is a tie

points = np.array([[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2], [3,3]])
p = np.array([2.5, 2])

def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        dist = distance(p, points[i])
        distances[i] = dist
    inds = np.argsort(distances) # returns list of indicies that would give the sorted array
    return inds[:k]

def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])

def generate_synth_data(n=50):
    """create two sets of f points from bivariate normal distributions"""
    a = ss.norm(0,1).rvs((n,2))   #mean 0, sd 1
    b = ss.norm(1,1).rvs((n,2))
    points = np.concatenate((a,b), axis = 0) #synthetic observations
    outcomes = np.concatenate((np.repeat(0,n),np.repeat(1,n))) #first group of outcomes are class 0, second group has outcome of 1
    return (points, outcomes)

n = 20
(points, outcomes) = generate_synth_data(n)

# plt.figure()
# plt.plot(points[:n,0], points[:n,1], "ro")
# plt.plot(points[n:,0], points[n:,1], "bo")
# plt.savefig("bivariatedata.pdf")

#Making a prediction grid - used to compute the class of each point that belong to a rectangualr region ofthe predictor space
def make_prediction_grid(predictors, outcomes, limits, h, k):
    """Classifies each point on the prediction grid"""
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys) #generates two 2d arrays of the points of each (x,y) location. The first output is the x-values of the coordinate pairs, the second is the y outputs
    
    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i,x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k) #note that the index is [j,i], not [i,j]. this  is because the first arg is the row of the array
    return (xx, yy, prediction_grid)


def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

from sklearn import datasets
iris = datasets.load_iris()
predictors = iris.data[:, 0:2]   #all rows, but cols  0 and 2 only
outcomes = iris.target
plt.plot(predictors[outcomes==0][:,0],predictors[outcomes==0][:,1], "ro")
plt.plot(predictors[outcomes==1][:,0],predictors[outcomes==0][:,1], "go")
plt.plot(predictors[outcomes==2][:,0],predictors[outcomes==0][:,1], "bo")
plt.savefig("iris.pdf")

k=5; filename = "iris_grid.pdf"; limits = (4,8,1.5,4.5); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, 50, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)