import numpy as np
data = np.loadtxt("MarchineLearning8dData.txt", delimiter=",") # Set of Data
k = 4 # Number of Clusters

indices = np.random.choice(len(data), k, replace=False) # Gets random indicies within the set of data

centroids = data[indices] # Indexs those indicies to give the exact 2d set of data
   
print(data.shape) # Should be 40 sets of 2 dimensional data
print("\nStart")
print("Random Centroids =" + str(centroids))

for iteration in range(10):

    labels = [] # A list of labels for each point of what cluster they are in


    # Now with the data randomly put into centroids, we can find cluster numbers
    # This is done by getting the short distance between the point and the centroids


    for point in data:

        distances = [] # A list of distances between centroids

        for centroid in centroids:
            # np.linalg.norm() is a function that computes distnace formula

            newdist = np.linalg.norm(point - centroid) # New var is made with the distance between the given centroid and given point
            distances.append(newdist) # Distance is then added to distance list
        cluster = np.argmin(distances) # The cluster var is set to the the smallest distance value
        labels.append(int(cluster)) # Adds the number cluster to the list for that data point

# At this point, the first set of clusters has been made and each data point is in a cluster
# Now the centroids have to be changed with this new information

    labels = np.array(labels) # Updates Labels
    new_centroids = np.zeros((k,data.shape[1])) # Sets the list of new centroids to zeros, so it can be filled in again

    for i in range(k):
        cluster_points = data[labels == i] # Gets data from the specific  numbered cluster
        if len(cluster_points) > 0:
            # Updates centroids with the mean of all points in that cluster
            new_centroids[i] = np.mean(cluster_points, axis = 0) # Axis must be zero to take mean down colunm-wise
        else:
            # This is the catch case if no points are in a given cluster, it will stop errors
            new_centroids[i] = centroids[i] 


    if np.allclose(centroids, new_centroids): #Converges if centroids and new centroids are close enough
        print("Converged at interation", iteration)
        break

    centroids = new_centroids # sets the centroids to the new calculated values

    print("\nIteration:", iteration)
    print("Centroids:", centroids)
    # After information is printed, the code will loop again with another interation, untill convergence