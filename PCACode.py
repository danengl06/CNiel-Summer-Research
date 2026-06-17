import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("PCAData.txt")


# FUNCTION START
def PCA (D, k):
    # Takes average down each column (Axis=0 tells it to take means by column)
    means = np.mean(D, axis=0)

    # Subtracts that calculated mean from the associated column
    DataCentered = D - means

    # Finding Covarience by matrix multiplication
    # D.shape[0] finds the number of rows in the data to use as a variable
    # This whole set is just using a known equation
    covariance = (DataCentered.T @ DataCentered) / (D.shape[0] - 1)

    # Finding Eigenvalues from covarience matrix using general eigenvalue solving function
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sorts found eigenvalues from highest to lowest
    # This is needed due to NumPy always out putting these values in assending order
    indices = np.argsort(eigenvalues)[::-1]

    # Replaces eigenvalues with the sorted values
    # In this code each column has one eigen vector
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # Sets up new cordinate system using the top k components as it's new axis
    principle_components = eigenvectors[:, :k]

    # Project the data from the principle components into the main data
    # This is done by matrix multiplication
    TransformedData = DataCentered @ principle_components

    print("Means\n", means)
    print("Eigenvalues\n", eigenvalues)
    print("principle_components\n", principle_components)
    print("TransformedData\n", TransformedData)

    return


(PCA(data, 6))