import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("MatchingPursuit2dData.txt") # Set of Data

print(data.shape)



atom1 = np.loadtxt("MatchingPursuitDictonary.txt")[0:5]
atom2 = np.loadtxt("MatchingPursuitDictonary.txt")[5:10]
atom3 = np.loadtxt("MatchingPursuitDictonary.txt")[10:15]
atom4 = np.loadtxt("MatchingPursuitDictonary.txt")[15:20]
atom5 = np.loadtxt("MatchingPursuitDictonary.txt")[20:25]
atom6 = np.loadtxt("MatchingPursuitDictonary.txt")[25:30]
atom7 = np.loadtxt("MatchingPursuitDictonary.txt")[30:35]
atom8 = np.loadtxt("MatchingPursuitDictonary.txt")[35:40]
atom9 = np.loadtxt("MatchingPursuitDictonary.txt")[40:45]
atoms = [atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, atom9]

# This is done to compress all given atoms into a dictionary of data
D = np.column_stack([a.flatten() for a in atoms])

# At this point all data is gathered, now the actual matching pursuit logic can be written


# Function Starts Here:
def matching_pursuit(data, D, num_atoms):


    # This sets the numbers of signals and their lenght to the data shape
    # It should be noted that in this example. data shape is (30,9)
    num_signals, signal_length = data.shape
    num_dict_atoms = D.shape[0]

    # This is the normalization formula that changes the D matrix
    # This makes each atom equal to the unit length
    # And this makes sure that the function works on shape similar and not size similarity
    D_norm = D / np.linalg.norm(D, axis=0, keepdims=True)

    # Sets the weight of both coefficents and the reconstructed signal
    coefficients = np.zeros((num_signals, num_dict_atoms))
    reconstruction = np.zeros_like(data)

    for i in range(num_signals):

        x = data[i] # Gets a specific indexed point of data
        residual = x.copy() # The amount of data that remains in the signal

        selected_atoms = [] # The atoms choosen to recreate the signal

        for _ in range(num_atoms): #Greedy aproximation loop

            # Matrix multiplication between dictionary and residual signal
            # These dot products measure similarity
            correlations = D_norm @ residual

            # Finds the atom with the most similarity and assigns it to best_atom
            best_atom = np.argmax(np.abs(correlations))

            # Assigns the index best_atom to the system (called atom_weight)
            #If atom_weight is large the residual shape is similar, if its small there is little similarity
            atom_weight = correlations[best_atom]

            # Changes weight of atom using best_atom and the index
            coefficients[i, best_atom] += atom_weight

            # Sums together the atom used and its weight
            atom_contribution = atom_weight * D_norm[best_atom]

            # Updates reconstructed signal with the contribution of that atom
            reconstruction[i] += atom_contribution

            # Update residual signal by decreasing the reconsturction from it
            residual = x - reconstruction[i]

            # Adds to the selected atoms
            selected_atoms.append(int(best_atom))

        # Prints all data collected
        print(f"\nIteration {i + 1}")
        print("Best atom:", best_atom)
        print("Coefficient (atom_weight):", atom_weight)
        print("Residual norm:", np.linalg.norm(residual))
        print("Residual:", residual)
        print("Current reconstruction:", reconstruction[i])
        print("\nAtoms used to reconstruct signal:", selected_atoms)


    return coefficients, reconstruction


# CALLS MATCHING PURSUIT FUNCTION
coefficients, reconstruction = matching_pursuit(data, D, num_atoms=4)

print("Data shape:", data.shape)
print("Dictionary shape:", D.shape)
print("Coefficient shape:", coefficients.shape)
print("Reconstruction shape:", reconstruction.shape)

import matplotlib.pyplot as plt

num_signals = data.shape[0]

rows = 5
cols = 6

plt.figure(figsize=(15, 10))

for i in range(num_signals):
    plt.subplot(rows, cols, i + 1)
    
    plt.plot(data[i], label="orig", linewidth=1)
    plt.plot(reconstruction[i], linestyle="--", linewidth=1)
    
    plt.title(f"{i}", fontsize=8)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()