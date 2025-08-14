import sys
sys.path.append('C:/Program Files/MATLAB/R2019b/extern/engines/python')
import matlab.engine
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io


# Start MATLAB engine
#eng = matlab.engine.start_matlab()

# Run the MATLAB script
#eng.run('C:/Users/sica/Downloads/mm-wave-master/mm-wave-master/client_center.m', nargout=0)

# Initialize the loop counter
i = 1

# Initialize an empty list to hold all points
all_points = []

# While loop to continuously retrieve points and append them
while True:
    eng = matlab.engine.start_matlab()
    eng.run('C:/Users/sica/Downloads/mm-wave-master/mm-wave-master/client_center.m', nargout=0)
    # Retrieve the variable 'A' from MATLAB
    points = eng.workspace['A']

    # Convert MATLAB array to a Python list or NumPy array (depending on the type)
    points = np.array(points)  # Convert to a NumPy array (adjust if you need a list)

    # Append the points to the all_points list
    all_points.append(points)

    # Optionally, print the saved points (for debugging purposes)
    print(f"Iteration {i}: {points}")

    # Increment the loop counter
    i += 1
    eng.quit()

    # Wait for a short time before the next iteration
    time.sleep(0.1)
    print("20s sleep")

    # You can add a condition to break the loop after a certain number of iterations
    # For example, after 100 iterations:
    if i > 60:  # Adjust this as needed
        break

# Convert the list of points to a single NumPy matrix
all_points_matrix = np.vstack(all_points)  # Stack arrays vertically to form a single matrix

# Save the matrix as a MATLAB .mat file
scipy.io.savemat('all_points.mat', {'all_points': all_points_matrix})
#eng.save('C:\Users\sica\Desktop\all_points.mat', nargout=0)


