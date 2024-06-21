import os.path
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Write airfoil data to file
airfoil_data = np.loadtxt('E63.DAT', skiprows=1)

file = 'polar.dat'
for num in range(1, 100):
    filename = f'polar{num}.dat'
    if not os.path.exists(filename):
        file = filename
        break

# Write commands to file
commands = f"""
LOAD E63.DAT
PANE
PPAR
N 200


OPER 
VPAR
N 200


OPER
VISC 1e6
M 0.2
PACC
{file}

ASeq -8 8 1

QUIT
"""

with open('commands.in', 'w') as f:
    f.write(commands)

# Run XFOIL
process = subprocess.run(['xfoil.exe'], input=commands, text=True, capture_output=True)

# Check for errors
if process.returncode != 0:
    print(f"Error: {process.stderr}")
else:
    print(f"Output: {process.stdout}")

# Read results from polar file
if os.path.exists('polar.dat'):
    results = np.loadtxt('polar.dat', skiprows=12)
    if results == []:
        print('No results found')

    alpha = results[:, 0]
    cl = results[:, 1]
    cd = results[:, 2]

    # Plot results
    plt.figure()
    plt.plot(alpha, cl)
    plt.xlabel('Angle of Attack (deg)')
    plt.ylabel('Lift Coefficient (Cl)')
    plt.title('Lift Coefficient vs. Angle of Attack')
    plt.grid(True)

    plt.figure()
    plt.plot(alpha, cd)
    plt.xlabel('Angle of Attack (deg)')
    plt.ylabel('Drag Coefficient (Cd)')
    plt.title('Drag Coefficient vs. Angle of Attack')
    plt.grid(True)

    plt.figure()
    plt.plot(cd, cl)
    plt.xlabel('Drag Coefficient (Cd)')
    plt.ylabel('Lift Coefficient (Cl)')
    plt.title('Lift Coefficient vs. Drag Coefficient')
    plt.grid(True)

    plt.show()
