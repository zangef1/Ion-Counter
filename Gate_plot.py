import matplotlib.pyplot as plt

def read_data(filepath):
    """
    Reads a two-column .dat file and converts time steps to nanoseconds.

    Parameters:
        filepath (str): Path to the input .dat file

    Returns:
        time_steps (list): Time in nanoseconds
        distance (list): Distance values (e.g., between residues)
    """
    time_steps = []  
    distance = []
    conversion_factor = 400  # Convert time step to picoseconds

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line and line[0] != '#' and line[0] != '@':
                columns = line.split()
                if len(columns) >= 2:
                    # Convert time step to nanoseconds
                    time_ns = int(columns[0]) * conversion_factor / 1000
                    dist = float(columns[1])
                    time_steps.append(time_ns)
                    distance.append(dist)
    return time_steps, distance

# ==== File Paths to Your Distance Data ====
filepath1 = '/Users/yongcheng/Documents/GIRK14/1_3/CHL/G14_1_3_CHL_HBC.dat'     # HBC gate distance
filepath2 = '/Users/yongcheng/Documents/GIRK14/1_3/CHL/G14_1_3_CHL_gloop.dat'    # G-loop gate distance

# ==== Load Data from Files ====
time_steps1, distance1 = read_data(filepath1)  # HBC data
time_steps2, distance2 = read_data(filepath2)  # G-loop data

# ==== Create the Plot ====
plt.figure(figsize=(10, 6))

# Plot HBC distances
plt.plot(time_steps1, distance1, label='HBC', color='red', linewidth=0.5)

# Define permeation frame numbers (from simulation analysis)
permeation_time = [
    56400, 83200, 109200, 147200, 153200, 166000, 182400, 193600, 196000, 259600, 
    381600, 667600, 398800, 447200, 452400, 461600, 466800, 473600, 533200, 660000, 
    672400, 690400, 750800, 752800, 766000, 771600, 779600, 794400, 889600, 897200, 
    901600, 912800, 667600
]

conversion_factor = 400  

# Convert frame numbers to nanoseconds using: (frame * ps/frame) / 1000 = ns
permeation_time_ns = [frame * conversion_factor / 1000 for frame in permeation_time]

# Draw vertical lines for each permeation event
for time_point in permeation_time_ns:
    plt.axvline(x=time_point, color='black', linestyle='-', linewidth=1.0)

# Plot G-loop distances
plt.plot(time_steps2, distance2, label='Gloop', color='blue', linewidth=0.5)

# ==== Styling and Labels ====
legend = plt.legend()
for line in legend.legendHandles:
    line.set_linewidth(20)

plt.xlabel('MD Simulation Time (ns)', fontsize=16)
plt.ylabel('Distance of Gates (Å)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best')

# Adjust plot limits
plt.ylim([0.0, 15]) 
plt.xlim(0, 1000)  # adjust if your sim time exceeds 1000 ns

# ==== Show the Plot ====
plt.tight_layout()
plt.show()
