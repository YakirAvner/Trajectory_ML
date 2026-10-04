import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

still = [
    (np.float64(599.4065259514751), np.float64(250.8934501937016)), 
    (np.float64(601.162193353617), np.float64(250.27493398110727)), 
    (np.float64(599.8692260362222), np.float64(249.63498897809356)), 
    (np.float64(598.8086965325303), np.float64(250.71157575835664)), 
    (np.float64(599.2336988855865), np.float64(248.89916960767894)), 
    (np.float64(599.3365013459754), np.float64(249.30071008557067)), 
    (np.float64(600.6239418624667), np.float64(249.0664244789947)), 
    (np.float64(601.1733552400951), np.float64(249.2534757448037)), 
    (np.float64(600.1592145310736), np.float64(250.3708781016581)), 
    (np.float64(599.0229377752926), np.float64(249.34341963444552)), 
    (np.float64(599.7273194951363), np.float64(250.22390504945378)), 
    (np.float64(599.643265345719), np.float64(251.25924218309174)), 
    (np.float64(600.0284418478454), np.float64(249.14743495403292)), 
    (np.float64(601.1396857099438), np.float64(249.27983367537064)), 
    (np.float64(599.1270265466329), np.float64(248.8391812592964)), 
    (np.float64(600.0341090221646), np.float64(248.50355816721327)), 
    (np.float64(599.489142053586), np.float64(249.6185790932599)), 
    (np.float64(599.9028654803236), np.float64(250.2317195963225)), 
    (np.float64(600.2407126655521), np.float64(250.1996917402291)), 
    (np.float64(599.2917763498682), np.float64(249.56729203351333)), 
    (np.float64(599.8950710785737), np.float64(249.16289101699113)), 
    (np.float64(599.8812044208282), np.float64(251.58668407123926)), 
    (np.float64(602.1899707251808), np.float64(249.44506306619024)), 
    (np.float64(601.9490995559505), np.float64(251.2377018186226)), 
    (np.float64(599.3324902663402), np.float64(252.15734026727532)), 
    (np.float64(600.1285658911293), np.float64(251.1850660305492)), 
    (np.float64(600.9404044903711), np.float64(250.31863289367172)), 
    (np.float64(600.5298673841387), np.float64(249.19297096964442)), 
    (np.float64(600.6972084613691), np.float64(249.98934317754862)), 
    (np.float64(600.4240869929174), np.float64(250.13834109973013)), 
    (np.float64(600.0946906904135), np.float64(250.06993408767477)), 
    (np.float64(598.8146703520564), np.float64(250.73677607219938)), 
    (np.float64(600.6493875264966), np.float64(249.2247093559205)), 
    (np.float64(600.4843488771525), np.float64(250.09221994136846)), 
    (np.float64(599.3134603976398), np.float64(251.2340351320648)), 
    (np.float64(599.9536210677), np.float64(250.73233190664666)), 
    (np.float64(599.2508572887301), np.float64(250.0647797523864)), 
    (np.float64(600.6168849952959), np.float64(249.9836754604254)), 
    (np.float64(600.1883010723379), np.float64(247.98569810940222))
]


windows_csv_file_path = "C:/Trajectory_ML/trajectory_still.csv"
# mac_csv_file_path = "/Users/yakiravner/Trajectory_ML/trajectory_s_curve.csv"
# frame_object = 480
# new_y_axis = frame_object - np.array([point[1] for point in still])  # Invert the y-axis values
# Convert the array to a DataFrame
df = pd.DataFrame(still, columns=['x', 'y'])
# df['y'] = new_y_axis

# Save the DataFrame to a CSV file
df.to_csv(windows_csv_file_path, index=False)

# Plot the trajectory
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal', adjustable='box')
ax.invert_yaxis()  # Invert y-axis to match the coordinate system
ax.set_xlim(0, 1000)
ax.set_ylim(1000, 0)
ax.plot(df['x'], df['y'], marker='o', linestyle='-', color='blue')
ax.set_xlabel('Downrange Distance (m)')
ax.set_ylabel('Altitude (m)')
ax.set_title('Trajectory')
ax.grid(True)
plt.show()