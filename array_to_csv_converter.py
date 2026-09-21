import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

array = [
    (np.float64(717.6341463414634), np.float64(344.1463414634146)), 
    (np.float64(714.34375), np.float64(344.046875)), 
    (np.float64(717.1428571428571), np.float64(343.92857142857144)), 
    (np.float64(723.0), np.float64(343.6111111111111)), 
    (np.float64(721.140350877193), np.float64(343.03508771929825)), 
    (np.float64(720.28), np.float64(341.68)), 
    (np.float64(717.96), np.float64(339.14)), 
    (np.float64(715.825), np.float64(333.95)), 
    (np.float64(713.8823529411765), np.float64(331.2549019607843)), 
    (np.float64(711.8867924528302), np.float64(328.6792452830189)), 
    (np.float64(709.328125), np.float64(324.234375)), 
    (np.float64(706.82), np.float64(320.84)), 
    (np.float64(703.9032258064516), np.float64(317.1290322580645)), 
    (np.float64(701.5263157894736), np.float64(314.2368421052632)), 
    (np.float64(698.0185185185185), np.float64(308.94444444444446)), 
    (np.float64(695.3170731707318), np.float64(305.1463414634146)), 
    (np.float64(692.5), np.float64(301.5)), 
    (np.float64(689.8648648648649), np.float64(297.7027027027027)), 
    (np.float64(685.34375), np.float64(291.46875)), 
    (np.float64(682.6333333333333), np.float64(287.3666666666667)), 
    (np.float64(679.5), np.float64(283.0)), 
    (np.float64(676.4137931034483), np.float64(278.41379310344826)), 
    (np.float64(671.3225806451613), np.float64(271.35483870967744)), 
    (np.float64(667.875), np.float64(266.6666666666667)), 
    (np.float64(664.3461538461538), np.float64(261.4230769230769)), 
    (np.float64(660.5333333333333), np.float64(256.3)), 
    (np.float64(655.04), np.float64(248.24)), 
    (np.float64(651.0), np.float64(242.5)), 
    (np.float64(646.96), np.float64(236.76)), 
    (np.float64(642.8571428571429), np.float64(231.14285714285714)), 
    (np.float64(636.0869565217391), np.float64(221.6086956521739)), 
    (np.float64(631.8181818181819), np.float64(215.4090909090909)), 
    (np.float64(626.8571428571429), np.float64(209.14285714285714)), 
    (np.float64(622.1111111111111), np.float64(202.11111111111111)), 
    (np.float64(614.6315789473684), np.float64(191.57894736842104)), 
    (np.float64(609.6315789473684), np.float64(184.57894736842104)), 
    (np.float64(604.2352941176471), np.float64(177.11764705882354)), 
    (np.float64(598.5294117647059), np.float64(169.2941176470588)), 
    (np.float64(589.9090909090909), np.float64(157.4090909090909)),
    (np.float64(584.1), np.float64(149.35)), 
    (np.float64(577.95), np.float64(140.85)), 
    (np.float64(571.7368421052631), np.float64(132.4736842105263)), 
    (np.float64(562.2352941176471), np.float64(119.11764705882354)), 
    (np.float64(555.8888888888889), np.float64(110.0)), 
    (np.float64(549.1176470588235), np.float64(100.88235294117646)), 
    (np.float64(542.8125), np.float64(92.1875)), 
    (np.float64(532.7647058823529), np.float64(78.29411764705883)), 
    (np.float64(525.8), np.float64(68.93333333333334)), 
    (np.float64(519.375), np.float64(60.1875)), 
    (np.float64(508.8888888888889), np.float64(46.0)), 
    (np.float64(501.75), np.float64(36.7)), 
    (np.float64(494.8421052631579), np.float64(26.842105263157897)), 
    (np.float64(487.8333333333333), np.float64(17.61111111111111)), 
    (np.float64(477.38461538461536), np.float64(3.0))
]
csv_file_path = "/Users/yakiravner/Trajectory_ML/trajectory#1.csv"
frame_object = 480
new_y_axis = frame_object - np.array([point[1] for point in array])
# Convert the array to a DataFrame
df = pd.DataFrame(array, columns=['x', 'y'])
df['y'] = new_y_axis

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

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