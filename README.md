# Trajectory_ML

Main Objective: 
    The code, a combination of PyTorch and Scikit-Learn, is an unsupervised trajectory machine learning (ML) model; capable 
    of classifying 2D array shaped trajectories containing raw (x, y) coordinates (meaning any array length) which are inputted in the model. 

Code Components contain 2 stages: 
    1. PyTorch - This stage will provide a deep learning (DL) model capable of standardizing geometrical shapes of
                 the 2D trajectory vector inputted in the model. 
    2. Scikit-Learn - In this stage, the system will provide a ML analysis of labelling the 2D trajectory vectors without
                      any human-labelling classification. As a result, this gives an advantage of training the model with real data 
                      and constructing confidence for the system to understand different maneuvers.

    In conclusion, the system is a confidencial way for existing systems to indicate in real-time:
        1. What the shape of the trajectory is,
        2. Is it a missile or not, 
        3. What is the predicted trajectory after index n-1. 

PyTorch Stage: 
    1. The input data must be the raw data inputted by the user for optimal analysis and training,
    2. The model will use a Density Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm. This algorithm
       uses a circled shape called an **"Epsilon Neighborhood" (EN)** which is placed randomly on a graph of points, and groups the points to automatically trace the shape by moving the EH through the points, in this case, of the trajectory.
    3. The DBSCAN algorithm chooses an unvisited coordinate as its initial point. 
    4. Next, it takes that point and places a circle (the EN) around it and checks how much density is in the radius of the EN
       according to a key parameter called the **minPTs** - initialized by the user - analyzing the number of points in the surface of the 
       circle. 
       4.1. If the number of points (P#) >= minPts: This is called a core point,
       4.2. If P# < minPTs: This is called a border point,
       4.3. If there are points that are part of any trajectory: This indicates as noise in the model
    5. Then, this part is the essential part of the DBSCAN algorithm. When identifying a core point, the EN moves to its neighbors
       creating a new circle  for these neighbor points. For each neighbor point that is identified, as well, as a core point, the EN
       keeps on hopping to each neighbor in the trajectory's coordinates until the value of P# < minPTs; indicating as a border point
       and finishes analyzing the trajectory's shape.   





