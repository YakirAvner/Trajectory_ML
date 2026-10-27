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
        1. What is the shape of the trajectory,
        2. Is it a missile or not, 
        3. What is the predicted trajectory after index n-1. 

**1. Trajectory Shape Identification Stage:**

**PyTorch Stage:**

    1. The input data is the raw (x, y) data inputted by the user. Before training, every trajectory is moved so it starts at
       (0, 0), and all trajectories are divided by one shared scale. This keeps each trajectory's real size and aspect ratio,
       so a short jitter and a long flight don't end up looking the same.

    2. The data is then sent to the LSTM Autoencoder. This encodes the data by compressing it to a size of 32 number array.

    3. Then, it decodes the compressed vector and calculates the Mean-Squared-Error (MSE) loss between the input 2D array and the
       decoded 2D array. MSE is an error score, not a percentage: the goal is an MSE as close to 0 as possible, meaning the decoded
       data is nearly identical to the input.

    4. Afterwards, the compressed 32 number array is sent to the Scikit-Learn model to group trajectories with similar shapes and
       mark the ones that fit no group as noise.

      **Encoder:** 

         1. The encoder receives the inputted data from the user; the input is an array of different trajectories,

         2. It reads the trajectory one point at a time, starting from the initial point. At each step the LSTM combines the new point
            with its internal memory, so it learns the movement from one point to the next.

         3. This process is repeated until the internal memory of the LSTM has reached the last index (len(n) - 1). The LSTM doesn't care
            if the length of each trajectory is different from one another (padding is ignored by packing the sequences).

         4. Finally, after reaching the last index of the trajectory, the LSTM's final internal memory (hidden state) is extracted.
            This final state is not only the last movement: it is a summary of the WHOLE path, built up step by step from every
            point read. The extracted vector is sent to the **Latent Space** stage.

      **Latent Space:**

         1. No matter how long the trajectory is, the extracted hidden state is always a 32 number array. Trajectories with similar
            shapes produce similar 32 number arrays, which is what the clustering stage relies on.

      **Decoder:**

         1. The decoder receives the 32 number array from the latent space; the objective is to reverse the process, decoding the 32
            number array into a new array that rebuilds the original data as accurately as possible.

         2. The same 32 number array is fed into the decoder LSTM at every step, once per point of the original trajectory (the length
            is known from the input, without revealing the coordinates themselves).

         3. The decoder LSTM keeps its own 32-size memory across steps, so each step knows where the previous steps were. A Linear layer
            turns each step's memory into one predicted (x, y) coordinate pair.

         4. Finally, a MSE loss function is used to score how the decoded array differs from the input data (padded positions are
            excluded from the score). The model then learns using **backpropagation**: for the current error score, it calculates how
            much each internal weight contributed to that error, and the optimizer (Adam) nudges every weight in the direction that
            lowers the error. Repeating this over many training rounds (epochs) and more data makes the reconstruction more precise.
         

**Scikit-Learn Stage:** 
    1. The input data is the 32 number array of each trajectory, produced by the trained encoder.

    2. The model uses a Hierarchical Density Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm. Each
       trajectory is one point in a 32-dimensional space, and HDBSCAN looks for areas where many of these points are packed
       closely together (dense areas). Each dense area becomes a cluster, meaning a group of trajectories with a similar shape.

    3. Unlike DBSCAN, HDBSCAN does not need a fixed radius (epsilon). It builds a hierarchy of clusters across all density levels
       and keeps the clusters that stay stable the longest, so it can find clusters with different densities.

    4. The key parameters are:

       4.1. **min_cluster_size**: the smallest number of trajectories that may form a cluster,

       4.2. **min_samples**: how many close neighbors a point needs to count as being in a dense area (a core point). Higher values
            make the model more conservative and label more trajectories as noise.

    5. The output for every trajectory is a cluster label and a membership strength between 0 and 1. Trajectories that don't
       belong to any dense area get the label -1, meaning noise.
