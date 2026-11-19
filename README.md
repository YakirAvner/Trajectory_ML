# Trajectory_ML

Main Objective: 
    The code is a combination of PyTorch and Scikit-Learn to build an unsupervised trajectory machine learning (ML) model; capable 
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

================================

**Status - The current code is a model for section "1" in the conclusion.** 

================================

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



**Code Structure:**
1. Setup (lines 1-219)
      
      Imports the libraries, loaded trajectories, and fixes the settings: 600 training steps, 6 copies per step, 32 points per trajectory, a 32-number shape code, 4 transformer passes and the target canvas size. It also fixes the random seeds so every run gives the same result.

2. Data preparation (lines 321-387)

      Redraws each trajectory as 32 points evenly spaced along its path, so the original number of points doesn’t matter.
      Shifts each trajectory to start at (0, 0).
      Divides all trajectories by one shared number (the median trajectory size), so values sit around 1 while relative sizes are kept.
      Makes a backwards copy of each trajectory.
      Packs everything into tensors, with a mask (real versus padding points), τ (position along the path, 0 to 1) and lengths.

3. The model (lines 231-314)

      Localization network: a small neural network that estimates a trajectory’s position, size and rotation.
      Spatial transformer: runs the localization network 4 times. Each pass removes the remaining pose and adds it to a running total. The result is the canonical form plus the total pose.
      LSTM encoder: reads the canonical form and compresses it into a 32-number shape code.
      LSTM decoder: redraws the canonical form from the shape code, inside a fixed −1 to 1 box.
      Pose put back: the learned position, size and rotation are re-applied to the redrawn form.

4. Training helpers (lines 357-387)

      random_pose gives each trajectory a random rotation, size (×1/3 to ×3) and position.
      masked_spread measures a trajectory’s size.
      size_fair_error measures rebuild error relative to size.
      batch_of_copies builds each batch: 6 copies of every trajectory, randomly forwards or backwards, each in two different random poses.

5. Training (lines 394-436), 600 steps
      
      Each step makes a fresh batch of 240 posed trajectories, runs them through the model, and scores the result with three losses:

      reconstruction: can the decoder redraw the shape from the shape code?
      agreement: do two poses of the same trajectory get the same canonical form?
      canvas: is the canonical form near the target size?

      The optimizer (AdamW, with a cosine learning-rate schedule) then adjusts the 26,119 weights to lower the total. Every 25 steps a GIF frame is saved, and every 50 steps progress is printed.

6. Measuring (lines 443-456)
      
      Using the trained model:

      the shape code of each of the 20 trajectories;
      the noise score: each trajectory’s rebuild error, averaged over 32 random poses;
      the invariance check: how much a shape code moves when only pose changes, compared with the typical distance between different trajectories.

7. Labelling (lines 458-503)

      Noise: the noise scores are split into a low and a high group at the clearest gap. If the high group’s errors are more than 3× the low group’s, it’s noise.
      Shape clusters: HDBSCAN groups the shape codes of the remaining trajectories. Any it leaves out also become noise.
      Kinds of noise: HDBSCAN groups the noise trajectories among themselves.
      Printout: each trajectory’s label, cluster strength and noise score.

8. Saving and figures (lines 510-557)

      Saves the trained weights to st_output/st_lstm_autoencoder.pt.
      results.png: each raw trajectory coloured by its label, with its canonical form and the decoder’s rebuild underneath.
      training_simulation.gif: the canonical forms changing as training progressed.
