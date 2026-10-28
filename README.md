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

    1. The input data must be the raw data inputted by the user for optimal analysis and training.

    2. The data is then sent to the LSTM Autoencoder. This encodes the raw data by compressing it to a size of 32 number array.

    3. Then, it decodes the compressed vector and calculates the Mean-Squared-Error (MSE) loss of the raw 2D array data and decoded
       2D array data. For the result, the decoded data must be as close to being identical as the raw data with a score very close to 100% accuracy of the data.

    4. Afterwards, the compressed 32 number array is sent to the Scikit-Learn model to analyze the shape of each trajectory and eliminate 
       noises.

      **Encoder:** 

         1. The encoder receives the inputted data from the user; the input is an array of different trajectories,

         2. It reads the initial point from the array; then, the next index point is read and it calculates the vector between both 
            points. 

         3. This process is repeated until the internal memory of the LSTM has reached the last index (len(n) - 1). The LSTM doesn't care
            if the length of each trajectory is different from one another. 

         4. Finally, the after reaching the last index of the trajectory it extracts the final vectored shape of the trajectory; ignoring
            the previous vectors analyzed. The extracted vector is sent to the **Latent Space** stage. 

      **Latent Space:**

         1. It captures the extracted vector from the internal memory; no matter how big the length of the trajectory's vector is, it's 
            minimized to a 32 number array. For each index, the coordinate points can change to a higher or lower value.

      **Decoder:**

         1. The decoder receives the 32 number array from the latent space; the objective is to reserve the process decoding the 32 number 
            array to a new array and rebuild accurately the original raw data. 

         2. Next, the decoder predicts what should be the first coordinate pair according to the 32 number array.

         3. When finishing predicting the first coordinate pair, it saves the context to itself and predict the next coordinate until it
            reached the last index (length is known by the encoder without revealing the coordinates themselves).

         4. Finally, a MSE loss function is used to score how the decoded array differs from the raw data. Furthermore, when training the 
            model, the data accumulated from training helps improve new data statictics by using a method called **backpropagation**.
            This terminology refers to how the ML model uses its past traning logs as a performance review and compares the error scores for every trained data to the new data inputted. 
            As a result, the more data it trains on, the better the optimization is for enhancing its knowledge to predict trajectory shapes more precisely. 
         

**Scikit-Learn Stage:** 
    1. The input data must be the compressed data inputted by the autoencoder for optimal analysis and training different clusters.

    2. The model will use a Hierarchial Density Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm. This algorithm takes every point inputted and checks its core distance 
       from every other point in the array. 

    3. 