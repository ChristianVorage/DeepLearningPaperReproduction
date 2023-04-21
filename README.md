
# Group 74 - Reproduction of the paper :
# "A Voxel Graph CNN for Object Classification with Event Cameras" 

# Team:
| Name | Student Number | E-mail | Tasks |
|------|----------------|--------|--------|
| Alex de Ruijter  |   4666291 |  a.l.deruijter@student.tudelft.nl | Dataloader, Google Cloud, Class System , Code Optimization  |
| Christian Vorage |   4667905 |  c.v.m.m.vorage@student.tudelft.nl | Blogpost, MFRL, SFRL, KNN, Data Preprocessing |
| Paul Kartoidjojo |   4280961 |  p.k.kartoidjojo@student.tudelft.nl | Blogpost, Graph Pooling, Voxelization , Data Converter |


## Sources
The reproduced paper and the data is availble in the following links.  
Data: 
https://www.garrickorchard.com/datasets/n-caltech101 \\
Paper: 
https://arxiv.org/abs/2106.00216



# Objective
The objective of this blog is to reproduce the result [Ours(w/SFRL)] and [Ours], using only the data of N-Cal (presented in table 1 of the paper). For this project, the code was fully reproduced from scratch. 

# 1. Introduction
In the reproduced paper, a Voxel Graph Convolutional Neural Network (VGCNN) is used to classify objects from event based cameras. Unlike conventional cameras that capture images, event based cameras capture changes in light intensity at each pixel, which is represented as negative and positive events. First the event-based camera data (represented in a bin file) is pre-processed into event data, then the vertices are selected and its accompanying featrures calculated. 

The features are calculated using 2 types of layers: the multiscale feature relational layer (MFRL) and the single-scale feature relational layer (SFRL). In the results of the paper,it is shown that including the SFRL will result in outperforming different sorts of methods using the voxel graph convolutional neural network (VGCNN) architecture. 



## Dataset
The paper uses different dataset, i.e. N-MNIST (N-M), NCaltech101 (N-Cal), CIFAR10-DVS (CIF10),
N-CARS (N-C) and ASL-DVS (ASL). 
In this blog, only the N-Cal data is used for the reproduction (as instructed).
Prior to usage of the data, it needs to be preprocessed by converting bin files into usable images.
In the data link, a link is provided for the conversion of the bin file to images.
These were modified for simplicity and convenience of use.
The data is converted into event data, which includes the x and y coordinate, the polarity value (boolean) and a timestamp [ms], $e_i = (x_i, y_i, t_i, p_i)$.
Ordering events based on timestamp range leads to an image, which can be represented as the GIF as seen below.


![picture](
  https://drive.google.com/uc?export=view&id=1F_MMYVssZ3fAJj1f1XbKUp0cLTfCtFbC)
Fig. 1: GIF of 4 samples from distinct classes from the dataset


# Network architecture
The procedure of the paper has the following architecture.
![picture](
  https://drive.google.com/uc?export=view&id=1OpHOmMkbwYPZaC6Wm2MfnmiZ-B7EEk0q)

Fig. 2 - Architecture

First the event data (represented in a bin file) is converted into voxelized event data, then the vertices are selected to calculate its accompanying features. 

The features are calculated using the 2 layers:  the multiscale feature relational layer (MFRL) and the single-scale feature relational layers (SFRL). The output of this data goes through a graph pooling layer. The MFRL and graph pooling layers are repeated in total 3 times. 


Afterwards this goes through another MFRL layer, followed by a non-linear layer, and average and max pooling layers. Each layer will be explained in the section below.


# Voxelisation
When voxelizing an image
each image's dimension (height x width) = ($H\times W$) is standardized to [304 x 240].
The proposed voxel size for the N-Cal dataset is set to $(v_h, v_w, v_a) = (5,5,3)$. 
The position of each voxel/vertex will depend on the spacial position of the events, where each vertex can contain multiple events. 
Vertices with a single/few events, can be noisy, leading to thousands of "useless events".
The paper came up with a simple solution to take the $N_p$  most important vertices. 
For the given dataset, this value is set to $N_p=2048$. The timestamp of each image is different, therefore the timestamps are normalized.  

### Normalize Timestamp
The timestamp is normalized using the following equation:

\begin{equation}
 {t_i}_N = \frac{(t_i-t_0)_N \times A}{t_{N-1}-t_0}
\end{equation}

According to the paper, the compensation coefficient is set to 9 for all datasets. 

### Voxelisation Method
The size of the resulting voxels in spatio-temporal space is: $H_{voxel} = \frac{H}{v_h}, W_{voxel} = \frac{W}{v_w}, A_{voxel} = \frac{A}{v_h}$.
To cut out the $N_p$ vertices, a voxelgrid is made with size $(H_{voxel}, W_{voxel}, A_{voxel})$. 
By iterating through each event, the corresponding vertex's value is increased by one (1) whenever the boolean is true and decreased by (-1) vice versa.


This makes it possible to extract the top $N_p$ vertices from the graph.
Afterwards, a list is made with the position of the top $N_p$ vertices as an item and the corresponding events as a second item. 

In the figure below, an example is given of an event stream of a face.
This event stream is converted into voxels and further refined by vertex selection and resizing. 

![picture](
  https://drive.google.com/uc?export=view&id=1hNMci-MGP_OkZe9PjB2TpKwb5SXB_tBY)
Fig 3: Voxelization 


#Multi-scale feature relational layer (MFRL)
In the next section more details related to SFRL is given. In short the SFRL has the following functions:

1. creating links connection among the vertices in the event based graph. 
2. Computing a scoring matrix for neighbors of each vertex. 
3. Using these information to integrate features from the neighborhood for each vertex.


The MFRL uses 2 SFRL blocks and 1 direct connection. One blocks of the SFRL determines local cues (adjacent neigbors) the other calculates spacial motions determines global changes (distant neighbors). 
Here the MFRL uses the extract features and spatial information from both the adjacent and distant neighbors separately for feature aggregation. 

By computing the scoring matrix based on the spatio-temporal relationships between the vertex and its neighbors, the model can better capture the correlations between the features of the central vertex and its neighbors.
##Single-scale feature relational layer (SFRL)

For each vertex ($\mathcal{V}$) its coordinates ($\mathcal{U}$) and features ($\mathcal{F}$) are fed into the SFRL module.
The coordinates are passed as input to the K-Nearest Neighbors (KNN) Algorithm. Sci-Kit Learn's implementation of KNN is used. 
Within this implementation the choice of algorithm to find the neigbours is BallTree, KDTree or Brute force. 
The choice between these options is based on the structure of the input data.
The distance metric chosen is the euclidian distance in spatio-temporal space (x,y,t).
The weight parameter is chosen as uniform, which means all points in each neighborhood are weighted equally.
Further details can be found on the Sci-Kit Learn website (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

The output of the the KNN-algorithm gives us the $N_{adj}$ neighbours for $K=10$ and $N_{dis}$ neighbours for $K=15$. 
The image below displays the result of the KNN-Algorithm for a single vertex using $ K = N_{adj}$. 
The 10 closest neighbours in spatio-temporal space (itself included) are displayed in red, with respect to the vertex of interest in blue.


![picture](
  https://drive.google.com/uc?export=view&id=1Q0immNmYqol0iJxQClkvwBFn8WXokG1s)

Fig 4: K-Nearest Neighbors Algorithm for single Vertex

The output of the KNN-Algorithm is then used to calculate the scoring matrix $\mathcal{M}_i$ for vertex $\mathcal{V_i}$ as defined in the equation below.

\begin{equation}
\mathcal{M}_i=\mathbb{Q}\left(\underset{j:(i, j) \in \mathcal{E}_i}{\mathbb{S}_G}\left(g_{i, j}\right) ; W_m\right)
\end{equation}

In this equation $g_{i,j} = [\mathcal{U_i}, \mathcal{U_i} - \mathcal{U_j}]$, is the geometric relation (euclidian distance in spatio-temporal space) between a vertex and its neigbours, $\mathbb{S}_G(.)$ is a function that stacks all these geometric relations, and $\mathbb{Q}$ consists of a linear mapping $W_m$ followed by batch normalization and a Tanh funtion.

The obtained scoring matrix $\mathcal{M}_i$ is used to reweight the vertices bases on spatio-temporal distance, when aggregating the neigbours' features for the central vertex. The equation below shows how this is done. 

\begin{equation}
\mathcal{F}_i^{\prime}=\sum \mathcal{M}_i\left(\mathbb{H}\left(\underset{j \in \mathbb{E}_i}{\mathbb{S}_F}\left(\mathcal{F}_j\right) ; W_f\right)\right)
\end{equation}

In this equation $\mathbb{S}_F$ stacks all features from neighbouring vertices. 
$\mathbb{H}$ is a fucntion which consists of a linear mapping $W_f$, a batch normalization, and a ReLU function. The function $\mathbb{H}$ then takes the stacked features as input and produces transformed features. Then, the scoring map $\mathcal{M_i}$ is applied to to re-weight the neighbours' features. The result is the summed up to obtain the aggregated features $\mathcal{F}_i^{\prime}$. 


# Graph pooling
The MFRL already does feature aggregation, therefore the graph pooling layer doesn't require doing that step. In the report $N_r$ random nodes are selected from the output of the MFRL. For the N-Cal dataset, the number of vertices of the graph pooling operator is set to  $N_r = 896$. For the average and max pooling, the standard *torch.mean()* and *torch.max()* are used, due to being one dimensional. 


# Network 

The following parameters are used for the network:


| $\mathbf{N_{adj}^{neigh}}$ | $\mathbf{N_{dis}^{neigh}}$ | Output vertex (graph pooling) | Dropout probability |
|-------------|-------------|----------------------|---------------------|
| 10          | 15          | 896                  | 0.5                |


For overfitting prevention a dropout layer is added with a probability of 0.5. Each fully-connected layer is followed by a LeakyReLU and a Batch Normalization, except for the prediction layer.


# Training 

The model is trained from scratch for 250 epochs by optimizing the crossentropy loss using stochastic gradient descent (SGD) with a base learning rate of $l_r^{init} = 0.1$, while reducing the learning rate until $l_r^{final} = 1 \cdot 10^{-6}$ using cosine annealing. The figure below shows the learning rate plotted against the index of the epochs. The batch size during training is set to 32.

![picture](
https://drive.google.com/uc?export=view&id=1XK2bZKKWNRQ3DdML1Zqb3JNsHYknitbe)

Fig 3: Cosine Annealing Learning Rate Scheduler

# Problems encountered in training
One of the unique challenges of this dataset is that processing the entire dataset and loading it into the RAM used by colab was problematic. The dataset was compressed to about 4.0 Gb of zipped binary event data, which meant that loading it in its entirity filled the RAM. To combat this problem, a alternative dataloader was written in python. First, all data was preprocessed ofline, keeping only the top 2048 voxels, and saving that data in an uncompressed numpy-zip format. That data was then loaded in a lazy manner using generator constructions inherent to python. In that way, the RAM usage for cycling through the data was restricted to only the batch size. Loading the data in this manner was found to have a negligable impact on the epoch training time. (in total +-10 msec)

Another challenge was the training time per epoch. The network needed to iterate through 217 batches of data with a batch size of 32 and a train-test split of 80-20. Per batch of data, the time of a combined forward and backwards pass was found to be about 15 seconds. That ballooned the time per epoch to about 1.5 hours. Profiling the code it was found that most of that time was spent on the calling of the KNN algorithm, which happened once per SFRL module. Sadly, this module has to be rewritten if a better performance is required, but that is left for further research.

# Training Performance

The picture below shows our training performance. Unfortunately we were only able to run the algorithm for 77 epochs, since every epoch took approximately 1.5 hours. The subplot on the left shows the train accuracy in red and the test accuracy in blue plotted agains the index of the epochs. The subplot on the right additionally shows the goal accuracy for reproduction $\approx 74 \%$, indicated with the green dot at the top-right of the graph. The train/test performance in the first ~20 epochs looks quite promising, however we see quite a big divergence between the train and test performance from epoch 20 onward, which seems to indicate overfitting of the model.

![picture](
https://drive.google.com/uc?export=view&id=1DLhymb1f9k6bdXyxIeTTXf4bFQthue5t)




# Result
The authors of the paper uses two methods to determine the accuracy of the event-based object classification, i.e. point-based and frame-based methods. For the reproducibility only the point-based method is computed and compared to their result. 


In the following table the classification accuracy between ours and the paper's result using the point-based methods, with and without SFRL. The result are based on the N-Cal data set. 


|               | Without SFRL | With SFRL |
|---------------|-------------|-----------|
| Paper result (250 epochs)  | 0.737       | 0.748     |
| Our result (10 epochs)    |    -         | 0.30          |


# Conclusion
In our attempt to reproduce the given paper using our own implementation, we can say that it was a successful effort. We were able to replicate the proposed method and achieved results. According to Raff (2019, [1]) a paper can be deemed
to be reproduced if the reproduction results are within 90% of the to be reproduced metric.
At this moment our intermediary result, 30% accuracy (at epoch 80) is well below the $0.9*74\% \approx 67\%$ threshold, which means we cannot say that this paper is reproduced correctly. The algorithm is however still running (120 hours and counting), which means there is still a (small) change it will reach the goal accuracy. 

We faced several challenges during the reproduction process. The preprocessing approach uses the N-Cal dataset, which presented a challenge as it was encoded for the purpose of data compression and the decoding code provided did not function properly. We also encountered a data loading issue on Google Colab, which we addressed by loading the data in batches.

Despite these challenges, the steps outlined in the paper were relatively straightforward and not highly complex. Therefore, the overal experience with reproducing the paper was satisfactory. 

In conclusion, the proposed method of implementing SFRL in the paper appears to be effective, as there is a significant improvement in accuracy. Reproducibility is critical in scientific research, and our efforts to reproduce this paper demonstrate the importance of transparent research practices.

# References

**[1]**  Raff, E. (2019). A Step Toward Quantifying Independently Reproducible Machine Learning Research. In Neural Information Processing Systems (Vol. 32, pp. 5485–5495). https://papers.nips.cc/paper/8787-a-step-toward-quantifying-independently-reproducible-machine-learning-research.pdf
