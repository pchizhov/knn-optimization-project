# knn-optimization-project
Repository for Algorithmics project at Tartu University.  

## Project content
Here we present implementation of the main data structures & algorithms used to perform KNN search:  
* KD-Tree - data structure that stores data as a binary tree by making axis-aligned splits of points across one of   
  the dimensions (usually interleaving them across tree levels). Each point is associated with some hyperrectangle;  
* Ball-Tree - data structure that stores data as a binary tree by making splits of points across random vectors that   
  approximate directions of the greatest spread. Each point is associated with some hypersphere;  
* Brute-Force - several-lines vectorized implementation based on the full pairwise distance matrix computation.  

## Repository structure
``` console
- animate                 <- slides of animations presented in Visualizations.ipynb 
- figures                 <- charts produced in TestReport.ipynb
- gifs                    <- animations presented in Visualizations.ipynb 
- src                     <- data structures & algorithms implementations
- TestReport.ipynb        <- performance test & comparison of implemented data structures & algorithms
- Visualizations.ipynb    <- visualisations of implemented data structures
```