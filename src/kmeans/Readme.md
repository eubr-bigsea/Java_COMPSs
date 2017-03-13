## Kmeans

Name: Kmeans

Contact Person: lucasmsp@dcc.ufmg.br

Platform: COMPSs

Adaptation from BSC's algorithm


### Description

K-means clustering is a method of cluster analysis that aims to partition ''n'' points into ''k'' clusters in which each point belongs to the cluster with the nearest mean. It follows an iterative refinement strategy to find the centers of natural clusters in the data.

When executed with COMPSs, K-means first splits a input dataset in a number of fragments received as parameter, each fragment being created by an initialization task.

After the initialization, the algorithm goes through a set of iterations. In every iteration, a computation task is created for each fragment; then, there is a reduction phase where the results of each computation are accumulated two at a time by merge tasks; finally, at the end of the iteration the main program post-processes the merged result, generating the current clusters that will be used in the next iteration. Consequently, if ''F'' is the total number of fragments, K-means generates ''F'' computation tasks and ''F-1'' merge tasks per iteration.


###  Versions
There are two versions of Kmeans, depending on the file system where the data was stored.

* Version 1: ''files'', where the files are stored in the usual file system.

* Version 2: ''HDFS'', where the files are stored in the HDFS.


###  Execution instructions
Usage:


	runcompss kmeans.files.KMeans -c <num_clusters> -i <maxIter>  -d <num_dim> -f  <num_frag> -t <train_datafile> -n <size_train>



	runcompss kmeans.HDFS.KMeansHDFS -c <num_clusters> -i <maxIter>  -d <num_dim> -f  <num_frag> -t <train_datafile>


where:

* - num_clusters:
* - maxIter: Number max of iterations
* - num_dim: Dimension of each record
* - size_train: (only in version 1) number of record in the train file
* - num_frag:   (optional in version 2) number of nodes used on the execution
* - train_datafile: File's path used to train the model

### Execution Example

	runcompss kmeans.files.KMeans    -c 2 -d 2 -i 10 -f 2 -n 150 -t /Datasets/Iris/iris_test.data

	runcompss kmeans.HDFS.KMeansHDFS -c 2 -d 2 -i 10 -f 2 -n 150 -t /user/hdfs/iris_test.data

