# Java COMPSs Algorithms

Some algorithms developed in Java COMPSs with or without using the HDFS's API developed in this project.


-----
In this repository there are the following algorithms:

1. [K-Means](https://en.wikipedia.org/wiki/K-means_clustering);
2. [K-NN  Classification](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm);
3. [SVM Classification](https://en.wikipedia.org/wiki/Support_vector_machine);

There are two versions to each algorithm, one version using files as input and other using the HDFS's API;



## How to compile

Fist of all, it's necessary to import the [HDFS's integration](https://github.com/eubr-bigsea/compss-hdfs) into the project. 

Second, compile it. If you prefer use Maven, you can use the `pom.xml` file in this project.


## How to Run a Application
 
Before execute a code, remember to start the HDFS:

> $HADOOP_PATH/sbin/start-dfs.sh


After that, just use the `runcompss` as usual. The parameters to each algoritm are different, see the Readme at each algorithm's folder to know how to submit an execution.
 

