# Java COMPSs Algorithms

Some algorithms developed in Java COMPSs with or without using the [HDFS's API](https://github.com/eubr-bigsea/compss-hdfs) developed in this project.


-----
In this repository there are the following algorithms:

1. [K-Means](https://en.wikipedia.org/wiki/K-means_clustering);
2. [K-NN  Classification](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm);
3. [SVM Classification](https://en.wikipedia.org/wiki/Support_vector_machine);

There are two versions to each algorithm, one version using files as input and other using the HDFS's API;



## How to compile

Fist of all, it's necessary to import the HDFS's integration into the project. 



In order to compile your code using Maven, you can use the `pom.xml`file in this project:

1. Download the [HDFS_Integration.jar](https://github.com/eubr-bigsea/compss-hdfs);
2. Create a Maven local repository using the command:

	mvn install:install-file -Dfile=HDFS_Integration.jar -DgroupId=HDFS_Integration -	DartifactId=HDFS_Integration -Dversion=1.0 -Dpackaging=jar

3. Compile it using the [pom.xml](https://github.com/eubr-bigsea/Java_COMPSs/blob/master/pom.xml).




## How to Run a Application
 
Before execute a code, remember to start the HDFS:

> $HADOOP_PATH/sbin/start-dfs.sh


After that, just use the `runcompss` as usual. The parameters to each algoritm are different, see the Readme at each algorithm's folder to know how to submit an execution.
 

