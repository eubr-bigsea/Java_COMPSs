## Kmeans

Name: Kmeans

Contact Person: lucasmsp@dcc.ufmg.br

obs: 

Platform: COMPSs


### Description




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

	runcompss kmeans.files.KMeans    -c 2 -d 28 -i 10 -f 2 -n 100000 -t train_data.csv 

	runcompss kmeans.HDFS.KMeansHDFS -c 2 -d 28 -i 3 -f 2  -n 100000 -t train_data.csv

