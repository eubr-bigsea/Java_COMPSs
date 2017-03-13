## K-NN Classifier

Name: KNN

Contact Person: lucasmsp@dcc.ufmg.br

Platform: COMPSs



### Description

K Nearest Neighbors (KNN) is a non-parametric method used for classification and regression. In k-NN classification, an object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

The optimal choice of the value k is highly data-dependent: in general a larger k suppresses the effects of noise, but makes the classification boundaries less distinct.

The algorithm reads a dataset composed by labels and features (numeric fields), each record by line as the pattern: label,feature1,feature2, ..., featureN


### Versions
There are two versions of KNN, depending on the file system where the data was stored.

* Version 1: ''files'', where the files are stored in the usual file system.

* Version 2: ''HDFS'', where the files are stored in the HDFS.


### Execution instructions

Usage:

	runcompss KNN.files.KNN    -K <K_value> -f <num_frag> -t <train_datafile> -v <test_datafile> -out <output_path>

	runcompss KNN.HDFS.KNNHDFS -K <K_value> -f <num_frag> -t <train_datafile> -v <test_datafile> -out <output_path>


where:

* - K_value: integer number used into the classification;
* - num_frag:  number of nodes used on the execution (optinal in version 2);
* - train_datafile: File's path used to train the model;
* - test_datafile: File's path to classification;
* - output_path: Path to save the result.


### Execution Example

	runcompss 	KNN.files.KNN    -K 3 -f 2  -t train_data.csv -v test_data.csv -out /home/user

	runcompss 	KNN.HDFS.KnnHDFS -K 3 -t /user/hdfs/train_data.csv -v /user/hdfs/test_data.csv -out /user/hdfs



