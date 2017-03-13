package KNN.HDFS_simple;

import integratedtoolkit.types.annotations.Parameter;
import integratedtoolkit.types.annotations.task.Method;
import integration.Block;
import integration.HDFS;

import java.util.ArrayList;

/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood) using data of HDFS
 *
 */

public interface KnnHDFSItf {

    @Method(declaringClass = "KNN.HDFS_simple.KnnHDFS")
    void readALLBlockFromHDFS(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) ArrayList<Block> FILE_TRAIN_SPLITS,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) int [] train_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) double[][] train_features
    );

    @Method(declaringClass = "KNN.HDFS_simple.KnnHDFS")
    int[][] ClassifyBlock(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) int[] train_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) double[][] train_features,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) Block b1,
            @Parameter(direction = Parameter.Direction.IN) int K,
            @Parameter(direction = Parameter.Direction.IN) int numDim
    );

    @Method(declaringClass = "KNN.HDFS_simple.KnnHDFS")
    int[] getKN(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) int[][] neighborhood,
            @Parameter(direction = Parameter.Direction.IN) int K
    );


    @Method(declaringClass = "KNN.HDFS_simple.KnnHDFS")
    void savePredictionToHDFS(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)  int[] result,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)  String defaultFS,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)  String filename
    );

}
