package KNN.HDFS_array;

import integratedtoolkit.types.annotations.Parameter;
import integratedtoolkit.types.annotations.task.Method;
import integration.Block;

/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood) using data of HDFS
 *
 */

public interface KnnHDFSItf {

    @Method(declaringClass = "KNN.HDFS_array.KnnHDFS")
    void readBlockFromHDFS(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)    Block blk,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) int [] test_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) double[][] test_features
    );


    @Method(declaringClass = "KNN.HDFS_array.KnnHDFS")
    void ClassifyBlock(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) int [] test_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) double[][] test_features,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) Block b1,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) int [][] semi_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)  double[][] semi_features,
            @Parameter(direction = Parameter.Direction.IN) int K,
            @Parameter(direction = Parameter.Direction.IN) int numDim
    );


    @Method(declaringClass = "KNN.HDFS_array.KnnHDFS")
    void merge(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) int[][] semi_lab,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) double[][] semi_dist,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)    int[][] right_lab,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)    double[][] right_dist,
            @Parameter(direction = Parameter.Direction.IN) int K
    );


    @Method(declaringClass = "KNN.HDFS_array.KnnHDFS")
    int[] getKN(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) int[][] neighborhood,
            @Parameter(direction = Parameter.Direction.IN) int K
    );


    @Method(declaringClass = "KNN.HDFS_array.KnnHDFS")
    void savePredictionToFile(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    int[] result,
            @Parameter(type = Parameter.Type.FILE, direction = Parameter.Direction.OUT)
                    String filename
    );

    /*
    @Method(declaringClass = "KNN.HDFS_array.KnnHDFS")
    void accumulate_error(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] correct,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] correct2

    );

    @Method(declaringClass = "KNN.HDFS_array.KnnHDFS")
    void evaluateFrag(
            @Parameter (direction = Parameter.Direction.IN)
                    int[] part,
            @Parameter (direction = Parameter.Direction.IN)
                    int[] partial,
            @Parameter (direction = Parameter.Direction.INOUT)
                    int[] correct
    );
    */


}
