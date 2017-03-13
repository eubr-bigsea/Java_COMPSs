package KNN.files_simple;


/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood)
 *
 */


import integratedtoolkit.types.annotations.Parameter;
import integratedtoolkit.types.annotations.task.Method;


public interface KnnItf {

    @Method(declaringClass = "KNN.files_simple.Knn")
    int[][] ClassifyBlock(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) double[][] test_features,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) int[] train_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) double[][] train_features,
            @Parameter(direction = Parameter.Direction.IN) int K,
            @Parameter(direction = Parameter.Direction.IN) int sizeTestPerFrag
    );


    @Method(declaringClass = "KNN.files_simple.Knn")
    int[] getKN(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) int[][] neighborhood,
            @Parameter(direction = Parameter.Direction.IN) int K
    );



    @Method(declaringClass = "KNN.files_simple.Knn")
    void savePredictionToFile(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    int[] result,
            @Parameter(type = Parameter.Type.FILE, direction = Parameter.Direction.OUT)
                    String filename
    );

}
