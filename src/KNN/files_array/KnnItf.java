package KNN.files_array;


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

    @Method(declaringClass = "KNN.files_array.Knn")
    void ClassifyBlock(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)    int [] test_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)    double[][] test_features,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)    int [] train_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)    double[][] train_features,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) int [][] semi_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) double[][] semi_features,
            @Parameter(direction = Parameter.Direction.IN) int K,
            @Parameter(direction = Parameter.Direction.IN) int numDim
    );



    @Method(declaringClass = "KNN.files_array.Knn")
    void merge(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) int[][] semi_lab,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) double[][] semi_dist,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)    int[][] right_lab,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)    double[][] right_dist,
            @Parameter(direction = Parameter.Direction.IN) int K
    );


    @Method(declaringClass = "KNN.files_array.Knn")
    int[] getKN(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) int[][] neighborhood,
            @Parameter(direction = Parameter.Direction.IN) int K
    );



    @Method(declaringClass = "KNN.files_array.Knn")
    void savePredictionToFile(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    int[] result,
            @Parameter(type = Parameter.Type.FILE, direction = Parameter.Direction.OUT)
                    String filename
    );

/*
    @Method(declaringClass = "KNN.files_array.Knn")
    void evaluateFrag(
            @Parameter(direction = Parameter.Direction.IN)
                    int[] part,
            @Parameter(direction = Parameter.Direction.IN)
                    double[] partial,
            @Parameter(direction = Parameter.Direction.INOUT)
                    int[] correct
    );




    @Method(declaringClass = "KNN.files_array.Knn")
    void accumulate(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] numcorrect,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] numcorrect2

    );
*/


}
