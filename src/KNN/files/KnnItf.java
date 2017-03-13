package KNN.files;


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
import java.util.ArrayList;

public interface KnnItf {



    @Method(declaringClass = "KNN.files.Knn")
    void  classifyFrag(
            @Parameter (type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    ArrayList<Sample> trainingSet,
            @Parameter (type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    ArrayList<Sample> testSet,
            @Parameter (direction = Parameter.Direction.IN)
                    int K,
            @Parameter (type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                Candidate[] partial_result
    );




    @Method(declaringClass = "KNN.files.Knn")
    Candidate[] merge(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Candidate[]  result,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Candidate[]  partial_result,
            @Parameter (direction = Parameter.Direction.IN)
                int K
    );



    @Method(declaringClass = "KNN.files.Knn")
    double[] getKN(
            @Parameter (type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Candidate[] neighborhood,
            @Parameter (direction = Parameter.Direction.IN)
                    int K
    );


    @Method(declaringClass = "KNN.files.Knn")
    void evaluateFrag(
            @Parameter (direction = Parameter.Direction.IN)
                    ArrayList<Sample>part,
            @Parameter (direction = Parameter.Direction.IN)
                    double[] partial,
            @Parameter (direction = Parameter.Direction.INOUT)
                    int[] correct
    );




    @Method(declaringClass = "KNN.files.Knn")
    void accumulate(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] numcorrect,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] numcorrect2

    );



}
