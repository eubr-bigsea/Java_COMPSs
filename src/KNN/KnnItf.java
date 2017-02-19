package KNN;


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


    @Method(declaringClass = "KNN.Knn")
    void classifyFrag(
            @Parameter (direction = Parameter.Direction.IN)
                    ArrayList<Sample> validationSet,
            @Parameter (direction = Parameter.Direction.IN)
                    ArrayList<Sample> trainingSet,
            @Parameter (direction = Parameter.Direction.INOUT)
                    int[] partial,
            @Parameter (direction = Parameter.Direction.IN)
                    int K
    );


    @Method(declaringClass = "KNN.Knn")
    void evaluateFrag(
            @Parameter (direction = Parameter.Direction.IN)
                    ArrayList<Sample>part,
            @Parameter (direction = Parameter.Direction.IN)
                    int[] partial,
            @Parameter (direction = Parameter.Direction.INOUT)
                    int[] correct
    );


    @Method(declaringClass = "KNN.Knn")
    void accumulate(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] numcorrect,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] numcorrect2

    );




}
