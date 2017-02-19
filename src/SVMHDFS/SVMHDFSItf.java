package SVMHDFS;


/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  SVM using data from HDFS
 *
 */


import integratedtoolkit.types.annotations.Parameter;
import integratedtoolkit.types.annotations.task.Method;
import integration.Block;

public interface SVMHDFSItf {

    @Method(declaringClass = "SVMHDFS.SVMHDFS")
    Sample loadfileFromHDFS(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Block blk,
            @Parameter(direction = Parameter.Direction.IN)
                    int numDim
    );


    @Method(declaringClass = "SVMHDFS.SVMHDFS")
    void calc_yp(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
            Sample XY,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
            double[] w,
            @Parameter( direction = Parameter.Direction.IN)
            int numDim,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
            double[] COST,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
            Sample s
    );

    @Method(declaringClass = "SVMHDFS.SVMHDFS")
    Sample predict_chunck(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Sample XY,
            @Parameter(direction = Parameter.Direction.IN)
                    double[] w

    );

    @Method(declaringClass = "SVMHDFS.SVMHDFS")
    void accumulate_cost(
            @Parameter( direction = Parameter.Direction.INOUT)
                    double[] COST,
            @Parameter(direction = Parameter.Direction.INOUT)
                    double[] COST2
    );

    @Method(declaringClass = "SVMHDFS.SVMHDFS")
    void accumulate_grad(
            @Parameter( direction = Parameter.Direction.INOUT)
            double[] grad,
            @Parameter( direction = Parameter.Direction.INOUT)
            double[] grad2

    );

    @Method(declaringClass = "SVMHDFS.SVMHDFS")
    void  accumulate_error(
            @Parameter( direction = Parameter.Direction.INOUT) int[] ACC,
            @Parameter( direction = Parameter.Direction.INOUT) int[] ACC2
    );


    @Method(declaringClass = "SVMHDFS.SVMHDFS")
    void updateWeight(
            @Parameter(direction = Parameter.Direction.IN)
                    double lr,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)
                    double[] grad_p,
            @Parameter( direction = Parameter.Direction.INOUT)
                    double [] w
    );

    @Method(declaringClass = "SVMHDFS.SVMHDFS")
    void verify (
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) Sample labels_result,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) Sample test,
            @Parameter( direction = Parameter.Direction.INOUT) int[] ACC
    );



    @Method(declaringClass = "SVMHDFS.SVMHDFS")
    void partial_grad(
            @Parameter(direction = Parameter.Direction.IN) int numDim,
            @Parameter(direction = Parameter.Direction.IN) Sample yp,
            @Parameter(direction = Parameter.Direction.IN) Sample XY,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    double[] grad
    );


}