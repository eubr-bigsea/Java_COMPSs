package SVM.files;


/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  SVM
 *
 */




import integratedtoolkit.types.annotations.Parameter;
import integratedtoolkit.types.annotations.task.Method;

public interface SVMItf {


    //calc_yp(double[] ypp, double COST[], double[][] X, int[] label, double[] w, int numDim )
    @Method(declaringClass = "SVM.files.SVM") //, isModifier=true, priority=true)
    void calc_yp(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    double[] ypp,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    double COST[],
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)
                    double[][] X,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    int[] label,
            @Parameter(direction = Parameter.Direction.IN)
                    double[] w,
            @Parameter(direction = Parameter.Direction.IN)
                    int numDim
    );


    //partial_grad(double[] grad_p,int numDim, double[] yp, double[][] train_features, int[] train_labels)
    @Method(declaringClass = "SVM.files.SVM")
    void  partial_grad(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                double[] grad_p,
            @Parameter(direction = Parameter.Direction.IN)
                int numDim,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)
                double[] yp,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)
                double[][] train_features,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)
                int[] train_labels
    );

    @Method(declaringClass = "SVM.files.SVM")
    void accumulate_cost(
            @Parameter( direction = Parameter.Direction.INOUT)
                    double[] COST,
            @Parameter(direction = Parameter.Direction.INOUT)
                    double[] COST2
    );

    @Method(declaringClass = "SVM.files.SVM")
    void accumulate_grad(
            @Parameter( direction = Parameter.Direction.INOUT)
                    double[] grad,
            @Parameter( direction = Parameter.Direction.INOUT)
                    double[] grad2

    );

    @Method(declaringClass = "SVM.files.SVM")
    void updateWeight(
            @Parameter(direction = Parameter.Direction.IN)
            double lr,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)
            double[] grad_p,
            @Parameter( direction = Parameter.Direction.INOUT)
            double [] w
    );


    @Method(declaringClass = "SVM.files.SVM")
    void  accumulate_error(
            @Parameter( direction = Parameter.Direction.INOUT) int[] ACC,
            @Parameter( direction = Parameter.Direction.INOUT) int[] ACC2
    );



    @Method(declaringClass = "SVM.files.SVM")
    int[] predict_chunck(
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN) int[] testY,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)  double[][] testX,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN) double[] w
    );


    @Method(declaringClass = "SVM.files.SVM")
    void verify (
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) int[] labels_result,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN) int[] test,
            @Parameter( direction = Parameter.Direction.INOUT) int[] ACC
    );






}
