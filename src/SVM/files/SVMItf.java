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


    @Method(declaringClass = "SVM.files.SVM") //, isModifier=true, priority=true)
    double[] calc_yp(
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)      double[][] X,
            @Parameter(direction = Parameter.Direction.IN)                                     double[] w,
            @Parameter(direction = Parameter.Direction.IN)                                     int numDim
    );

    @Method(declaringClass = "SVM.files.SVM")
    double[] calc_CostAndGrad(
            @Parameter(direction = Parameter.Direction.IN)                                  int numDim,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)   double[] yp,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)   double[][] train_features,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)   int[] train_labels,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT) double[] COST,
            @Parameter(direction = Parameter.Direction.IN)                                  int f,
            @Parameter(direction = Parameter.Direction.IN)                                  double lambda,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)   double[] w
    );


    @Method(declaringClass = "SVM.files.SVM")
    void  accumulateCostAndGrad(
            @Parameter( direction = Parameter.Direction.INOUT) double[] grad,
            @Parameter( direction = Parameter.Direction.IN)    double[] grad2,
            @Parameter( direction = Parameter.Direction.INOUT) double[] COST,
            @Parameter( direction = Parameter.Direction.IN)    double[] COST2
    );


    @Method(declaringClass = "SVM.files.SVM")
    void updateWeight(
            @Parameter(direction = Parameter.Direction.IN)                                  double lr,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)   double[] grad_p,
            @Parameter( direction = Parameter.Direction.INOUT)                              double [] w
    );


    @Method(declaringClass = "SVM.files.SVM")
    int[] predict_chunck(
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)  double[][] testX,
            @Parameter(type = Parameter.Type.OBJECT,  direction = Parameter.Direction.IN)  double[] w
    );

    @Method(declaringClass = "SVM.files.SVM")
    void savePredictionToFile(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)  int[] result,
            @Parameter(type = Parameter.Type.FILE, direction = Parameter.Direction.OUT)   String filename
    );



}
