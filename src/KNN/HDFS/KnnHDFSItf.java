package KNN.HDFS;

import integratedtoolkit.types.annotations.Parameter;
import integratedtoolkit.types.annotations.task.Method;
import integration.Block;
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


    @Method(declaringClass = "KNN.HDFS.KnnHDFS")
    void readBlockFromHDFS(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Block blk,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    Samples s
    );

    //void ClassifyBlock(Samples test_blk, Block b1, Candidate partial_result, int K)
    @Method(declaringClass = "KNN.HDFS.KnnHDFS")
    void ClassifyBlock(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Samples test_blk,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Block b1,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    Candidate partial_result,
            @Parameter( direction = Parameter.Direction.IN)
                    int K
    );


    @Method(declaringClass = "KNN.HDFS.KnnHDFS")
    Candidate merge(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Candidate  result,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Candidate  partial_result,
            @Parameter (direction = Parameter.Direction.IN)
                    int K
    );

    @Method(declaringClass = "KNN.HDFS.KnnHDFS")
    double[] getKN(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    Candidate neighborhood,
            @Parameter (direction = Parameter.Direction.IN)
                    int K
    );

    @Method(declaringClass = "KNN.HDFS.KnnHDFS")
    void savePredictionToFile(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
            int[] result,
            @Parameter(type = Parameter.Type.FILE, direction = Parameter.Direction.OUT)
            String filename
    );

    /*
    @Method(declaringClass = "KNN.HDFS.KnnHDFS")
    void accumulate_error(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] correct,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    int[] correct2

    );

    @Method(declaringClass = "KNN.HDFS.KnnHDFS")
    void evaluateFrag(
            @Parameter (direction = Parameter.Direction.IN)
                    Samples part,
            @Parameter (direction = Parameter.Direction.IN)
                    double[] partial,
            @Parameter (direction = Parameter.Direction.INOUT)
                    int[] correct
    );
    */


}
