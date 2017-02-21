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

    @Method(declaringClass = "KNN.HDFS.KnnHDFS")
    void ClassifyBlock(
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    Samples test_blk,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.IN)
                    ArrayList<Block> FILE_TRAIN_SPLITS,
            @Parameter(type = Parameter.Type.OBJECT, direction = Parameter.Direction.INOUT)
                    Samples partial_result,
            @Parameter( direction = Parameter.Direction.IN)
                    int K
    );


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
                    Samples partial,
            @Parameter (direction = Parameter.Direction.INOUT)
                    int[] correct
    );



}
