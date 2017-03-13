
package kmeans.HDFS;



import integratedtoolkit.types.annotations.Parameter;
import integratedtoolkit.types.annotations.Parameter.Direction;
import integratedtoolkit.types.annotations.task.Method;
import integration.Block;

import java.util.ArrayList;
import java.util.List;


public interface KMeansHDFSItf {


	@Method(declaringClass = "kmeans.HDFS.KMeansHDFS")
	Fragment read(
			@Parameter(type = Parameter.Type.OBJECT, direction = Direction.IN) Block blk,
			@Parameter(direction = Direction.IN) int numDimensions
	);





	@Method(declaringClass = "kmeans.HDFS.KMeansHDFS")
	void computeNewLocalClusters(
			@Parameter(direction = Direction.IN) int myK,
			@Parameter(direction = Direction.IN) int numDimensions,
			@Parameter(direction = Direction.IN) float[] points,
			@Parameter(direction = Direction.IN)					float[] clusterPoints,
			@Parameter(direction = Direction.INOUT)					float[] newClusterPoints,
			@Parameter(direction = Direction.INOUT)					int[] clusterCounts
	);

	@Method(declaringClass = "kmeans.HDFS.KMeansHDFS")
	void accumulate(
			@Parameter(direction = Direction.INOUT)					float[] onePoints,
			@Parameter					float[] otherPoints,
			@Parameter(direction = Direction.INOUT)					int[] oneCounts,
			@Parameter					int[] otherCounts
	);



}
