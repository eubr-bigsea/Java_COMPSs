
package kmeansHDFS;



import integratedtoolkit.types.annotations.Parameter;
import integratedtoolkit.types.annotations.Parameter.Direction;
import integratedtoolkit.types.annotations.task.Method;
import integration.Block;


public interface KMeansHDFSItf {


	@Method(declaringClass = "kmeansHDFS.KMeansHDFS")
	void read(
			@Parameter(type = Parameter.Type.OBJECT, direction = Direction.IN) Block blk,
			@Parameter(direction = Direction.IN) int numDimensions,
			@Parameter(type = Parameter.Type.OBJECT, direction = Direction.INOUT) Fragment Point
	);

	@Method(declaringClass = "kmeansHDFS.KMeansHDFS")
	void computeNewLocalClusters(
			@Parameter
					int myK,
			@Parameter
					int numDimensions,
			@Parameter(direction = Direction.IN)
					float[] points,
			@Parameter
					float[] clusterPoints,
			@Parameter(direction = Direction.INOUT)
					float[] newClusterPoints,
			@Parameter(direction = Direction.INOUT)
					int[] clusterCounts
	);

	@Method(declaringClass = "kmeansHDFS.KMeansHDFS")
	void accumulate(
			@Parameter(direction = Direction.INOUT)
					float[] onePoints,
			@Parameter
					float[] otherPoints,
			@Parameter(direction = Direction.INOUT)
					int[] oneCounts,
			@Parameter
					int[] otherCounts
	);

	@Method(declaringClass = "kmeansHDFS.KMeansHDFS")
    float[] initPointsFrag(
			@Parameter
					int oneCounts,
			@Parameter
					int otherCounts
	);

}
