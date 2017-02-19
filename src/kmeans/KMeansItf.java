/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package kmeans;


import integratedtoolkit.types.annotations.Parameter;
import integratedtoolkit.types.annotations.Parameter.Direction;
import integratedtoolkit.types.annotations.task.Method;


public interface KMeansItf {

	@Method(declaringClass = "kmeans.KMeans")
	void computeNewLocalClusters(
		@Parameter
		int myK,
		@Parameter
		int numDimensions,
		@Parameter
		float[] points,
		@Parameter
		float[] clusterPoints,
		@Parameter(direction = Direction.INOUT)
		float[] newClusterPoints,
		@Parameter(direction = Direction.INOUT)
		int[] clusterCounts
	);

	@Method(declaringClass = "kmeans.KMeans")
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

}
