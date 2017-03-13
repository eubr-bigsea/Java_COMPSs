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
package kmeans.files;


import java.lang.reflect.Array;

public class KMeans {
    
    /**
     * Given the current cluster, compute a new cluster
     */
    public static void computeNewLocalClusters(int myK,	int numDimensions, float[] points,
                                               float[] clusterPoints, float[] newClusterPoints, int[] clusterCounts) {
    	int numPoints = points.length / numDimensions;
        printPoints(points);

        for (int pointNumber = 0; pointNumber < numPoints; pointNumber++) { //to each point in this block
            int closest = -1;
            float closestDist = Float.MAX_VALUE;
            for (int k = 0; k < myK; k++) {
                float dist = 0;
                for (int dim = 0; dim < numDimensions; dim++) {
                    float tmp = points[pointNumber * numDimensions + dim] - clusterPoints[k * numDimensions + dim];
                    dist += tmp * tmp;
                }
                if (dist < closestDist) {
                    closestDist = dist;
                    closest = k;
                }
            }
            
            for (int dim = 0; dim < numDimensions; dim++) {
                newClusterPoints[closest * numDimensions + dim] += points[pointNumber * numDimensions + dim];
            }
            clusterCounts[closest]++;
        }
        System.out.println(clusterCounts[0]);
        System.out.println(clusterCounts[1]);
    }

    public static void accumulate(float[] onePoints, float[] otherPoints,
    	int[] oneCounts, int[] otherCounts) {
	    for (int i = 0; i < otherPoints.length; i++) {
	    	onePoints[i] += otherPoints[i];
        }

        for (int i = 0; i < otherCounts.length; i++) {
         	oneCounts[i] += otherCounts[i];
    	}

    }


    private static void localReduction(float[] points, int[] counts, int K, int numDimensions, float[] cluster) {
    	for (int k = 0; k < K; k++) {
            float tmp = (float)counts[k];
            for (int dim = 0; dim < numDimensions; dim++) {
                if (tmp == 0.0)
                    tmp = (float) 0.0000001;
                points[k * numDimensions + dim] /= tmp;
            }
        }
	
        System.arraycopy(points, 0, cluster, 0, cluster.length);
    }

    public static boolean compareArrays(float[] array1, float[] array2) {
        boolean b = true;
        if (array1 != null && array2 != null){
            if (array1.length != array2.length)
                b = false;
            else
                for (int i = 0; i < array2.length; i++) {
                    if (array2[i] != array1[i]) {
                        b = false;
                    }
                }
        }else{
            b = false;
        }
        return b;
    }
    
    private static void initializePoints(KMeansDataSet data,int numFrag) {
        // Initialize cluster (copy first points)	
        int nFrag = 0, startPos = 0;
        int toCopy = data.currentCluster.length;


        //int partToCopy = data.currentCluster.length / numFrag;
        while (toCopy > 0) {
        	int copied = copyToCluster(data.points[nFrag], data.currentCluster, toCopy, startPos);
        	toCopy -= copied;
        	startPos += copied;
        	nFrag++;

        }
    }

    private static int copyToCluster(float[] points, float[] cluster, int toCopy, int startPos) {
    	int canCopy = Math.min(toCopy, Array.getLength(points));
        int j = 0;
    	for (int i = startPos; i < startPos + canCopy; i++) {
    		cluster[i] = points[j++];
    	}
    	return j;
    }

    @SuppressWarnings("unused")
	private static void printPoints(float[] points) {
    	//System.out.println("No print");
    	for (int i = 0; i < points.length; i+=2)
			System.out.println(points[i] + " "+points[i+1]);
    	System.out.println("");
    }

    @SuppressWarnings("unused")
	public static void main(String[] args) {
    	// Default values
        int K = 0;
        int iterations = 0;
        int nPoints = 0;
        int nDimensions = 0;
        int nFrags = 0;
		String fileName = "";
        int argIndex = 0;

        // Get and parse arguments
        while (argIndex < args.length) {
            String arg = args[argIndex++];
            if (arg.equals("-c")) {
                K = Integer.parseInt(args[argIndex++]);   
            } else if (arg.equals("-i")) {
                iterations = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-n")) {
            	nPoints = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-d")) {
            	nDimensions = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-f")) {
            	nFrags = Integer.parseInt(args[argIndex++]);
            } else {
            	// WARN: Disabled
            	fileName = arg;
            }
        }

        System.out.println("Running with the following parameters:");
        System.out.println("- Clusters: " + K);
        System.out.println("- Iterations: " + iterations);
        System.out.println("- Points: " + nPoints);
        System.out.println("- Dimensions: " + nDimensions);
        System.out.println("- Nodes: " + nFrags);

        // Uncomment this line to load data from a file
       	KMeansDataSet data = KMeansDataSet.readPointsFromFile(fileName, nPoints, nDimensions, nFrags, K);

        initializePoints(data,nFrags);

        // Do the requested number of iterations
        for (int iter = 0; iter < iterations; iter++) {

            int[][] clusterCounts = new   int[nFrags][K];
            float[][] newClusters = new float[nFrags][K*nDimensions];
            float[] bkp_clusters = data.currentCluster;

        	// Computation
        	for (int f = 0; f < nFrags; f++) {
        		float[] points_frag = data.points[f];
        		computeNewLocalClusters(K, nDimensions, points_frag, data.currentCluster, newClusters[f], clusterCounts[f]);
            }
        	
        	// Reduction: points and counts
        	// Stored in newClusters[0], clusterCounts[0]
            int size = newClusters.length;
        	int i = 0;
        	int gap = 1;
        	while (size > 1) {
        		accumulate(newClusters[i], newClusters[i + gap], clusterCounts[i], clusterCounts[i + gap]);
        		size--;
        		i = i + 2 * gap;
        		if (i == newClusters.length) {
        			gap *= 2;
        			i = 0;
        		}
        	}


            // Local reduction to get the new clusters
            // Adjust cluster coordinates by dividing each point value
            // by the number of points in the cluster
        	localReduction(newClusters[0], clusterCounts[0], K, nDimensions, data.currentCluster);

            if(compareArrays(bkp_clusters,data.currentCluster) && iter!=0){
                System.out.println("Iterations:"+ iter);
                break;
            }


        }
        
        // All done. Print the results
        System.out.println("Result clusters: ");
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < nDimensions; j++) {
                if (j > 0) {
                	System.out.print(" ");
                }
                System.out.print(data.currentCluster[k*nDimensions + j]);
            }
            System.out.println();
        }
        System.out.println();



    }
	
}
