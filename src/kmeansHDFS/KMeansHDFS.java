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
package kmeansHDFS;

import integration.Block;
import integration.HDFS;

import java.lang.reflect.Array;
import java.util.ArrayList;


public class KMeansHDFS {


    public static void read(Block blk,int numDimensions,Fragment Point){
        long startTime = System.nanoTime();

        String[] records = blk.getRecords();
        float[] p = new float[numDimensions * records.length];
        for (int i = 0; i<records.length;i++){
            String[] col = records[i].split(",");
            if(col.length>1) {
                for (int j = 1; j < numDimensions; ++j) { // 1 to 28
                    p[i * numDimensions + j - 1] = Float.parseFloat(col[j]);
                }
            }
        }
        Point.setFeatures_part(p);

        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("[INFO] - Readed block file -> Time elapsed: %.2f seconds\n",seconds0);

    }


    /**
     * Given the current cluster, compute a new cluster
     */
    public static void computeNewLocalClusters(int myK,	int numDimensions, float[] points,
                                               float[] clusterPoints, float[] newClusterPoints, int[] clusterCounts) {

        System.out.println("Start computeNewLocalClusters");
        int numPoints = points.length / numDimensions;
        for (int pointNumber = 0; pointNumber < numPoints; pointNumber++) {
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
    }

    public static void accumulate(float[] onePoints, float[] otherPoints, int[] oneCounts, int[] otherCounts) {
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
                points[k * numDimensions + dim] /= tmp;
            }
        }
	
        System.arraycopy(points, 0, cluster, 0, cluster.length);
    }
    
    private static void initializePoints(Fragment[] data, float[] currentCluster,int numFrags) { //chunck and cluster

        // Initialize cluster (copy first points)
        int nFrag = 0, startPos = 0;
        int toCopy = currentCluster.length;
        while (toCopy > 0) {
        	int copied = copyToCluster(data[nFrag].getFeatures_part(), currentCluster, toCopy, startPos);
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
    	System.out.println("No print");
    	//for (int i = 0; i < points.length; i++)
			//System.out.print(points[i] + " ");
    	System.out.println("");
    }


	public static void main(String[] args) {

    	// Default values
        int K = 4;
        int iterations = 100;
        int nPoints = 1000;
        int nDimensions = 28;
        int nFrags = 2;
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

        /*
            Reading part

         */

        String defaultFS = System.getenv("MASTER_HADOOP_URL");
        HDFS dfs =  new HDFS(defaultFS);
        ArrayList<Block> HDFS_SPLITS_LIST = dfs.findALLBlocks(fileName);
        nFrags = HDFS_SPLITS_LIST.size();

        System.out.println("Running with the following parameters:");
        System.out.println("- Filename: "+ fileName);
        System.out.println("- Clusters: " + K);
        System.out.println("- Iterations: " + iterations);
        System.out.println("- Points: " + nPoints);
        System.out.println("- Dimensions: " + nDimensions);
        System.out.println("- Frag: " + nFrags);


        Fragment[] Points = new Fragment[nFrags];
        for (int j = 0; j < nFrags; j++)
            Points[j] = new Fragment();

        int i = 0;
        for(Block b : HDFS_SPLITS_LIST) {
            read(b,nDimensions,Points[i]);
            i++;
        }

        float[] cluster = new float[K * nDimensions];
        int[][] clusterCounts = new int[nFrags][K];
        float[][] newClusters = new float[nFrags][K*nDimensions];

        initializePoints(Points, cluster, nFrags);

        // Do the requested number of iterations
        for (int iter = 0; iter < iterations; iter++) {
            // Computation
        	for (int j = 0; j < nFrags; j++) {
        		float[] frag = Points[j].getFeatures_part();
                computeNewLocalClusters(K, nDimensions, frag, cluster, newClusters[j], clusterCounts[j]);
            }
        	
        	// Reduction: points and counts
        	// Stored in newClusters[0], clusterCounts[0]
        	int size = newClusters.length;
        	int g = 0;
        	int gap = 1;
        	while (size > 1) {
        		accumulate(newClusters[g], newClusters[g + gap], clusterCounts[g], clusterCounts[g + gap]);
        		size--;
        		g = g + 2 * gap;
        		if (g == newClusters.length) {
        			gap *= 2;
        			g = 0;
        		}
        	}
            
            // Local reduction to get the new clusters
            // Adjust cluster coordinates by dividing each point value
            // by the number of points in the cluster
        	localReduction(newClusters[0], clusterCounts[0], K, nDimensions, cluster);
        }
        
        // All done. Print the results
        System.out.println("Result clusters: ");
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < nDimensions; j++) {
                if (j > 0) {
                	System.out.print(" ");
                }
                System.out.print(cluster[k*nDimensions + j]);
            }
            System.out.println();
        }
        System.out.println();
    }

	
}
