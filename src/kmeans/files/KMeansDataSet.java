package kmeans.files;

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

import java.io.*;


/**
 * A class to encapsulate an input set of 
 * Points for use in the kmeans.files.HDFS program and the
 * reading/writing of a set of points to data files.
 */

public class KMeansDataSet {
    //private static final int cookie = 0x2badfdc0;
    //private static final int version = 1;
    
    public final int numPoints;
    public final int numDimensions;
    public final float[][] points;  //points splitted
    public final float[] currentCluster;
    
    
    public KMeansDataSet(int np, int nd, float[][] pts, float[] cluster) {
        numPoints = np;
        numDimensions = nd;
        points = pts;
        currentCluster = cluster;
    }

    public static KMeansDataSet readPointsFromFile(String fileName,int numPoints,int numDimensions,int numFrags, int K) {
        int i_t = 0;
        int i =0;
        int j = 0;
        int f = 0;


        float[][] points = null;
        float[] cluster = new float[K * numDimensions];
        try {
            BufferedReader lines = new BufferedReader(new FileReader(fileName));
            System.out.printf("Reading %d %d-dimensional points from %s\n", numPoints, numDimensions, fileName);

            int pointsPerFragment = numPoints / numFrags;
            points = new float[numFrags][pointsPerFragment * numDimensions];

            for (i_t = 0; i_t < numPoints; i_t++,i++) {
                if ((i_t != 0) && ((i_t % pointsPerFragment) == 0)) {
                    f++;
                    i=0;
                }
                String line = lines.readLine();
                String[] tok = line.split(",");

                if(tok.length>1) {
                   // System.out.println(line);
                    for (j = 1; j <= numDimensions; ++j) { // 1 to 28
                        points[f][i * numDimensions + j - 1] = Float.parseFloat(tok[j]);
                    }
                }
            }

            // for (f=0;f<2;f++)
            //     for (i=0;i<150;i+=2)
            //        System.out.println(f +"|"+ points[f][i]+" "+points[f][i+1]);

        } catch (FileNotFoundException e) {
            System.err.println("Unable to open file " + fileName);
        } catch (IOException e) {
            System.err.printf("File did not contain enough data for %d %d-dimenstional points\n", numPoints, numDimensions);
            System.err.printf("Only found %d floats; expected to find %d\n", i * numDimensions + j, numPoints * numDimensions);
            e.printStackTrace();
        }

        return new KMeansDataSet(numPoints, numDimensions, points, cluster);

    }
}
