package KNN.HDFS_simple;

/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood) using data of HDFS with array
 *
 * Input: labels != 0
 *
 */


import integration.Block;
import integration.HDFS;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class KnnHDFS {

    //readALLBlockFromHDFS: Read all the Train File in the HDFS
    public static void readALLBlockFromHDFS(ArrayList<Block> FILE_TRAIN_SPLITS, int [] train_labels, double[][] train_features) {
        long startTime = System.nanoTime();

        int index = 0;
        for (Block blk: FILE_TRAIN_SPLITS) {
            String[] lines = blk.getRecords('\n');

            System.out.println("Number lines: " + lines.length);
            StringTokenizer tokenizer;
            for (int l = 0; l < lines.length; l++) {
                tokenizer = new StringTokenizer(lines[l], ",");
                train_labels[index] = (int) Float.parseFloat(tokenizer.nextToken());
                for (int i = 0; tokenizer.hasMoreTokens(); i++)
                    train_features[index][i] = Double.parseDouble(tokenizer.nextToken());
                index++;
            }

            System.out.println("[INFO] - Readed " + lines.length + " records");
        }
        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double) estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - readALLBlockFromHDFS -> Time elapsed: %.2f seconds\n",seconds);
    }

    //getPopularElement: to choose the most popular element in the array
    public static int getPopularElement(int[] a){ //do better K-1

        int count = 1, tempCount;
        int popular = a[0];
        int temp;

        for (int i = 0; i < (a.length - 2); i++){
            temp = a[i];
            tempCount = 0;
            for (int j = 1; j < (a.length -1); j++){
                if (temp == a[j])
                    tempCount++;
            }
            if (tempCount > count){
                popular = temp;
                count = tempCount;
            }
        }

        return popular;
    }

    //distance: to calculate the distance between the record_a and the record_b
    public static double distance(double[] a, double[] b) {

        int sum = 0;
        for(int i = 0; i < a.length; i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }

        return  Math.sqrt(sum); // euclidian distance would be sqrt(sum)... (int)
    }

    //getKN: do the prediction of one partition
    public static int[] getKN(int[][] neighborhood,int K){
        long startTime = System.nanoTime();

        int size = neighborhood.length;
        int[] result = new int[size];
        for (int i =0;i<size;i++){
            result[i] = getPopularElement(neighborhood[i]);
        }

        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("\n[INFO] - getKN -> Time elapsed: %.2f seconds\n",seconds0);

        return result;
    }

    //savePredictionToFile: to save the predicition in a file
    public static void savePredictionToHDFS(int[] result, String defaultFS, String filename){
        HDFS dfs = new HDFS(defaultFS);

        String file1 = "";
        for (int i = 0; i < result.length; i++) {
            if (result[i] != 0)
                file1 +=result[i]+"\n";
        }
        dfs.writeFILE(file1,"hdfs://localhost:9000/user/pdm2/"+filename,false);
        System.out.println("File "+filename+" saved");
    }

    //ClassifyBlock: do the prediction
    public static int [][] ClassifyBlock(int [] train_labels,   double[][] train_features, Block b1, int K, int numDim){
        long startTime = System.nanoTime();

        String[] tokens;
        String[] lines = b1.getRecords('\n');

        int   [] test_labels    = new     int[lines.length];
        double[][]test_features  = new double[lines.length][numDim];
        int [][] semi_labels = new int[lines.length][K+1];

        for (int l = 0; l<lines.length;l++) {
            tokens = lines[l].split(",");
            test_labels[l] = (int) Float.parseFloat(tokens[0]);
            for (int i = 1; i < tokens.length; i++)
                test_features[l][i - 1] = Double.parseDouble(tokens[i]);
        }

        double  [][] semi_features    = new double[lines.length][K+1];

        System.out.println("[INFO] - Readed "+lines.length+" records");
        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("[INFO] - Readed Training file -> Time elapsed: %.2f seconds\n",seconds0);

        for (int s=0; s<test_labels.length;s++) {

            if(s% 10000 == 0)
                System.out.println("Record:" + s);

            double tmp_dist  = 0;
            int    tmp_label = 0;

            //Setting up
            for (int d=0; d<K;d++){
                semi_labels  [s][d] = 0;
                semi_features[s][d] = Double.MAX_VALUE;
            }

            for (int s2 = 0; s2<train_features.length;s2++) {
                semi_features[s][K] = distance(train_features[s2], test_features[s]);
                semi_labels  [s][K] = train_labels[s2];

                for (int j=K; j>0;j--){ //Insert Sort  - OK
                    if(semi_features[s][j] < semi_features[s][j-1]){

                        tmp_label =  semi_labels[s][j];
                        semi_labels[s][j]    = semi_labels[s][j-1];
                        semi_labels[s][j-1]  = tmp_label;

                        tmp_dist = semi_features[s][j];
                        semi_features[s][j]    = semi_features[s][j-1];
                        semi_features[s][j-1]  = tmp_dist;
                    }
                }
            }

        }

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("\n[INFO] - KNN.HDFS_simple.ClassifyBlock -> Time elapsed: %.2f seconds\n",seconds);
        return semi_labels;
    }




    public static void main(String[] args) {

        int numFrags = 0;
        int numDim = 0;
        int sizeTrain = 0;
        int K = 0;
        Boolean force_split = false;
        String trainingSet_name     =  "";
        String validationSet_name   =  "";
        String defaultFS = System.getenv("MASTER_HADOOP_URL");

        // Get and parse arguments
        int argIndex = 0;
        while (argIndex < args.length) {
            String arg = args[argIndex++];
            if (arg.equals("-K")) {
                K = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-f")) {
                force_split=true;
                numFrags  = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-t")) {
                trainingSet_name = args[argIndex++];
            }else if (arg.equals("-v")) {
                validationSet_name = args[argIndex++];
            }else if (arg.equals("-nt")) {
                sizeTrain = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-nd")) {
               numDim = Integer.parseInt(args[argIndex++]);
            }
        }
        if (trainingSet_name.equals("") || validationSet_name.equals("")){
            System.out.println("[ERROR] - You need to choose a file to train and test");
            System.exit(0);
        }

        System.out.println("Running with the following parameters:");
        System.out.println("- K Neighborhood: " + K);
        System.out.println("- Dimension: " + numDim);
        System.out.println("- Training set: " + trainingSet_name);
        System.out.println("- Test set: " + validationSet_name);
        System.out.println("- Frag Number:" + numFrags+"\n");

        //HDFS part
        HDFS dfs =  new HDFS(defaultFS);
        ArrayList<Block> FILE_TRAIN_SPLITS;
        ArrayList<Block> FILE_TEST_SPLITS;
        if (force_split) {
            FILE_TRAIN_SPLITS = dfs.findBlocksByRecords(trainingSet_name, numFrags);
            FILE_TEST_SPLITS  = dfs.findBlocksByRecords(validationSet_name, numFrags);
        }else{
            FILE_TRAIN_SPLITS = dfs.findALLBlocks(trainingSet_name);
            FILE_TEST_SPLITS = dfs.findALLBlocks(validationSet_name);
        }
        numFrags = FILE_TEST_SPLITS.size();


        int      [] train_labels   = new    int[sizeTrain];
        double [][] train_features = new double[sizeTrain][numDim];
        int[] result_part_final;

        readALLBlockFromHDFS(FILE_TRAIN_SPLITS, train_labels, train_features);
        //Read the test set and classifier
        for(int f1=0; f1<numFrags;f1++){
            int [][] semi_labels = ClassifyBlock(train_labels, train_features, FILE_TEST_SPLITS.get(f1), K,numDim);
            result_part_final = getKN(semi_labels,K);
            savePredictionToHDFS(result_part_final, defaultFS,"Result_knnHDFS_simple"+f1+".out");
        }

    }
}