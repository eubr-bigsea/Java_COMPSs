package KNN.HDFS_array;

/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood) using data of HDFS with array
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


    public static void readBlockFromHDFS(Block blk, int [] test_labels, double[][] test_features) {
        long startTime = System.nanoTime();

        String[] lines = blk.getRecords('\n');

        System.out.println("Number lines: " + lines.length);
        StringTokenizer tokenizer;
        for (int l = 0; l<lines.length;l++) {
           // System.out.println("L"+l+": "+lines[l]);
            tokenizer = new StringTokenizer(lines[l],",");
            test_labels[l] = (int) Float.parseFloat(tokenizer.nextToken());
            for (int i = 0; tokenizer.hasMoreTokens(); i++)
                test_features[l][i] = Double.parseDouble(tokenizer.nextToken());
        }

        //Tratar erro de mais

        System.out.println("[INFO] - Readed "+lines.length+" records");

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double) estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - readBlockFromHDFS -> Time elapsed: %.2f seconds\n",seconds);
    }


    public static void ClassifyBlock(int [] test_labels,   double[][] test_features, Block b1,
                                     int [][] semi_labels, double[][] semi_features, int K, int numDim){
        long startTime = System.nanoTime();

        String[] tokens;
        String[] lines = b1.getRecords('\n');

        int   [] train_labels    = new    int[lines.length];
        double[][]train_features  = new double[lines.length][numDim];

        for (int l = 0; l<lines.length;l++) {
            tokens = lines[l].split(",");
            train_labels[l] = (int) Float.parseFloat(tokens[0]);
            for (int i = 1; i < tokens.length; i++)
                train_features[l][i - 1] = Double.parseDouble(tokens[i]);
        }

        System.out.println("[INFO] - Readed "+train_features.length+" records");
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
                semi_labels  [s][d] = -1;
                semi_features[s][d] = Double.MAX_VALUE;
            }

            for (int s2 = 0; s2<train_features.length;s2++) { //last position
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
        System.out.printf("\n[INFO] - ClassifierFromHDFSblock -> Time elapsed: %.2f seconds\n",seconds);
    }

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

    public static double distance(double[] a, double[] b) {

        int sum = 0;
        for(int i = 0; i < a.length; i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }

        return  Math.sqrt(sum); // euclidian distance would be sqrt(sum)... (int)
    }

    @SuppressWarnings("unused")
    public static void evaluateFrag(int[] part, int[] partial, int[] correct){
        long startTime = System.nanoTime();

        for (int i = 0; i<part.length;i++) {
           // if(i%5000 == 0)
            //    System.out.println(partial[i] +" " + part.getLabel(i));
            if (partial[i] == part[i])
                correct[0]++;
        }
        correct[1] = part.length;

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double) estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - evaluateFrag -> Time elapsed: %.2f seconds\n\n",seconds);
    }

    @SuppressWarnings("unused")
    public static void accumulate_error(int[] correct, int[] correct2){
        for(int f=0; f<correct2.length;f++)
            correct[f] += correct2[f];
    }


    public static void merge(int[][] semi_lab,  double[][] semi_dist,
                             int[][] right_lab, double[][] right_dist, int K){
        long startTime = System.nanoTime();

        for (int i_record=0;i_record<semi_dist.length;i_record++){
            int i_left =0;
            int i_right=0;

            double[] left_dist = semi_dist[i_record].clone();
            int   [] left_lab  = semi_lab [i_record].clone();


//            System.out.print("L ");
//            for(int d=0; d<left_dist.length;d++)
//                System.out.print(left_dist[d]+" ");
//            System.out.println(" ");
//
//            System.out.print("R ");
//            for(int d=0; d<right_dist[i_record].length;d++)
//                System.out.print(right_dist[i_record][d]+" ");
//            System.out.println(" ");

            for (int s=0;s<K;s++) {
                if (left_dist[i_left] > right_dist[i_record][i_right]) {
                    semi_dist[i_record][s] = right_dist[i_record][i_right];
                    semi_lab [i_record][s] = right_lab [i_record][i_right];
                    i_right++;
                } else {
                    semi_dist[i_record][s] = left_dist[i_left];
                    semi_lab [i_record][s] = left_lab [i_left];
                    i_left++;
                }
            }

//            System.out.print("M ");
//            for(int d=0; d<semi_lab[i_record].length;d++)
//                System.out.print(semi_dist[i_record][d]+" ");
//            System.out.println(" ");
        }

        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("\n[INFO] - merge -> Time elapsed: %.2f seconds\n",seconds0);
    }

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


    public static void savePredictionToFile(int[] result, String filename){
        BufferedWriter outputWriter = null;
        try {
            outputWriter = new BufferedWriter(new FileWriter(filename));

            for (int i = 0; i < result.length; i++) {
                outputWriter.write(result[i]+"\n");
                // Or: outputWriter.write(Integer.toString(result[i]);
            }
            outputWriter.flush();
            outputWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }


    }



    public static void main(String[] args) {

        int numFrags = 2;
        int numDim = 28;
        int sizeTest  = 100000;
        int K = 1;
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
                numFrags  = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-t")) {
                trainingSet_name = args[argIndex++];
            }else if (arg.equals("-v")) {
                validationSet_name = args[argIndex++];
            }else if (arg.equals("-nv")) {
                sizeTest = Integer.parseInt(args[argIndex++]);
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
        ArrayList<Block> FILE_TRAIN_SPLITS = dfs.findBlocksByRecords(trainingSet_name,numFrags);
        ArrayList<Block> FILE_TEST_SPLITS  = dfs.findBlocksByRecords(validationSet_name,numFrags);

        int sizeTestPerFrag = (int) Math.ceil((float)sizeTest/numFrags);
        int      [][] test_labels   = new    int[numFrags][sizeTestPerFrag+1];
        double [][][] test_features = new double[numFrags][sizeTestPerFrag+1][numDim];

        int      [][][][] semi_labels   = new    int[numFrags][numFrags][sizeTestPerFrag+1][K+1];
        double   [][][][] semi_dist     = new double[numFrags][numFrags][sizeTestPerFrag+1][K+1];
        int[] result_part_final;

        //Read the test set and classifier
        for(int f1=0; f1<numFrags;f1++){
            readBlockFromHDFS(FILE_TEST_SPLITS.get(f1), test_labels[f1],test_features[f1]);
            for(int f2=0; f2<numFrags;f2++) {
                ClassifyBlock(test_labels[f1], test_features[f1], FILE_TRAIN_SPLITS.get(f2), semi_labels[f1][f2], semi_dist[f1][f2], K, numDim);
            }

            //Merge Results
            int size = numFrags;
            int i = 0;
            int gap = 1;
            while (size > 1) {
                merge(semi_labels[f1][i], semi_dist[f1][i], semi_labels[f1][i + gap], semi_dist[f1][i+gap],K);
                size--;
                i = i + 2 * gap;
                if (i == numFrags) {
                    gap *= 2;
                    i = 0;
                }
            }

            result_part_final = getKN(semi_labels[f1][0],K);
            savePredictionToFile(result_part_final,"knnHDFS_array"+f1+".out");

        }
    }
}