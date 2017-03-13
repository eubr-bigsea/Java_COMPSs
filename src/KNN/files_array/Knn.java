package KNN.files_array;

/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood)
 *
 */

import java.io.*;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class Knn {



    public static void loadfile_and_split(double[][][] features, int[][] labels,
                                          String fileName, int sizeTrainPerFrag,  int numTotal) {

        long startTime = System.nanoTime();

        try {
            BufferedReader lines = new BufferedReader(new FileReader(fileName));
            StringTokenizer tokenizer;

            int index=0;
            int f = -1;
            for (int i=0; i<numTotal;i++){
                String line = lines.readLine();

                if ((index % sizeTrainPerFrag) == 0){
                    f++;
                    index=0;
                }

                tokenizer = new StringTokenizer(line,",");
                labels[f][index] = (int) Float.parseFloat(tokenizer.nextToken());
                int index2 = 0;
                while (tokenizer.hasMoreTokens()) {
                        features[f][index][index2] = Double.parseDouble(tokenizer.nextToken());
                        index2++;
                }

                index++;
            }

            long estimatedTime = System.nanoTime() - startTime;
            double seconds = (double)estimatedTime / 1000000000.0;
            System.out.printf("[INFO] loadfile_and_split->Time elapsed: %.2f seconds\n",seconds);

        } catch (FileNotFoundException e) {
            System.out.println("ERROR - KNN.loadfile_and_split");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("ERROR - KNN.loadfile_and_split");
            e.printStackTrace();
        }
    }

    public static double distance(double[] a, double[] b) {
        int sum = 0;
        for(int i = 0; i < a.length; i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return Math.sqrt(sum); // euclidian distance would be sqrt(sum)...
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

    public static int getPopularElement(int[] a){


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


    public static void ClassifyBlock(int [] test_labels,   double[][] test_features,
                                     int [] train_labels,  double[][] train_features,
                                     int [][] semi_labels, double[][] semi_features,
                                     int K, int numDim){

        long startTime = System.nanoTime();

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

    public static void merge(int[][] semi_lab,  double[][] semi_dist,
                             int[][] right_lab, double[][] right_dist, int K){
        long startTime = System.nanoTime();

        for (int i_record=0;i_record<semi_dist.length;i_record++){
            int i_left =0;
            int i_right=0;

            double[] left_dist = semi_dist[i_record].clone();
            int   [] left_lab  = semi_lab [i_record].clone();

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
        }

        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("\n[INFO] - merge -> Time elapsed: %.2f seconds\n",seconds0);
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


    @SuppressWarnings("unused")
    public static void evaluateFrag(int[] part, double[] partial, int[] correct){
        long startTime = System.nanoTime();

        for (int i = 0; i<part.length;i++) {
            if (partial[i] == part[i])
                correct[0]++;
        }

        System.out.println("Correct:"+ correct[0]);

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double) estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - evaluateFrag -> Time elapsed: %.2f seconds\n\n",seconds);

    }

    @SuppressWarnings("unused")
    public static void accumulate(int[] numcorrect, int[] numcorrect2){

        for(int f=0; f<numcorrect2.length;f++)
            numcorrect[f] += numcorrect2[f];

    }

    public static void main(String[] args) throws IOException {

        int numFrag = 2;
        int K = 1;
        String trainingName = "";
        String testName     = "";
        int sizeTrain = 100000;
        int sizeTest = 100000;
        int numDim = 28;

        // Get and parse arguments
        int argIndex = 0;
        while (argIndex < args.length) {
            String arg = args[argIndex++];
            if (arg.equals("-K")) {
                K = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-f")) {
                numFrag = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-t")) {
                trainingName = args[argIndex++];
            }else if (arg.equals("-v")) {
                testName = args[argIndex++];
            }else if (arg.equals("-nd")) {
                numDim = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-nt")) {
                sizeTrain  = Integer.parseInt(args[argIndex++]);
            }else if(arg.equals("-nv")) {
                sizeTest = Integer.parseInt(args[argIndex++]);
            }
        }
        if (trainingName.equals("") || testName.equals("")){
            System.out.println("[ERROR] - You need to choose a file to train and test");
            System.exit(0);
        }

        /*
                Running

         */

        System.out.println("Running Knn.files with the following parameters:");
        System.out.println("- K Neighborhood: " + K);
        System.out.println("- Nodes: " + numFrag);
        System.out.println("- Training set: " + trainingName);
        System.out.println("- Test set: " + testName+"\n");

        int         sizeTrainPerFrag = (int) Math.ceil((float)sizeTrain/numFrag);
        int      [][] train_labels   = new int[numFrag][sizeTrainPerFrag];
        double [][][] train_features = new double[numFrag][sizeTrainPerFrag][numDim];

        int         sizeTestPerFrag = (int) Math.ceil((float)sizeTrain/numFrag);
        int      [][] test_labels   = new int[numFrag][sizeTestPerFrag];
        double [][][] test_features = new double[numFrag][sizeTestPerFrag][numDim];

        loadfile_and_split(train_features,train_labels,trainingName, sizeTrainPerFrag, sizeTrain);  //Only in Master
        loadfile_and_split(test_features ,test_labels ,testName    , sizeTestPerFrag , sizeTest);   //Only in Master

        int      [][][][] semi_labels   = new    int[numFrag][numFrag][sizeTestPerFrag][K+1];
        double   [][][][] semi_dist     = new double[numFrag][numFrag][sizeTestPerFrag][K+1];
        int[] result_part_final;

        System.out.println("[INFO] - Classifier");

        //Split the TestSet and classifying it
        for (int f1 = 0 ; f1<numFrag;f1++) { //each test
            for(int f2= 0;f2<numFrag;f2++) { //each train
                ClassifyBlock(test_labels[f1], test_features[f1],
                             train_labels[f2],  train_features[f2],
                             semi_labels[f1][f2], semi_dist[f1][f2],
                             K, numDim);
            }

            //Merge Results
            int size = numFrag;
            int i = 0;
            int gap = 1;
            while (size > 1) {
                merge(semi_labels[f1][i], semi_dist[f1][i], semi_labels[f1][i + gap], semi_dist[f1][i+gap],K);
                size--;
                i = i + 2 * gap;
                if (i == numFrag) {
                    gap *= 2;
                    i = 0;
                }
            }

            result_part_final = getKN(semi_labels[f1][0],K);
            savePredictionToFile(result_part_final,"knnCOMPSs_array"+f1+".out");


        }

    }
}