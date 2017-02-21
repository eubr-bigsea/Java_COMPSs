package KNN.HDFS;

/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood) using data of HDFS
 *
 */

import java.util.ArrayList;
import integration.Block;
import integration.HDFS;

public class KnnHDFS {


    public static void readBlockFromHDFS(Block blk,Samples s) {
        long startTime = System.nanoTime();

        String[] lines = blk.getRecords();
        for (int l = 0; l<lines.length;l++) {
            String[] tokens = lines[l].split(",");
            if (tokens.length > 1) {
                s.addLabels((int) Float.parseFloat(tokens[0]));
                double[] col = new double[tokens.length - 1];
                for (int i = 1; i < tokens.length; i++)
                    col[i - 1] = Double.parseDouble(tokens[i]);
                s.addFeature(col);
            }
        }

        System.out.println("[INFO] - Readed "+s.getSizebyLabels()+" records");

        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("[INFO] - readBlockFromHDFS -> Time elapsed: %.2f seconds\n",seconds0);
    }


    public static void ClassifyBlock(Samples test_blk, ArrayList<Block> FILE_TRAIN_SPLITS, Samples partial_result, int K){
        long startTime = System.nanoTime();

        Samples trainingSet = new Samples();
        String[] tokens;

        try{
            for(Block b1 : FILE_TRAIN_SPLITS) {
                String [] lines = b1.getRecords();
                for (int  l = 0; l<lines.length;l++) {
                    tokens = lines[l].split(",");
                    if (tokens.length>1) {
                        trainingSet.addLabels( (int) Float.parseFloat(tokens[0]));
                        double[] col = new double[tokens.length - 1];
                        for (int i = 1; i < tokens.length; i++)
                            col[i - 1] = Double.parseDouble(tokens[i]);
                        trainingSet.addFeature(col);
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("Error: ClassifierFromHDFSblock - Part1");
            e.printStackTrace();
        }

        System.out.println("[INFO] - Readed "+trainingSet.getSizebyFeatures()+" records");
        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("[INFO] - Readed Training file -> Time elapsed: %.2f seconds\n",seconds0);

        for (int s =0; s<test_blk.getSizebyFeatures();s++) {
            if(s% 10000 == 0)
                System.out.println("Record:" + s);
            double[] cols = test_blk.getFeatureAll(s);
            int label = classify(trainingSet,cols, K);
            partial_result.addLabels(label);
        }

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("\n[INFO] - ClassifierFromHDFSblock -> Time elapsed: %.2f seconds\n",seconds);
    }


    public static int  classify(Samples trainingSet, double[] cols, int K) {

        int label = -1;
        double [][] bestPoints = new double[2][K+1];

        try {
            double tmp_dist = 0;
            double tmp_label = 0;
            //Setting up
            for (int d=0; d<K;d++){
                bestPoints[0][d] = -1.0;
                bestPoints[1][d] = Double.MAX_VALUE;
            }

            for (int s = 0; s<trainingSet.getSizebyFeatures();s++) {
                bestPoints[1][K] = distance(trainingSet.getFeatureAll(s), cols);
                bestPoints[0][K] = trainingSet.getLabel(s);

                for (int j=K; j>0;j--){ //Insert Sort  - OK
                    if(bestPoints[1][j] < bestPoints[1][j-1]){
                        tmp_label = bestPoints[0][j];
                        bestPoints[0][j]    = bestPoints[0][j-1];
                        bestPoints[0][j-1]  = tmp_label;

                        tmp_dist = bestPoints[1][j];
                        bestPoints[1][j]    = bestPoints[1][j-1];
                        bestPoints[1][j-1]  = tmp_dist;
                    }
                }

            }

            label = (int) getPopularElement(bestPoints[0]);

        }catch (Exception e) {
            System.out.println("[ERROR] - classify");
            e.printStackTrace();
        }

        return label;
    }


    public static double getPopularElement(double[] a){ //do better K-1

        int count = 1, tempCount;
        double popular = a[0];
        double temp;
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


    public static void evaluateFrag(Samples part, Samples partial, int[] correct){
        long startTime = System.nanoTime();

        for (int i = 0; i<part.getSizebyLabels();i++) {
            if(i%5000 == 0)
                System.out.println(partial.getLabel(i) +" " + part.getLabel(i));
            if (partial.getLabel(i) == part.getLabel(i))
                correct[0]++;
        }
        correct[1] = part.getSizebyLabels();

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double) estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - evaluateFrag -> Time elapsed: %.2f seconds\n\n",seconds);
    }

    public static void accumulate_error(int[] correct, int[] correct2){

        for(int f=0; f<correct2.length;f++)
            correct[f] += correct2[f];

    }


    public static void main(String[] args) {

        int frag = 2;
        int K = 1;
        String trainingSet_name     =  "";
        String validationSet_name   = "";
        String defaultFS = System.getenv("MASTER_HADOOP_URL");

        // Get and parse arguments
        int argIndex = 0;
        while (argIndex < args.length) {
            String arg = args[argIndex++];
            if (arg.equals("-K")) {
                K = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-f")) {
                frag = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-t")) {
                trainingSet_name = args[argIndex++];
            }else if (arg.equals("-v")) {
                validationSet_name = args[argIndex++];
            }
        }
        if (trainingSet_name.equals("") || validationSet_name.equals("")){
            System.out.println("[ERROR] - You need to choose a file to train and test");
            System.exit(0);
        }


        HDFS dfs =  new HDFS(defaultFS);
        ArrayList<Block> FILE_TRAIN_SPLITS = dfs.findALLBlocks(trainingSet_name);
        ArrayList<Block> FILE_TEST_SPLITS  = dfs.findALLBlocks(validationSet_name);

        int numFrags = FILE_TEST_SPLITS.size();

        System.out.println("Running with the following parameters:");
        System.out.println("- K Neighborhood: " + K);
        System.out.println("- Training set: " + trainingSet_name);
        System.out.println("- Test set: " + validationSet_name);
        System.out.println("- Frag Number:" + numFrags+"\n");

        Samples[] test_result = new Samples[numFrags];
        Samples[] test_file = new Samples[numFrags];
        int[][] correct = new int[numFrags][2];

        //Start the arrays,
        //Read the test set and classifier
        for(int i=0; i<numFrags;i++){
            test_file[i] = new Samples();// needed to start the object
            test_result[i]  = new Samples();// needed to start the object
            readBlockFromHDFS(FILE_TEST_SPLITS.get(i), test_file[i]);
            ClassifyBlock(test_file[i],FILE_TRAIN_SPLITS, test_result[i],K);
            evaluateFrag(test_file[i],test_result[i],correct[i]);
        }

        //Accumulate
        int size = numFrags;
        int i = 0;
        int gap = 1;
        while (size > 1) {
            accumulate_error(correct[i], correct[i + gap]);
            size--;
            i = i + 2 * gap;
            if (i == numFrags) {
                gap *= 2;
                i = 0;
            }
        }

        //Evaluation
        System.out.println("Accuracy: " + (double) correct[0][0] / correct[0][1] * 100 + "%");


    }
}