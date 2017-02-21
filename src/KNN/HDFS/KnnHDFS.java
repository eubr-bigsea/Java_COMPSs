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


    public static void ClassifyBlock(Samples test_blk, Block b1, Candidate partial_result, int K){
        long startTime = System.nanoTime();

        Samples trainingSet = new Samples();
        String[] tokens;

        try{
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

        } catch (Exception e) {
            System.out.println("Error: ClassifierFromHDFSblock - Part1");
            e.printStackTrace();
        }

        System.out.println("[INFO] - Readed "+trainingSet.getSizebyFeatures()+" records");
        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("[INFO] - Readed Training file -> Time elapsed: %.2f seconds\n",seconds0);

        for (int s =0; s<test_blk.getSizebyFeatures();s++) {
            if(s% 100 == 0)
                System.out.println("Record:" + s);
            double[] cols = test_blk.getFeatureAll(s);
            double [][] bestPoints = new double[2][K+1];
            //Candidate c = new Candidate();
           // c.setK(K+1);

            double tmp_dist = 0;
            double tmp_label = 0;
            //Setting up
            for (int d=0; d<K;d++){
                bestPoints[0][d] = -1.0;
                bestPoints[1][d] = Double.MAX_VALUE;
            }

            for (int s2 = 0; s2<trainingSet.getSizebyFeatures();s2++) {
                bestPoints[1][K] = distance(trainingSet.getFeatureAll(s2), cols);
                bestPoints[0][K] = trainingSet.getLabel(s2);

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

            partial_result.addDistances(bestPoints[1]);
            partial_result.addLabels(bestPoints[0]);


//            for(int d=0; d<bestPoints[1].length;d++)
//                System.out.print(bestPoints[1][d]+" ");
//            System.out.println(" ");

          //  int label = classify(trainingSet,cols, K);
          //  partial_result.addLabels(label);
        }

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("\n[INFO] - ClassifierFromHDFSblock -> Time elapsed: %.2f seconds\n",seconds);
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


    public static void evaluateFrag(Samples part, double[] partial, int[] correct){
        long startTime = System.nanoTime();

        for (int i = 0; i<part.getSizebyLabels();i++) {
           // if(i%5000 == 0)
            //    System.out.println(partial[i] +" " + part.getLabel(i));
            if (partial[i] == part.getLabel(i))
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


    public static Candidate merge(Candidate result, Candidate partial_result, int K){
        long startTime = System.nanoTime();

        Candidate tmp = new Candidate();

        for (int i_record=0;i_record<result.getSize();i_record++){
            int i_left =0;
            int i_right=0;

            double[] left_dist = result.getDistance(i_record);
            double[] left_lab = result.getLabel(i_record);
            double[] right_dist = partial_result.getDistance(i_record);
            double[] right_lab = partial_result.getLabel(i_record);

            double[] record_d = new double[K+1];
            double[] record_l = new double[K+1];
//            System.out.print("L ");
//            for(int d=0; d<left_dist.length;d++)
//                System.out.print(left_dist[d]+" ");
//            System.out.println(" ");
//
//            System.out.print("R ");
//            for(int d=0; d<right_dist.length;d++)
//                System.out.print(right_dist[d]+" ");
//            System.out.println(" ");

            for (int s=0;s<K;s++) {
                if (left_dist[i_left] > right_dist[i_right]) {
                    record_d[s] = right_dist[i_right];
                    record_l[s] = right_lab[i_right];
                    i_right++;
                } else {
                    record_d[s] = left_dist[i_left];
                    record_l[s] = left_lab[i_left];
                    i_left++;
                }
            }
            tmp.addRecord(record_d,record_l);

//            System.out.print("M ");
//            for(int d=0; d<record_d.length;d++)
//                System.out.print(record_d[d]+" ");
//            System.out.println(" --- ");


        }
        result=tmp;

        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("\n[INFO] - merge -> Time elapsed: %.2f seconds\n",seconds0);
        return result;
    }

    public static double[] getKN(Candidate neighborhood,int K){
        long startTime = System.nanoTime();

        int size = neighborhood.getSize();
        double[] result = new double[size];
        for (int i =0;i<size;i++){
//            double[] dist = neighborhood[i].getDistances();
//            for(int d=0; d<dist.length;d++)
//                System.out.print(dist[d]+" ");
//            System.out.println(" ");
            result[i] = getPopularElement(neighborhood.getLabel(i));
        }

        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("\n[INFO] - getKN -> Time elapsed: %.2f seconds\n",seconds0);

        return result;
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
            } else if (arg.equals("-t")) {
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
        int numFrags2 = FILE_TRAIN_SPLITS.size();

        System.out.println("Running with the following parameters:");
        System.out.println("- K Neighborhood: " + K);
        System.out.println("- Training set: " + trainingSet_name);
        System.out.println("- Test set: " + validationSet_name);
        System.out.println("- Frag Number:" + numFrags+"\n");


        Samples[] test_file = new Samples[numFrags];
        int[][] numcorrect = new int[numFrags][2];

        Candidate [][] partial_result = new Candidate[frag][frag];
        //Read the test set and classifier
        for(int f1=0; f1<numFrags;f1++){
            test_file[f1] = new Samples();// needed to start the object
            readBlockFromHDFS(FILE_TEST_SPLITS.get(f1), test_file[f1]);

            for(int f2=0; f2<numFrags2;f2++) {
                partial_result[f1][f2] =  new Candidate();// needed to start the object
                ClassifyBlock(test_file[f1], FILE_TRAIN_SPLITS.get(f2), partial_result[f1][f2], K);
            }

            //Merge Results
            int size = frag;
            int i = 0;
            int gap = 1;
            while (size > 1) {
                partial_result[f1][i] = merge(partial_result[f1][i], partial_result[f1][i + gap],K);
                size--;
                i = i + 2 * gap;
                if (i == frag) {
                    gap *= 2;
                    i = 0;
                }
            }

            double[] result_part_final = getKN(partial_result[f1][0],K);
            evaluateFrag(test_file[f1],result_part_final,numcorrect[f1]);
        }

        //Accumulate
        int size = numFrags;
        int i = 0;
        int gap = 1;
        while (size > 1) {
            accumulate_error(numcorrect[i], numcorrect[i + gap]);
            size--;
            i = i + 2 * gap;
            if (i == numFrags) {
                gap *= 2;
                i = 0;
            }
        }

        //Evaluation
        System.out.println("Accuracy: " + (double) numcorrect[0][0] / numcorrect[0][1] * 100 + "%");

    }
}