package KNN.files;

/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood)
 *
 */

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Knn {


    public static ArrayList<Sample> readFile(String file)  {

        long startTime = System.nanoTime();
        Sample sample = new Sample();
        ArrayList<Sample> samples = new ArrayList<Sample>();
        int j =0;

        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line = null; //reader.readLine(); // ignore header
            while((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");
                if (tokens.length>1) {
                    if(j% 10000 == 0)
                        System.out.println("Line:"+j);
                    j++;
                    sample = new Sample();
                    sample.setLabel((int) Float.parseFloat(tokens[0]));

                    double[] col = new double[tokens.length - 1];
                    for (int i = 1; i < tokens.length; i++) {
                        col[i - 1] = Double.parseDouble(tokens[i]);
                    }
                    sample.setCols(col);
                    samples.add(sample);
                }
            }
            reader.close();
        } catch (FileNotFoundException e) {
            System.out.println("[ERROR] - readFile");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("[ERROR] - readFile");
            e.printStackTrace();
        }

        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("\n[INFO] - readFile -> Time elapsed: %.2f seconds\n",seconds0);

        return samples;
    }

    public static double distance(double[] a, double[] b) {
        int sum = 0;
        for(int i = 0; i < a.length; i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return Math.sqrt(sum); // euclidian distance would be sqrt(sum)...
    }


    public static double[] getKN(Candidate[] neighborhood,int K){
        long startTime = System.nanoTime();

        double[] result = new double[neighborhood.length];
        for (int i =0;i<neighborhood.length;i++){
//            double[] dist = neighborhood[i].getDistances();
//            for(int d=0; d<dist.length;d++)
//                System.out.print(dist[d]+" ");
//            System.out.println(" ");
            result[i] = getPopularElement(neighborhood[i].getLabels());
        }

        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("\n[INFO] - getKN -> Time elapsed: %.2f seconds\n",seconds0);

        return result;
    }

    public static double getPopularElement(double[] a){


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


    public static void classifyFrag(ArrayList<Sample> trainingSet, ArrayList<Sample> testSet, int K, Candidate[] partial_result) {
        long startTime = System.nanoTime();

//        System.out.println("test:" + testSet.size());
//        System.out.println("train" + trainingSet.size());
//        System.out.println("K:"+ K);



        int f =-1;
        for (Sample sampleTest : testSet) {
            f++;
            double[][] tmp = new double[2][K+1];
            Candidate c = new Candidate();
            c.setK(K+1);

            if(f%1000 == 0)
                System.out.println("Record:"+f);

            for (int k = 0; k < K; k++) {
                tmp[0][k] = -1.0;
                tmp[1][k] = Double.MAX_VALUE;
            }

            for (Sample sampleTrain : trainingSet) {

                tmp[1][K ] = distance(sampleTest.getCols(), sampleTrain.getCols());
                tmp[0][K ] = sampleTrain.getLabel();

                double tmp_dist;
                double tmp_label;

                for (int j = K; j > 0; j--) { //Insert Sort

                    if (tmp[1][j] < tmp[1][j - 1]) {
                        tmp_label = tmp[0][j];
                        tmp[0][j] = tmp[0][j - 1];
                        tmp[0][j - 1] = tmp_label;

                        tmp_dist = tmp[1][j];
                        tmp[1][j] = tmp[1][j - 1];
                        tmp[1][j - 1] = tmp_dist;
                    }
                }
            }
            c.setDistances(tmp[1]);
            c.setLabels(tmp[0]);

            partial_result[f]=c;
            partial_result[f]=c;


//            double[] dist = partial_result[f].getDistances();
//            for(int d=0; d<dist.length;d++)
//                System.out.print(dist[d]+" ");
//            System.out.println(" ");
        }




        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double) estimatedTime / 1000000000.0;
        System.out.printf("\n[INFO] - classifyFrag -> Time elapsed: %.2f seconds\n",seconds);
        //return partial_result.clone();

    }

    public static Candidate[] merge(Candidate[] result, Candidate[] partial_result, int K){
        long startTime = System.nanoTime();


        for (int i_record=0;i_record<result.length;i_record++){
            int i_left =0;
            int i_right=0;

            double[] left_dist = result[i_record].getDistances();
            double[] left_lab = result[i_record].getLabels();
            double[] right_dist = partial_result[i_record].getDistances();
            double[] right_lab = partial_result[i_record].getLabels();

            Candidate tmp = new Candidate();
            tmp.setK(K);

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
                    tmp.setDistance(s,right_dist[i_right]);
                    tmp.setLabel(s,right_lab[i_right]);
                    i_right++;
                } else {
                    tmp.setDistance(s,left_dist[i_left]);
                    tmp.setLabel(s,left_lab[i_left]);
                    i_left++;
                }
            }

//            System.out.print("M ");
//            double[] oi = tmp.getDistances();
//            for(int d=0; d<oi.length;d++)
//                System.out.print(oi[d]+" ");
//            System.out.println(" --- ");

            result[i_record]=tmp;
        }


        long estimatedTime0 = System.nanoTime() - startTime;
        double seconds0 = (double) estimatedTime0 / 1000000000.0;
        System.out.printf("\n[INFO] - merge -> Time elapsed: %.2f seconds\n",seconds0);
        return result.clone();
    }



    public static void evaluateFrag(ArrayList<Sample>part, double[] partial, int[] correct){
        long startTime = System.nanoTime();

        for (int i = 0; i<part.size();i++) {
            if (partial[i] == part.get(i).getLabel())
                correct[0]++;
        }

        System.out.println("Correct:"+ correct[0]);

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double) estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - evaluateFrag -> Time elapsed: %.2f seconds\n\n",seconds);

    }

    public static void accumulate(int[] numcorrect, int[] numcorrect2){

        for(int f=0; f<numcorrect2.length;f++)
            numcorrect[f] += numcorrect2[f];

    }

    public static void main(String[] args) throws IOException {

        int frag = 2;
        int K = 1;
        String trainingName = "";
        String testName     = "";

        // Get and parse arguments
        int argIndex = 0;
        while (argIndex < args.length) {
            String arg = args[argIndex++];
            if (arg.equals("-K")) {
                K = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-f")) {
                frag = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-t")) {
                trainingName = args[argIndex++];
            }else if (arg.equals("-v")) {
                testName = args[argIndex++];
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
        System.out.println("- Nodes: " + frag);
        System.out.println("- Training set: " + trainingName);
        System.out.println("- Test set: " + testName+"\n");

        ArrayList<Sample> trainingSet   = readFile(trainingName);   //Only in Master
        ArrayList<Sample> TestSet       = readFile(testName);       //Only in Master

        int TestSize = TestSet.size();
        int TrainSize = trainingSet.size();
        int start_v = 0;
        int end_v = TestSize/frag;
        int i;

        int[][] numcorrect = new int[frag][1]; //only to compute
        Candidate [][][] partial_result = new Candidate[frag][frag][TestSize/frag];
        Candidate c = new Candidate();
        c.setK(K+1);
        for (int f1 = 0 ; f1<frag;f1++) {
            for(int f2= 0;f2<frag;f2++) {
                for(int f3= 0;f3<TestSize/frag;f3++)
                    partial_result[f1][f2][f3] = c;
            }
        }


        //Split the TestSet and classifying it
        for (int f1 = 0 ; f1<frag;f1++) {
            ArrayList<Sample> test_part = new ArrayList<Sample>(TestSet.subList(start_v, end_v));
            start_v = start_v + TestSize/frag;
            end_v   = end_v   + TestSize/frag;

            int start_t = 0;
            int end_t=TrainSize/frag;
            for(int f2= 0;f2<frag;f2++) { //change after
                ArrayList<Sample> train_part = new ArrayList<Sample>(trainingSet.subList(start_t, end_t));
                start_t = start_t + trainingSet.size() / frag;
                end_t = end_t + trainingSet.size() / frag;
                classifyFrag(train_part, test_part,K,partial_result[f1][f2]);
            }


            //Merge Results
            int size = frag;
            i = 0;
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
            evaluateFrag(test_part,result_part_final,numcorrect[f1]);

        }


        //Accumulate Evaluation
        int size = frag;
        i = 0;
        int gap = 1;
        while (size > 1) {
            accumulate(numcorrect[i], numcorrect[i + gap]);
            size--;
            i = i + 2 * gap;
            if (i == frag) {
                gap *= 2;
                i = 0;
            }
        }

        System.out.println("Accuracy: " + (double) numcorrect[0][0] / TestSize * 100 + "%");

    }
}