package SVM.HDFS;


/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  SVM using data from HDFS
 *
 */



import integration.Block;
import integration.HDFS;

import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class SVMHDFS {

    public static double[] calc_yp( Sample XY, double[] w, int numDim){
        long startTime = System.nanoTime();

        double[] ypp = new double[XY.getSize()];

        for(int m=0;m<XY.getSize();m++){  //for all elements of the block
            ypp[m] = 0;
            for(int d=0;d<numDim;d++) {
                ypp[m] += XY.getFeature(m, d) * w[d];
            }

        }

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - calc_yp-> Time elapsed: %.4f seconds\n\n",seconds);
        return ypp;
    }

    public static double[] calc_CostAndGrad(int numDim, int f, double lambda, double[] w,
                                            double[] yp, Sample XY,  double[] COST){
        long startTime = System.nanoTime();

        double cost =0;

        double[] grad = new double[numDim];

        for(int m=0;m<XY.getSize();m++) {
            for (int d = 0; d < numDim; d++) {
                if (XY.getLabel(m) * yp[m] - 1 < 0)
                    grad[d] -= XY.getLabel(m) * XY.getFeature(m, d);
            }

            if (XY.getLabel(m) * yp[m] - 1 < 0) {
                cost += (1 - XY.getLabel(m) * yp[m]);
            }
        }

        if(f == 0 )
            for(int d=0;d<numDim;d++) {
                grad[d] += Math.abs(lambda * w[d]);
                cost += 0.5*lambda*w[d]*w[d];
            }

        COST[0] = cost;

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - partial_grad -> Time elapsed: %.4f seconds\n\n",seconds);

        return  grad;
    }


    public static Sample loadfileFromHDFS(Block blk,  int numDim) {

        long startTime = System.nanoTime();

        ArrayList<Integer> Labels = new ArrayList<Integer>();
        ArrayList<double[]> Features = new ArrayList<double[]>();
        Sample train = new Sample();

        try {
            StringTokenizer tokenizer;

            int index=0;

            String lines[] = blk.getRecords('\n');
            while(index<lines.length){

                if(index%10000 == 0 )
                    System.out.println("Line:"+ index);

                String line = lines[index];
                tokenizer = new StringTokenizer(line, ",");
                if(tokenizer.countTokens()>1) {

                    int label = (int) Float.parseFloat(tokenizer.nextToken());

                    if (label >0)
                        Labels.add(label);
                    else
                        Labels.add(-1);

                    int index2 = 0;
                    double[] feature = new double[numDim];
                    while (tokenizer.hasMoreTokens()) {
                        feature[index2] = Double.parseDouble(tokenizer.nextToken());
                        index2++;
                    }
                    Features.add(feature);
                }
                index++;
            }

            train.setFeatures(Features);
            train.setLabels(Labels);


            long estimatedTime = System.nanoTime() - startTime;
            double seconds = (double)estimatedTime / 1000000000.0;
            System.out.printf("loadfileFromHDFS->Time elapsed: %.4f seconds\n#\n",seconds);

            return train;

        }catch (Exception e) {
            System.out.println("ERROR - SVM.HDFS.loadfileFromHDFS");
            e.printStackTrace();
        }
        return null;
    }


    public static void  updateWeight(double lr, double[] grad_p,double [] w){

        for(int d=0;d<w.length;d++){
            w[d] -= lr*grad_p[d];
        }

    }


    public static void accumulateCostAndGrad(double[] grad, double[] grad2,
                                             double[] COST, double[] COST2){
        for(int d=0;d<grad2.length;d++)
            grad[d] +=grad2[d];

        COST[0] += COST2[0];
    }


    public static ArrayList<Integer> predict_chunck(Sample XY, double[] w){
        long startTime = System.nanoTime();

        ArrayList<Integer> r = new ArrayList<Integer>();
        for(int i=0;i<XY.getSize();i++)	{
            r.add( predict(XY.getFeatureAll(i), w));
        }

        double seconds = (double)(System.nanoTime() - startTime)/ 1000000000.0;
        System.out.printf("[INFO] - predict_chunck -> Time elapsed: %.4f seconds\n\n",seconds);

        return r;
    }

    public static int predict(double[] x, double[] w){
        double pre=0;
        for(int j=0;j<x.length;j++){
            pre+=x[j]*w[j];
        }
        if(pre >=0)
            return   1;
        else return -1;
    }


    //savePredictionToFile: to save the predicition in a file
    public static void savePredictionToHDFS(ArrayList<Integer> result, String defaultFS, String filename){
        HDFS dfs = new HDFS(defaultFS);

        String file1 = "";
        for (int i = 0; i < result.size(); i++) {
            int tmp = result.get(i);
            if (tmp != 0)
                file1 +=tmp+"\n";
        }
        dfs.writeFILE(file1,defaultFS+filename,false);
        System.out.println("File "+filename+" saved");
    }


    public static void main(String[] args) throws IOException{

        /*###########################################

             #### Load and set the environment  ####

          ###########################################*/

        String trainFile = "";
        String testFile =  "";
        String path_output = "";
        int numDim  = 0;
        int numFrag = 0;
        Boolean force_split = false;

        //SVM's parameters
        double lambda = 0.001;      // Coefficient for Penalty part  (regularization parameter) -> 0 to +inf
        double lr = 0.0001;         // Learning rate parameter
        double threshold = 0.001;   // Cost's threshold -> Tolerance for stopping criterion
        int maxIters = 3;

        // Get and parse arguments
        int argIndex = 0;
        while (argIndex < args.length) {
            String arg = args[argIndex++];
            if (arg.equals("-c")) {
                lambda = Double.parseDouble(args[argIndex++]);
            }else if (arg.equals("-lr")) {
                lr = Double.parseDouble(args[argIndex++]);
            }else if (arg.equals("-thr")) {
                threshold = Double.parseDouble(args[argIndex++]);
            }else if (arg.equals("-nd")) {
                numDim = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-i")) {
                maxIters = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-t")) {
                trainFile = args[argIndex++];
            }else if (arg.equals("-v")) {
                testFile = args[argIndex++];
            }
            else if (arg.equals("-f")) {
                numFrag = Integer.parseInt(args[argIndex++]);
            }else if (arg.equals("-out")) {
                force_split=true;
                path_output = args[argIndex++];
            }
        }

        System.out.println("Running SVM.HDFS with the following parameters:");
        System.out.println("- Lambda: " + lambda);
        System.out.println("- Learning rate: " + lr);
        System.out.println("- Threshold: " + threshold);
        System.out.println("- Iterations: " + maxIters);
        System.out.println("- Dimensions: " + numDim);
        System.out.println("- Train Points: " + trainFile);
        System.out.println("- Test Points: " + testFile);
        System.out.println("- Output path: " + path_output+"\n");


        //Finding the list of blocks
        String defaultFS = System.getenv("MASTER_HADOOP_URL");
        HDFS dfs =  new HDFS(defaultFS);
        ArrayList<Block> FILE_TRAIN_SPLITS;
        ArrayList<Block> FILE_TEST_SPLITS;
        if (force_split) {
            FILE_TRAIN_SPLITS = dfs.findBlocksByRecords(trainFile, numFrag);
            FILE_TEST_SPLITS  = dfs.findBlocksByRecords(testFile, numFrag);
        }else{
            FILE_TRAIN_SPLITS = dfs.findALLBlocks(trainFile);
            FILE_TEST_SPLITS = dfs.findALLBlocks(testFile);
        }

        int numFrag1 = FILE_TRAIN_SPLITS.size();
        int numFrag2 = FILE_TEST_SPLITS.size();

        Sample[] train = new Sample[numFrag1];

        for(int f=0; f<numFrag1;f++)
            train[f] = loadfileFromHDFS(FILE_TRAIN_SPLITS.get(f), numDim);




        /*########################################

                #### Create Model ####

         ########################################*/


        double[]        w = new double[numDim];
        double[][] grad_p = new double[numFrag1][numDim];
        double[][]   COST = new double[numFrag1][2];


        for(int iter=0;iter<maxIters;iter++){

            for (int f=0;f<numFrag1;f++) {
                double[] yp = calc_yp(train[f], w, numDim );
                grad_p[f] = calc_CostAndGrad(numDim,f, lambda,w,  yp,  train[f], COST[f] );
            }

            //Accumulate gradient and cost
            int size = numFrag1;
            int i = 0;
            int gap = 1;
            while (size > 1) {
                accumulateCostAndGrad(grad_p[i], grad_p[i + gap], COST[i],   COST[i + gap]);
                size--;
                i = i + 2 * gap;
                if (i == numFrag1) {
                    gap *= 2;
                    i = 0;
                }
            }



            System.out.println("[INFO] - Current Cost: "+ COST[0][0]);
            if(COST[0][0]< threshold){
                System.out.println("[INFO] - Final Cost: "+ COST[0][0]);
                break;
            }

            //Step:Update
            updateWeight(lr,grad_p[0],w);

        }

       /*########################################

                #### Test Model ####

         ########################################*/

        Sample[] test  = new Sample[numFrag2];

        for(int f=0; f<numFrag2;f++)
            test[f] = loadfileFromHDFS(FILE_TEST_SPLITS.get(f), numDim);


        for (int f =0; f<numFrag2;f++) {
            ArrayList<Integer> labels_result = predict_chunck(test[f], w);
            savePredictionToHDFS(labels_result,defaultFS,path_output+"/Result-SVM_HDFS_"+f+".out");

        }


    }

}


