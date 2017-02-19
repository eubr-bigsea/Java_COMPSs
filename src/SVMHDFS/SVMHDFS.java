package SVMHDFS;


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


    public static Sample loadfileFromHDFS(Block blk,  int numDim) {

        long startTime = System.nanoTime();

        ArrayList<Integer> Labels = new ArrayList<Integer>();
        ArrayList<double[]> Features = new ArrayList<double[]>();
        Sample train = new Sample();

        try {
            StringTokenizer tokenizer;

            int index=0;
            int readed = 0;

            String lines[] = blk.getRecords();
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
            System.out.println("ERROR - SVMHDFS.loadfileFromHDFS");
            e.printStackTrace();
        }
        return null;
    }


    public static void  calc_yp( Sample XY, double[] w, int numDim, double[] COST, Sample s){

        long startTime = System.nanoTime();

        double[] ypp = new double[XY.getSize()];
        double cost =0;

        for(int m=0;m<XY.getSize();m++){  //for all elements of the block
            ypp[m] = 0;
            for(int d=0;d<numDim;d++) {
                ypp[m] += XY.getFeature(m, d) * w[d];
            }

            if (XY.getLabel(m) * ypp[m] - 1 < 0) {
                cost += (1 - XY.getLabel(m) * ypp[m]);
            }

        }

        s.addFeature(ypp);
        COST[0] = cost;

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - calc_yp-> Time elapsed: %.4f seconds\n\n",seconds);

    }

    public static void partial_grad(int numDim, Sample yp, Sample XY, double[] grad){
        long startTime = System.nanoTime();

        for (int d = 0; d < numDim; d++)
            grad[d] = 0;

        for(int m=0;m<XY.getSize();m++) {
            for (int d = 0; d < numDim; d++) {
                if (XY.getLabel(m) * yp.getFeature(0, m) - 1 < 0)
                    grad[d] -= XY.getLabel(m) * XY.getFeature(m, d);
            }
        }

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - partial_grad -> Time elapsed: %.4f seconds\n\n",seconds);
    }

    public static void  updateWeight(double lr, double[] grad_p,double [] w){

        for(int d=0;d<w.length;d++){
            w[d] -= lr*grad_p[d];
        }

    }



    public static Sample predict_chunck(Sample XY, double[] w){
        long startTime = System.nanoTime();

        ArrayList<Integer> r = new ArrayList<Integer>();
        for(int i=0;i<XY.getSize();i++)	{
            r.add( predict(XY.getFeatureAll(i), w));
        }
        Sample s = new Sample();
        s.setLabels(r);

        double seconds = (double)(System.nanoTime() - startTime)/ 1000000000.0;
        System.out.printf("[INFO] - predict_chunck -> Time elapsed: %.4f seconds\n\n",seconds);

        return s;
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

    public static void accumulate_error(int[] ACC,int[] ACC2){

        for(int f=0; f<ACC2.length;f++)
            ACC[f] += ACC2[f];

    }

    public static void accumulate_cost(double[] COST, double[] COST2){
        //for(int f=0; f<COST2.length;f++)
        COST[0] += COST2[0];

    }

    public static void accumulate_grad(double[] grad, double[] grad2){
        for(int d=0;d<grad2.length;d++)
            grad[d] +=grad2[d];
    }

    public static void verify (Sample labels_result, Sample test, int[] ACC){
        int error=0;
        int total = 0;
        for(int i=0;i<test.getSize();i++)	{
            total++;
            if(labels_result.getLabel(i)!=test.getLabel(i))
                error++;
        }

        ACC[0] = error;
        ACC[1] = total;

    }





    public static void main(String[] args) throws IOException{

        /*###########################################

             #### Load and set the environment  ####

          ###########################################*/

        String trainFile = "/user/pdm2/higgs-train-0.1m.csv";
        String testFile = "/user/pdm2/higgs-train-0.1m.csv";
        int sizeTest = 100000;
        int numDim = 28;

        //Finding the list of blocks
        String defaultFS = System.getenv("MASTER_HADOOP_URL");
        HDFS dfs =  new HDFS(defaultFS);
        ArrayList<Block> FILE_TRAIN_SPLITS = dfs.findALLBlocks(trainFile);
        ArrayList<Block> FILE_TEST_SPLITS  = dfs.findALLBlocks(testFile);

        int numFrag1 = FILE_TRAIN_SPLITS.size();
        int numFrag2 = FILE_TEST_SPLITS.size();

        Sample[] train = new Sample[numFrag1];
        Sample[] test  = new Sample[numFrag2];

        double lambda = 0.001; //0.0001;
        double lr = 0.0001; //0.001;
        double threshold = 0.001;
        int maxIters = 3;


        for(int f=0; f<numFrag1;f++)
            train[f] = loadfileFromHDFS(FILE_TRAIN_SPLITS.get(f), numDim);

        for(int f=0; f<numFrag2;f++)
            test[f] = loadfileFromHDFS(FILE_TEST_SPLITS.get(f), numDim);


        /*########################################

                #### Create Model ####

         ########################################*/


        double[]        w = new double[numDim];
        double       cost = 0;
        double[][] grad_p = new double[numFrag1][numDim];
        Sample[] yp       = new Sample[numFrag1];
        double[][]   COST = new double[numFrag1][2];
        int[][]       ACC = new int[numFrag2][2];

        for(int iter=0;iter<maxIters;iter++){


            //Calc yp
            for (int f=0;f<numFrag1;f++) {
                yp[f] = new Sample();
                yp[f].start();
                calc_yp(train[f], w, numDim, COST[f], yp[f]);
                partial_grad(numDim,  yp[f],  train[f], grad_p[f]);
            }

            //Accumulate gradient and cost
            int size = numFrag1;
            int i = 0;
            int gap = 1;
            while (size > 1) {
                accumulate_grad(grad_p[i], grad_p[i + gap]);
                accumulate_cost(COST[i],   COST[i + gap]);
                size--;
                i = i + 2 * gap;
                if (i == numFrag1) {
                    gap *= 2;
                    i = 0;
                }
            }


            for(int d=0;d<numDim;d++) {
                grad_p[0][d] += Math.abs(lambda * w[d]);
                COST[0][0] += 0.5*lambda*w[d]*w[d];
            }

            System.out.println("[INFO] - Current Cost: "+ COST[0][0]);
            if(COST[0][0]< threshold){
                break;
            }

            //Step:Update
            updateWeight(lr,grad_p[0],w);

        }

       /*########################################

                #### Test Model ####

         ########################################*/

        int error=0;
        Sample[] labels_result = new Sample[numFrag2];
        //Parallel
        for (int f =0; f<numFrag2;f++) {
            labels_result[f] = predict_chunck(test[f], w);
            verify(labels_result[f], test[f], ACC[f]);

        }

        //Accumulate Error
        int size = ACC.length;
        int i = 0;
        int gap = 1;
        while (size > 1) {
            accumulate_error(ACC[i], ACC[i + gap]);
            size--;
            i = i + 2 * gap;
            if (i == ACC.length) {
                gap *= 2;
                i = 0;
            }
        }

        System.out.println("Total Length:"+ACC[0][1]);
        System.out.println("Error:"+ACC[0][0]);
        System.out.println("Error rate:"+((double)ACC[0][0]/ACC[0][1]));
        System.out.println("Acc rate:"+((double)(ACC[0][1]-ACC[0][0])/ACC[0][1]));

        /*

            cost:100000.0
            cost:395456.9721037722
            cost:3910008.9329382903
            Iters:3
            total:100000
            error:47166
            error rate:0.47166
            acc rate:0.52834

         */

    }

}


