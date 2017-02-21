package SVM.files;


/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  SVM
 *
 */



import java.io.*;
import java.util.StringTokenizer;

public class SVM {


    public static void  calc_yp(double[] ypp, double COST[], double[][] X, int[] label, double[] w, int numDim ){
        long startTime = System.nanoTime();
        double cost =0;

        for(int m=0;m<X.length;m++){
            ypp[m]=0;
            for(int d=0;d<numDim;d++)
                ypp[m]+=X[m][d]*w[d];

            if (label[m]*ypp[m]-1< 0) {
                cost += (1 - label[m] * ypp[m]);
            }
        }


        COST[0]=cost;

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - calc_yp -> Time elapsed: %.2f seconds\n\n",seconds);

    }

    public static void  partial_grad(double[] grad_p,int numDim, double[] yp, double[][] train_features, int[] train_labels){

        long startTime = System.nanoTime();

        for(int d=0;d<numDim;d++) {
            grad_p[d] = 0;
            for (int m = 0; m < train_labels.length; m++)
                if (train_labels[m]*yp[m]-1 < 0)
                    grad_p[d] -= train_labels[m]*train_features[m][d];
        }

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - partial_grad -> Time elapsed: %.2f seconds\n\n",seconds);

    }

    public static void accumulate_cost(double[] COST, double[] COST2){
        COST[0] += COST2[0];
    }

    public static void accumulate_grad(double[] grad, double[] grad2){
        for(int d=0;d<grad2.length;d++)
            grad[d] +=grad2[d];
    }

    public static void accumulate_error(int[] ACC,int[] ACC2){
        for(int f=0; f<ACC2.length;f++)
            ACC[f] += ACC2[f];
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


    public static void verify (int[] labels_result, int[] test, int[] ACC){
        int error=0;
        int total = 0;
        for(int i=0;i<test.length;i++)	{
            total++;
            if(labels_result[i]!=test[i])
                error++;
        }

        ACC[0] = error;
        ACC[1] = total;

    }


    public static int[] predict_chunck(int[] testY, double[][] testX,  double[] w){
        long startTime = System.nanoTime();

        int[] label_result = new  int[testX.length];
        for(int i=0;i<testX.length;i++)	{
            label_result[i] = predict(testX[i], w);
        }

        long estimatedTime = System.nanoTime() - startTime;
        double seconds = (double)estimatedTime / 1000000000.0;
        System.out.printf("[INFO] - predict_chunck -> Time elapsed: %.2f seconds\n\n",seconds);

        return label_result;
    }


    public static void  updateWeight(double lr, double[] grad_p,double [] w){

        for(int d=0;d<w.length;d++){
            w[d] -= lr*grad_p[d];
        }

    }

    public static void loadfile_and_split(double[][][] features, int[][] labels, String fileName, int sizeTrainPerFrag,int numTotal) {

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
                if(tokenizer.countTokens()>1) {
                    int label = (int) Float.parseFloat(tokenizer.nextToken());
                    if (label > 0)
                        labels[f][index] = 1;
                    else
                        labels[f][index] = -1;

                    int index2 = 0;
                    while (tokenizer.hasMoreTokens()) {
                        features[f][index][index2] = Double.parseDouble(tokenizer.nextToken());
                        index2++;
                    }
                }
                index++;
            }

            long estimatedTime = System.nanoTime() - startTime;
            double seconds = (double)estimatedTime / 1000000000.0;
            System.out.printf("[INFO] loadfile_and_split->Time elapsed: %.2f seconds\n\n",seconds);

        } catch (FileNotFoundException e) {
            System.out.println("ERROR - SVM.loadfile_and_split");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("ERROR - SVM.loadfile_and_split");
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        /*###########################################

             #### Load and set the environment  ####

          ###########################################*/

        //SVM's parameters
        double lambda = 0.001;
        double lr = 0.0001;
        double threshold = 0.001;
        int maxIters = 3;

        //Dataset's parameters
        int numDim = 28;
        int sizeTrain = 100000;
        int sizeTest  = 100000;
        int numFrag = 2;

        int sizeTrainPerFrag = (int) Math.floor((float)sizeTrain/numFrag);
        int[][] train_labels = new int[numFrag][sizeTrainPerFrag];
        double[][][] train_features = new double[numFrag][sizeTrainPerFrag][numDim];
        String trainFile = "/media/lucasmsp/Dados/TEMP/Dataset/higgs-train-0.1m.csv";

        loadfile_and_split(train_features,train_labels,trainFile, sizeTrainPerFrag,sizeTrain);


        /*########################################

                #### Create Model ####

         ########################################*/


        double[]      w = new double[numDim];
        double[][] COST = new double[numFrag][2];
        int[][]     ACC = new int[numFrag][2];
        double[][] grad_p = new double[numFrag][numDim];
        double[][] yp     = new double[numFrag][sizeTrainPerFrag];

        for(int iter=0;iter<maxIters;iter++){

            for (int f=0;f<numFrag;f++) {
                calc_yp(yp[f], COST[f], train_features[f], train_labels[f], w, numDim);
                partial_grad(grad_p[f],numDim,  yp[f],  train_features[f], train_labels[f]);
            }


            //Accumulate gradient and Cost
            int size = numFrag;
            int i = 0;
            int gap = 1;
            while (size > 1) {
                accumulate_grad(grad_p[i], grad_p[i + gap]);
                accumulate_cost(COST[i], COST[i + gap]);
                size--;
                i = i + 2 * gap;
                if (i == numFrag) {
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

        train_labels    =  null;
        train_features  =  null;

        int sizeTestPerFrag = (int) Math.floor((float)sizeTest/numFrag);
        int[][] test_labels = new int[sizeTest][sizeTestPerFrag];
        double[][][] test_features = new double[sizeTest][sizeTestPerFrag][numDim];
        String testFile = "/media/lucasmsp/Dados/TEMP/Dataset/higgs-train-0.1m.csv";

        loadfile_and_split(test_features,test_labels,testFile,sizeTestPerFrag,sizeTest);


        int error=0;
        int[][] labels_result = new int[numFrag][sizeTestPerFrag];

        //Parallel
        for (int f =0; f<numFrag;f++) {
            labels_result[f] = predict_chunck(test_labels[f], test_features[f], w);
            verify(labels_result[f], test_labels[f], ACC[f]);
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


    }

}
