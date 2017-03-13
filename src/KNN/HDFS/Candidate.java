package KNN.HDFS;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by lucasmsp on 20/02/17.
 */
public class Candidate implements Serializable{

    ArrayList<int[]> labels = new ArrayList<int[]>();
    ArrayList<double[]> distances = new ArrayList<double[]>();
    int K =1;


    public Candidate() {
    }

    public void setK(int K){
        this.K = K;
        //labels = new double[K+1];
       // distances = new double[K+1];
    }

    public int getK() {
        return K;
    }

    public int getSize(){
        return labels.size();
    }

    public ArrayList<int[]> getALLLabels() {
        return labels;
    }
    public ArrayList<double[]> getALLDistances() {
        return distances;
    }


    public void setLabels(ArrayList<int[]>  labels) {
        this.labels = labels;
    }


    public double[] getDistance(int i){
        return distances.get(i);

    }

    public int[] getLabel(int i){
        return labels.get(i);
    }

    public void addRecord(double[] dist, int[] label){
        distances.add(dist);
        labels.add(label);
    }



    public void setDistances(ArrayList<double[]> distances) {
        this.distances = distances;
    }

    public void addDistances(double[] dists){
        distances.add(dists);
    }

    public void addLabels(int[] lab){
        labels.add(lab);
    }


    public void addDistance(int i,int d,double dist){
        distances.get(i)[d] = dist;
    }
    public void addLabel(int i,int d, int l){
        labels.get(i)[d]=l;
    }
}
