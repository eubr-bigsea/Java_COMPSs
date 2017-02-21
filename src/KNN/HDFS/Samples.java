package KNN.HDFS;

import java.io.Serializable;
import java.util.ArrayList;

/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood) using data of HDFS
 *
 */

public class Samples implements Serializable {

    public ArrayList<double[]> Features = new ArrayList<double[]>();
    public ArrayList<Integer>  Labels  = new ArrayList<Integer>();


    public Samples() { }

    public void setFeatures(ArrayList<double[]> features) {
        Features = features;
    }
    public ArrayList<double[]> getFeatures() {
        return Features;
    }

    public double getFeature(int m,int d){
        return Features.get(m)[d];
    }
    public double[] getFeatureAll(int m){
        return Features.get(m);
    }
    public int getLabel(int d){
        return Labels.get(d);
    }



    public void start(){
        Labels = new ArrayList<Integer>();
        Features = new ArrayList<double[]>();
    }

    public void addLabels(int l){
        if (Labels == null) //DELETE
            Labels = new ArrayList<Integer>();

        Labels.add(l);
    }

    public void addFeature(double[] f){
        if (Features == null)
            Features = new ArrayList<double[]>();
        Features.add(f);
    }

    public ArrayList<Integer> getLabels() {
        return Labels;
    }

    public void setLabels(ArrayList<Integer> labels) {
        Labels = labels;
    }

    public int getSizebyLabels(){
        return Labels.size();
    }
    public int getSizebyFeatures(){
        return Features.size();
    }
}
