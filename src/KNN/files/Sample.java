package KNN.files;

import java.io.Serializable;

/**
 *
 * Created by Lucas Miguel S Ponce.
 * Student of master degree in Computer's Science
 * UFMG - 1/2017
 * Algorithm:  k-NN (K Nearst Neighborhood)
 *
 */

public class Sample implements Serializable {
    int label;
    double [] cols;

    public Sample(double[] cols, int label) {
        this.cols = cols;
        this.label = label;
    }

    public Sample( ) {

    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public double[] getCols() {
        return cols;
    }

    public void setCols(double[] cols) {
        this.cols = cols;
    }
}