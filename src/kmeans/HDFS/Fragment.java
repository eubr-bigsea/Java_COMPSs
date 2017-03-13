package kmeans.HDFS;

import java.io.Serializable;
import java.util.ArrayList;


public class Fragment implements Serializable {

    public Fragment() {
        Features = new ArrayList<float[]>(1);

    }

    public ArrayList<float[]> Features = new ArrayList<float[]>(1);


    public float[] getFeatureAll(){
        return Features.get(0);
    }


    public void setFeatures(float[] f) {
        Features.add(f);

    }



}
