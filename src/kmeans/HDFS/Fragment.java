package kmeans.HDFS;

import java.io.Serializable;

public class Fragment implements Serializable {

    public float[] features_part = new float[10];

    public Fragment() {
    }

    public float[] getFeatures_part() {
        return features_part;
    }

    public void setFeatures_part(float[] features_part) {
        this.features_part = features_part;
    }

}
