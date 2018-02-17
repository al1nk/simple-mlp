package perceptron.datasets;

import java.io.IOException;
import java.util.List;

public interface DatasetLoader {

    String path = System.getProperty("user.dir")+"/";

    void load(List<float[]> trainData, List<Integer> trainRefs,
              List<float[]> testData, List<Integer> testRefs,
              boolean oneHot) throws IOException;

    int getClassesNumber();

}
