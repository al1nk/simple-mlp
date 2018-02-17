package perceptron.datasets;

import caltech.CalTech101;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

public class CaltechDataset implements DatasetLoader  {

    /* Parametres */
    // les N premiers exemples pour l'apprentissage
    private static int N = 2000;
    // les T derniers exemples  pour l'evaluation
    private static int T = 1000;

    public CaltechDataset(int n, int t){
        N = n;
        T = t;
    }

    @Override
    public void load(List<float[]> trainData, List<Integer> trainRefs,
                     List<float[]> testData, List<Integer> testRefs,
                     boolean oneHot) throws IOException {

        System.out.println("=== Caltech Silhouettes ===");
        System.out.println("Loading datasets");
        CalTech101 CT;
        try {
            CT = new CalTech101(path+"caltech101.mat");
        } catch (FileNotFoundException fnf) {
            fnf.printStackTrace();
            throw fnf;
        }


        int img_len = CT.getTestImage(0).length + 1;

        System.out.println("**Train set");
        // Load all train
        for(int i=0; i<N; i++) {
            trainRefs.add(CT.getTrainLabel(i));

            float[] img = new float[img_len];
            img[0] = 1;
            for (int j = 1; j < img_len; j++) {
                if (oneHot) img[j] = (float) CT.getTrainImage(i)[j - 1];
                else img[j] = (float) CT.getTrainImage(i)[j - 1] * 255;
            }
            trainData.add(img);
        }

        System.out.println("**Test set");
        // Load all test
        for(int i=0; i<T; i++) {
            testRefs.add(CT.getTestLabel(i));
            float[] img = new float[img_len];
            img[0] = 1;
            for (int j = 1; j < img_len; j++) {
                if (oneHot) img[j] = (float) CT.getTestImage(i)[j - 1];
                else img[j] = (float) CT.getTestImage(i)[j - 1] * 255;
            }
            testData.add(img);
        }

        System.out.println("Dataset successfully loaded.");

    }

    @Override
    public int getClassesNumber() {
        return 111;
    }
}
