package perceptron.datasets;

import mnisttools.MnistReader;
import perceptron.ImageConverter;

import java.util.List;

public class MNISTDataset implements DatasetLoader {

    /* Parametres */
    // les N premiers exemples pour l'apprentissage
    private static int N = 2000;
    // les T derniers exemples  pour l'evaluation
    private static int T = 1000;

    public MNISTDataset(int n, int t){
        N = n;
        T = t;
    }

    @Override
    public void load(List<float[]> trainData, List<Integer> trainRefs,
                     List<float[]> testData, List<Integer> testRefs,
                     boolean oneHot) {

        System.out.println("=== MNIST ===");
        System.out.println("Loading datasets");
		/* Lecteur d'image */
        MnistReader db = new MnistReader(path+"train-labels-idx1-ubyte",
                                        path+"train-images-idx3-ubyte");
		/* Taille des images et donc de l'espace de representation */
        final int SIZEW = ImageConverter.image2VecteurReel_withB(db.getImage(1)).length;


        System.out.println("**Train set");
		/* Creation des donnees */
		/* Donnees d'apprentissage */
        for (int i = 1; i <= N; i++) {
            if (oneHot) trainData.add(ImageConverter.image2VecteurBinaire(db.getImage(i)));
            else trainData.add(ImageConverter.image2VecteurReel_withB(db.getImage(i)));
            trainRefs.add(db.getLabel(i));
        }

        System.out.println("**Test set");
        for (int i = N+1; i <= N+T+1; i++) {
            if (oneHot) testData.add(ImageConverter.image2VecteurBinaire(db.getImage(i)));
            else testData.add(ImageConverter.image2VecteurReel_withB(db.getImage(i)));
            testRefs.add(db.getLabel(i));
        }

        System.out.println("Dataset successfully loaded.");

    }

    @Override
    public int getClassesNumber() {
        return 10;
    }
}
