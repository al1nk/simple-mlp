package perceptron;

import java.awt.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

import java.util.*;
import java.util.List;
import java.util.Observable;


import mnisttools.MnistManager;

import perceptron.actfunctions.ActivationFunction;
import perceptron.actfunctions.ReLU;
import perceptron.actfunctions.Sigmoid;
import perceptron.actfunctions.Tanh;

import perceptron.datasets.CaltechDataset;
import perceptron.datasets.DatasetLoader;
import perceptron.datasets.MNISTDataset;


public class Multicouche extends Observable implements Runnable {


    //region Paramètres
    /**
     * les N premiers exemples pour l'apprentissage
     */
    private static final int N = 4000;
    /**
     * les T derniers exemples  pour l'evaluation
      */
    private static final int T = 1000;
    /**
     * taille de la couche cachée
      */
    private static final int H = 50;

    /**
     * Nombre d'epoque max
      */
    private final static int EPOCHMAX = 30;
    /**
     * Learning rate
     * Sigmoid works best with 10<lr<0.5
     * ReLU works best with 0.05<lr<0.005
     */
    private static float lr = (float) 0.001;
    /**
     * Coefficient de regularisation
     */
    private static float reg = (float) 0.001;
    /**
     * Taille d'un batch
     * 1 -> Online learning
     * ]1, N[ -> Mini-batch learning
     * N -> Batch learning
     */
    private static int batchSize = 400;
    /**
     * Determine si les données sont binarisées
     * Works best with Sigmoid and Tanh
     */
    private static boolean onehot = false;
    /**
     * Fonction d'activation de la couche cachée
     * Sigmoid works best with one hot data
     * ReLU works best with arbitrarily large data
      */
    private static ActivationFunction actFunc = new ReLU();

    /**
     *  Les donnees
     */
    private static DatasetLoader dataset = new MNISTDataset(N, T);

    /**
     * Observers list
     * Enables plotting
     */
    private List<Observer> observers = new ArrayList<>();


    /**
     * Matrice utilisée pour passer les données a l'observateur/interface graphique
     */
    private float[][] monitor = null;

    //endregion


    public Multicouche(Observer... os) {
        for (Observer o : os)
            this.addObserver(o);
    }

    @Override
    public void run() {
        System.out.println("Start ...");

        List<float[]> trainData = new ArrayList<>();
        List<float[]> testData  = new ArrayList<>();
        List<Integer> trainRefs = new ArrayList<>();
        List<Integer> testRefs  = new ArrayList<>();

        try {
            dataset.load(trainData, trainRefs,
                    testData, testRefs, onehot);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Size of training "+trainData.size());

        List<float[][]> w = initModel(trainData.get(0).length, //Taille des entrées
                H,  //Taille de couche cachée
                dataset.getClassesNumber()); // Nombre de classes

        System.out.println("Number of hidden layers "+w.size());

        fit(trainData, trainRefs, testData, testRefs, w, EPOCHMAX);

        monitor(trainData, trainRefs, testData, testRefs, w, true);
        hasChanged();
        notifyObservers();

        print("==========", "Confusion matrix :");
        printMatrix(getConfusionMatrix(trainData, trainRefs, w));
        print("============", "Accuracy (train) : ", monitor[0][0]+"%");
        print("============", "Accuracy (test) : ", monitor[1][0]+"%");

    }

    public float[][] getMonitor() {
        return monitor;
    }

    //region Observable

    @Override
    public synchronized void addObserver(Observer o) {
        super.addObserver(o);
        this.observers.add(o);
    }

    @Override
    public synchronized void deleteObserver(Observer o) {
        super.deleteObserver(o);
        this.observers.remove(o);
    }

    @Override
    public void notifyObservers() {
        super.notifyObservers();
        notifyObservers(null);
    }

    @Override
    public void notifyObservers(Object arg) {
        super.notifyObservers(arg);
        for (Observer o : observers)
            o.update(this, arg);
    }

    @Override
    public synchronized void deleteObservers() {
        super.deleteObservers();
        for (Observer o : observers)
            deleteObserver(o);
    }

    //endregion


    //region Core
    /**
     * Crée un modèle de regression aléatoirement
     * @param in dimension des entrées
     * @param h dimension de la couche cachée
     * @param out nombre de catégories/dimension de sortie
     * @return {w12, w23}
     */
    private static List<float[][]> initModel(int in, int h, int out) {
        // return all the weigth matrices initialized
        List<float[][]> w = new ArrayList<>(2);

        Random r = new Random();

        w.add(new float[h][in]);
        w.add(new float[out][h]);

        for (float[] line : w.get(0)) {
            for (int i = 0; i < line.length; i++) {
                line[i] = (float) ((r.nextFloat()*2 - 1)*0.01);
            }
        }

        for (float[] line : w.get(1)) {
            for (int i = 0; i < line.length; i++) {
                line[i] = (float) ((r.nextFloat()*2 - 1)*0.01);
            }
        }

        return w;
    }

    /**
     * Entraine le modèle pour un certain nombre d'époques
     * @param trainData données d'entrainement
     * @param trainRefs étiquettes des données
     * @param w modèle a entrainer
     * @param niter nombre d'époques
     */
    private void fit(List<float[]> trainData, List<Integer> trainRefs,
                     List<float[]> testData, List<Integer> testRefs,
                     List<float[][]> w, int niter) {
        // update the w online (image by image) or by batch

        System.out.println("Starting Training Process ...");

//        print("===Initial===", "Accuracy (train) : ", accuracy(trainData, trainRefs, w, false)+"%");
        for (int k = 0; k < niter; k++) {
            for (int i = 0; i < trainData.size() / batchSize; i++) {
                fitBatch(trainData.subList(i*batchSize, (i + 1)*batchSize), //On slice l'ensemble de test
                        trainRefs.subList(i*batchSize, (i + 1)*batchSize), //D'ou le changement de typage d'ArrayList vers List
                        w);
            }

            monitor = monitor(trainData, trainRefs, testData, testRefs, w, false);
            hasChanged();
            notifyObservers();

//            print(k, "============", "Accuracy (train) : ", monitor(trainData, trainRefs, w, false)[0]+"%",
//                    "Logprob (train) : ", monitor(trainData, trainRefs, w, false)[1]);

        }

    }

    /**
     * Entraine un batch de données
     * @param trainData batch de données
     * @param label étiquettes d'entrainement
     * @param w modèle a entrainer
     * @return quantification de l'entrainement (réussite)
     */
    private static void fitBatch(List<float[]> trainData, List<Integer> label, List<float[][]> w) {

        // update the w's for an image or a whole mini-batch
        List<float[][]> z = feedforward(w, trainData);

        int nInst = trainData.size(); //Number of instances/examples shown to the model
        int oDim = w.get(1).length;
        int hDim = w.get(0).length; // = w.get(1)[0].length
        int iDim = w.get(0)[0].length;

        float[][] labels = new float[nInst][oDim];
        for (int i = 0; i < nInst; i++)
            labels[i][label.get(i)] = 1; // Zeros array except where i = label which is equal to 1

        float[][] outputs = z.get(4);
        float[][] hOutputs = z.get(2);
        float[][] hAct = z.get(1);
        float[][] inputs = z.get(0);


        float[][] deltas = ComputeDeltaL3(w.get(1), labels, outputs);
        float[][] dW2 = MAJ_l3(w.get(1), deltas, hOutputs);
        float[][] dW = MAJ_l2(w.get(0), w.get(1), labels, hAct, inputs, deltas);

        //maj w
        for (int i = 0; i < oDim; i++) {
            for (int j = 0; j < hDim; j++) {
                w.get(1)[i][j] += w.get(1)[i][j] * w.get(1)[i][j] * reg;
                w.get(1)[i][j] -= lr * (dW2[i][j] / batchSize);
            }
        }

        //maj w
        for (int i = 0; i < hDim; i++) {
            for (int j = 0; j < iDim; j++) {
                w.get(0)[i][j] += w.get(0)[i][j] * w.get(0)[i][j] * reg;
                w.get(0)[i][j] -= lr * (dW[i][j] / batchSize);
            }
        }
    }


    // Compute neurons ouput for all the minibatch
    /**
     * Forward-pass
     * @param w modèle
     * @param data données
     * @return {[z1=inputs],[u2], [z2], [u3], [z3]}
     */
    private static List<float[][]> feedforward(List<float[][]> w, List<float[]> data) {
        List<float[][]> z = new ArrayList<>();

        float[][] w_1 = w.get(0);
        float[][] w_2 = w.get(1);

        //Preprocessing the data to make it fit float[][] type
        float[][] p_data = new float[data.size()][data.get(0).length];
        for (int i = 0; i < data.size(); i++)
            p_data[i] = data.get(i);
        z.add(p_data);


        float[][] u_2 = dot(p_data, transpose(w_1)); z.add(u_2);
        float[][] z_2 = actFunc.f(u_2); z.add(z_2);

        float[][] u_3 = dot(z_2, transpose(w_2)); z.add(u_3);
        float[][] z_3 = softmax(u_3); z.add(z_3);

        // z will contain u_j and z_j for the two layers
        return z;
    }

    // Compute the term (z_j^(3) - p)
    /**
     * Calcul de l'écart des predictions avec les étiquettes
     * @param w modèle
     * @param label étiquettes
     * @param z_3 prédictions
     * @return {pour i = index d'une donnée : zi - di}
     */
    private static float[][] ComputeDeltaL3(float[][] w, float[][] label, float[][] z_3) {
        float[][] delta = new float[label.length][w.length];

        for (int i = 0; i < delta.length; i++) {
            for (int j = 0; j < delta[i].length; j++) {
                delta[i][j] = z_3[i][j]-label[i][j];
            }
        }
        return delta;
    }

    // MAJ for the last layer : z_i^(2)(z_j^(3) - p)
    /**
     * Calcul des dW23 a retrancher aux w23 pour les mettre a jour
     * @param w modèle
     * @param delta écart entre les predictions et les étiquettes
     * @param z_2 activations de la couche cachée
     * @return somme des dW23 pour les exemples du batch
     */
    private static float[][] MAJ_l3(float[][] w, float[][] delta, float[][] z_2) {

        int oDim = w.length; //Dimension de sortie
        int hDim = w[0].length; //Dimension de la couche cachée
        int nInst = delta.length; //Nombre de données

        float[][] dW2 = new float[oDim][hDim];
        for (int k = 0; k < nInst; k++)
            for (int i = 0; i < oDim; i++)
                for (int j = 0; j < hDim; j++)
                    dW2[i][j] += z_2[k][j]*delta[k][i];


        return dW2;
    }

    // MAJ for the first layer : z_i^(1)f_2'(u_j^(2)) \sum_k wjk^(3) \delta_k^(3)
    /**
     * Calcul des dW12 (gradient a retrancher aux w12)
     * @param w poids entre les deux premères couches
     * @param w2 poids entre les deux dernières couches
     * @param label étiquettes
     * @param u_2 pré-activations de la couche cachée
     * @param z_1 entrées
     * @param delta écart entre les prédictions et les etiquettes
     * @return somme des dW12 pour les exemples du batch
     */
    private static float[][] MAJ_l2(float[][] w, float[][] w2, float[][] label,
                                    float[][] u_2, float[][] z_1, float[][] delta) {

        int i_dim = w[0].length;
        int h_dim = w.length;
        int n_inst = label.length;

        float[][] dHidden = dot(delta, w2);

        for (int i = 0; i < dHidden.length; i++)
            for (int j = 0; j < dHidden[0].length; j++)
                dHidden[i][j] *= actFunc.d_f(u_2[i][j]);

        float[][] dW = new float[h_dim][i_dim];
        for (int k = 0; k < n_inst; k++)
            for (int i = 0; i < h_dim; i++)
                for (int j = 0; j < i_dim; j++)
                    dW[i][j] += z_1[k][j]*dHidden[k][i];


        return dW;
    }

    /**
     * Calcule la matrice de confusion d'un jeu de données
     * @param data jeu de données
     * @param refs étiquettes
     * @param w modèle
     * @return matrice de confusion
     */
    private static int[][] getConfusionMatrix(List<float[]> data, List<Integer> refs,
                                                List<float[][]> w){
        float[][] outputs = feedforward(w, data).get(4);

        int[][] ret = new int[outputs[0].length][outputs[0].length];
        for (int i = 0; i < outputs.length; i++) {
            ret[max(outputs[i])][refs.get(i)]++;
        }
        return ret;
    }


    /**
     * Estime la précision des prédictions sur un dataset donné en calculant le taux de bonnes prédictions,
     * puis les logprobabilités
     * Ecrit les images non reconnues dans des fichiers png dans le repertoire imgs/
     * @param data donées avec lesquelles estimer la précision
     * @param labels étiquettes des données
     * @param w modèle a estimer
     * @param prints implrimer les images
     * @return {taux de predictions justes, logprobs}
     */
    private float[] monitor(List<float[]> data, List<Integer> labels, List<float[][]> w,
                                  boolean prints){

        List<float[][]> ouputs = feedforward(w, data);
        float[][] predictions = ouputs.get(4);
        List<float[]> wrongs = new ArrayList<>();

        float fitness = 0;
        float logprob = 0;

        for (int i = 0; i < data.size(); i++){
            if (max(predictions[i]) == labels.get(i))
                fitness++;
            else
                wrongs.add(data.get(i));

            logprob -= Math.log(ouputs.get(4)[i][labels.get(i)]);
        }

        if (prints)
            try {
                printImgPNG("mal_classe", wrongs);
            } catch (IOException ioe) {
                ioe.getStackTrace();
            }

        return new float[] {(fitness/data.size())*100, logprob/data.size()};
    }
    private float[][] monitor(List<float[]> trainData, List<Integer> trainLabels,
                              List<float[]> testData, List<Integer> testLabels,
                              List<float[][]> w, boolean prints)
    { return new float[][]{monitor(trainData, trainLabels, w, prints),
                            monitor(testData, testLabels, w, prints)};
    }


    //endregion



    //region Tools
    /**
     * Array summation
     * @param values array of values to sum together
     * @return sum
     */
    public static float sum(float[] values){
        float sum = 0;
        for(float v : values)
            sum+=v;
        return sum;
    }

    /**
     * Maximum function
     * @param values array from which we should take max
     * @return index of the biggest element
     */
    private static int max(float[] values){

        int maxIndex = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[maxIndex] < values[i])   maxIndex = i;
        }
        return maxIndex;
    }

    /**
     * Maximum function
     * @param values list from which we should take max
     * @return index of the biggest element
     */
    private static int max(List<Integer> values){

        int maxIndex = 0;
        for (int i = 1; i < values.size(); i++) {
            if (values.get(maxIndex) < values.get(i)) maxIndex = i;
        }
        return maxIndex;
    }

    /**
     * Softmax function
     * @param x input array
     * @return normalized probabilities
     */
    private static float[] softmax(float[] x){
        // Here, we are bothering with finding the biggest element of the layer and subbing it inside the exponential
        // to prevent computing huge values (and eventually overflowing)
        float[] softened = new float[x.length];
        float biggest = x[max(x)];
        float sum = 0;
        for (float v : x) {
            sum += Math.exp(v-biggest);
        }

        for (int i = 0; i < softened.length; i++) {
            softened[i] = (float) ((Math.exp(x[i]-biggest))/sum);
        }
        return softened;
    }

    /**
     * Softmax function (2D)
     * @param x input matrix
     * @return normalized probabilities
     */
    private static float[][] softmax(float[][] x){
        float[][] ret = x.clone();
        for (int i = 0; i < ret.length; i++) {
            ret[i] = softmax(ret[i]);
        }
        return ret;
    }

    /**
     * Vector dot product
     * @param x first vector
     * @param y second vector
     * @return dot product (scalar)
     */
    public static float dot(float[] x, float[] y) {
        if (x.length != y.length)
            throw new IllegalArgumentException("Matrix product failed : format mismatch (#(x) = " + x.length +
                                                ", #(y) = " + y.length + ")");
        float res = 0;
        for(int i=0; i<x.length; i++)
            res += x[i]*y[i];
        return res;
    }

    /**
     * Matrix dot product
     * @param u first matrix
     * @param v second matrix
     * @return dot product (matrix)
     */
    private static float[][] dot(float[][] u, float[][] v){

        int ux = u.length, uy = u[0].length, vx = v.length, vy = v[0].length;
        if (uy != vx)
            throw new IllegalArgumentException("Matrix product failed : format mismatch (u_y = "+ uy +", v_x = "+vx+")");

        float[][] ret = new float[ux][vy];

        for (int i = 0; i < ret.length; i++)
            for (int j = 0; j < ret[0].length; j++)
                ret[i][j] = 0;


        for (int i = 0; i < ux; i++) {
            for (int j = 0; j < vy; j++) {
                for (int k = 0; k < uy; k++) {
                    ret[i][j] += u[i][k] * v[k][j];
                }
            }
        }


        return ret;
    }

    /**
     * Matrix transposition
     * @param x matrix
     * @return x^T
     */
    private static float[][] transpose(float[][] x){

        float[][] ret = new float[x[0].length][x.length];
        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x[0].length; j++)
                ret[j][i] = x[i][j];
        return ret;
    }


    //Logging

    /**
     * Print an arbitrary set of printable objects
     * @param objs objects
     */
    private static void print(Object... objs){
        for (Object o : objs)
            System.out.println(o.toString());
        System.out.println("\n");
    }

    /**
     * Print dimensions of a matrix
     * @param name label to display
     * @param x matrix
     */
    public static void printProfit(String name, float[][] x){
        System.out.println(name+" ("+x.length+ ", "+x[0].length+")");
    }

    /**
     * Save images to file (PPM)
     * @param filename filename
     * @param imgs images
     * @throws IOException exception thrown when writing the file failed
     */
    public static void printImgPPM(String filename, float[]... imgs) throws IOException {
        for (int k = 0 ; k < imgs.length ; k++){
            int[][] temp = new int[28][28];
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    temp[i][j] = (int) imgs[k][28*i+j];
                }
            }
            MnistManager.writeImageToPpm(temp, filename+"("+k+").ppm");
        }
    }

    /**
     * Save images to file (PNG)
     * @param filename filename
     * @param imgs images
     * @throws IOException exception thrown when writing the file failed
     */
    private static void printImgPNG(String filename, List<float[]> imgs) throws IOException {
        File folder = new File( System.getProperty("user.dir")+"/imgs/");
        folder.mkdir();
        File[] content = folder.listFiles();
        if (content != null) for (File file : content) file.delete();

        for (int k = 0; k < imgs.size(); k++) {
            float[] img = imgs.get(k);
            int width = 28, height = 28;

            int[] vimage = new int[width*height];
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    float pxl = img[i*width + j];
                    if (!onehot)
                        pxl /= 255;
                    vimage[i*width + j] = new Color(pxl, pxl, pxl).getRGB();
                }
            }

            BufferedImage bimage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            bimage.setRGB(0, 0, width, height, vimage, 0, width);

            File outputfile = new File(folder, filename+"("+k+").png");
            try {
                ImageIO.write(bimage, "png", outputfile);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }

    /**
     * Display a matrix
     * @param x matrix
     */
    private static void printMatrix(float[][] x){
        for (float[] line : x)
            System.out.println(Arrays.toString(line));
        System.out.println("\n");
    }
    private static void printMatrix(int[][] x){
        for (int[] line : x)
            System.out.println(Arrays.toString(line));
        System.out.println("\n");
    }


    /**
     * Display an arbitrary number of matrices
     * @param x matrices
     */
    public static void printMatrix(float[][]... x) {
        for (float[][] matrix : x) {
            printMatrix(matrix);
            System.out.println("\n");
        }
    }


    //endregion

}