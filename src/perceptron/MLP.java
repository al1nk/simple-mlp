package perceptron;

import processing.core.PApplet;

public class MLP {

    public static void main(String... args) {

        // On crée une interface minimale avec la bibliothèque Grafica (par Javier Gracia Carpio,
        // https://jagracar.com/grafica.php, GNU Lesser General Public License),
        // qui est une extension de la bibliothèque Processing (https://processing.org/, GNU Lesser General Public License)

        Multicouche mlp = new Multicouche();
        MainMonitor monitor = new MainMonitor(mlp);
        mlp.addObserver(monitor);

        PApplet.runSketch(new String[] {"MLP"}, monitor);
    }
}
