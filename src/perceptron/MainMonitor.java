package perceptron;

import processing.core.*;
import grafica.*;

import java.awt.*;
import java.util.Observable;
import java.util.Observer;

public class MainMonitor extends PApplet implements Observer {

    private int epochCounter = 0;

    private Multicouche mlp = null;
    private Thread t;

    private GPlot trAccPlt, trLogprobsPlt, tsAccPlt, tsLogprobsPlt;
    private float trAcc, trLogprob, tsAcc, tsLogprob;
    private GPointsArray trAccPts = new GPointsArray(),
            trLogprobPts = new GPointsArray(),
            tsAccPts = new GPointsArray(),
            tsLogprobPts = new GPointsArray();


    public MainMonitor(Multicouche mlp) {
        this.mlp = mlp;
        this.t = new Thread(mlp, "mlp");
    }

    public void settings(){
        size(1050, 590);
    }

    public void setup(){
        background(240);
        noLoop();

        trAccPlt = new GPlot(this, -15, 0);
        trAccPlt.setBgColor(240);
        trAccPlt.setPointColor(new Color(76, 112, 183).getRGB());
        trAccPlt.setLineColor(new Color(60, 82, 150).getRGB());
        trAccPlt.setDim(450, 200);
        trAccPlt.setYLim(-2, 100);
        trAccPlt.getTitle().setText("Train set");
        trAccPlt.getTitle().setTextAlignment(LEFT);
        trAccPlt.getTitle().setRelativePos(0);
        trAccPlt.getYAxis().getAxisLabel().setText("Taux");
        trAccPlt.getYAxis().getAxisLabel().setTextAlignment(RIGHT);
        trAccPlt.getYAxis().getAxisLabel().setRelativePos(1);

        trLogprobsPlt = new GPlot(this, -15, 270);
        trLogprobsPlt.setBgColor(240);
        trLogprobsPlt.setPointColor(new Color(122, 183, 151).getRGB());
        trLogprobsPlt.setLineColor(new Color(61, 152, 93).getRGB());
        trLogprobsPlt.setDim(450, 200);
        trLogprobsPlt.getYAxis().getAxisLabel().setText("Logporbs");
        trLogprobsPlt.getYAxis().getAxisLabel().setTextAlignment(RIGHT);
        trLogprobsPlt.getYAxis().getAxisLabel().setRelativePos(1);
        trLogprobsPlt.getXAxis().getAxisLabel().setText("Nombre d'époques");

        tsAccPlt = new GPlot(this, 520, 0);
        tsAccPlt.setBgColor(240);
        tsAccPlt.setPointColor(new Color(76, 112, 183).getRGB());
        tsAccPlt.setLineColor(new Color(60, 82, 150).getRGB());
        tsAccPlt.setDim(450, 200);
        tsAccPlt.setYLim( -2, 102);
        tsAccPlt.getTitle().setText("Test set");
        tsAccPlt.getTitle().setTextAlignment(LEFT);
        tsAccPlt.getTitle().setRelativePos(0);
        tsAccPlt.getYAxis().getAxisLabel().setText("Taux");
        tsAccPlt.getYAxis().getAxisLabel().setTextAlignment(RIGHT);
        tsAccPlt.getYAxis().getAxisLabel().setRelativePos(1);

        tsLogprobsPlt = new GPlot(this, 520, 270);
        tsLogprobsPlt.setBgColor(240);
        tsLogprobsPlt.setPointColor(new Color(122, 183, 151).getRGB());
        tsLogprobsPlt.setLineColor(new Color(61, 152, 93).getRGB());
        tsLogprobsPlt.setDim(450, 200);
        tsLogprobsPlt.getYAxis().getAxisLabel().setText("Taux");
        tsLogprobsPlt.getYAxis().getAxisLabel().setTextAlignment(RIGHT);
        tsLogprobsPlt.getYAxis().getAxisLabel().setRelativePos(1);
        tsLogprobsPlt.getXAxis().getAxisLabel().setText("Nombre d'époques");

        t.start();
    }

    public void draw(){

        trAccPts.add(epochCounter/2, trAcc);
        trLogprobPts.add(epochCounter/2, trLogprob);
        tsAccPts.add(epochCounter/2, tsAcc);
        tsLogprobPts.add(epochCounter/2, tsLogprob);

        trAccPlt.setPoints(trAccPts);
        trLogprobsPlt.setPoints(trLogprobPts);
        tsAccPlt.setPoints(tsAccPts);
        tsLogprobsPlt.setPoints(tsLogprobPts);


        //On dessine nos Graphiques

        trAccPlt.defaultDraw();
        trLogprobsPlt.defaultDraw();
        tsAccPlt.defaultDraw();
        tsLogprobsPlt.defaultDraw();
    }

    @Override
    public void update(Observable o, Object arg) {
        if (o == mlp){
            //On met a jour les graphiques
            float[][] data = mlp.getMonitor();

            trAcc = data[0][0]; trLogprob = data[0][1];
            tsAcc = data[1][0]; tsLogprob = data[1][1];
            epochCounter++;
            redraw();
        }
    }
}
