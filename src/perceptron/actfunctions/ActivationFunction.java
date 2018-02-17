package perceptron.actfunctions;

public interface ActivationFunction {

    float f(float x);
    default float[] f(float[] x) {
        float[] ret = x.clone();
        for (int i = 0; i < x.length; i++)
            x[i] = f(x[i]);
        return ret;
    }
    default float[][] f(float[][] x) {
        float[][] ret = x.clone();
        for (int i = 0; i < x.length; i++)
            x[i] = f(x[i]);
        return ret;
    }

    float d_f(float x);
    default float[] d_f(float[] x) {
        float[] ret = x.clone();
        for (int i = 0; i < x.length; i++)
            x[i] = d_f(x[i]);
        return ret;
    }
    default float[][] d_f(float[][] x) {
        float[][] ret = x.clone();
        for (int i = 0; i < x.length; i++)
            x[i] = d_f(x[i]);
        return ret;
    }

}
