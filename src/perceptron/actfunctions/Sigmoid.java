package perceptron.actfunctions;

public class Sigmoid implements ActivationFunction {
    @Override
    public float f(float x) {
        return (float) (1.0/(1.0+Math.exp(-x)));
    }

    @Override
    public float d_f(float x) {
        return f(x)*(1-f(x));
    }

}
