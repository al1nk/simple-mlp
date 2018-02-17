package perceptron.actfunctions;

public class Tanh implements ActivationFunction {
    @Override
    public float f(float x) {
        return (float) Math.tanh(x);
    }

    @Override
    public float d_f(float x) {
        return (float) (1-Math.pow(f(x), 2));
    }
}
