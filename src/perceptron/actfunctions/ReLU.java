package perceptron.actfunctions;

public class ReLU implements ActivationFunction {
    @Override
    public float f(float x) {
        return (x>0)?x:0;
    }

    @Override
    public float d_f(float x) {
        return (x>0)?1:0;
    }
}
