public class LinearRegressionExample {
    public static double mean(double[] arr) {
        double sum = 0;
        for (double v : arr) {
            sum += v;
        }
        return sum / arr.length;
    }

    public static void main(String[] args) {
        double[] X = {1, 2, 3, 4, 5};
        double[] Y = {2, 4, 5, 4, 5};

        int n = X.length;
        double meanX = mean(X);
        double meanY = mean(Y);

        double numerator = 0, denominator = 0;
        for (int i = 0; i < n; i++) {
            numerator += (X[i] - meanX) * (Y[i] - meanY);
            denominator += (X[i] - meanX) * (X[i] - meanX);
        }
        double m = numerator / denominator;  
        double b = meanY - m * meanX;        

        System.out.println("Linear Regression Equation: Y = " + m + "X + " + b);

        double xTest = 6;
        double yPred = m * xTest + b;
        System.out.println("Prediction for X = " + xTest + " : Y = " + yPred);

        System.out.println("\nPredicted values:");
        for (int i = 0; i < n; i++) {
            double yHat = m * X[i] + b;
            System.out.println("X = " + X[i] + ", Actual Y = " + Y[i] + ", Predicted Y = " + yHat);
        }
    }
}
