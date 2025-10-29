public class LogisticRegressionExample {

    // Sigmoid function
    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // Training logistic regression using gradient descent
    public static void main(String[] args) {
        // Example dataset (X = feature, Y = label: 0 or 1)
        double[] X = {0, 1, 2, 3, 4, 5};
        int[] Y = {0, 0, 0, 1, 1, 1};

        int n = X.length;
        double w = 0.0;   // weight
        double b = 0.0;   // bias
        double lr = 0.1;  // learning rate
        int epochs = 1000; // training iterations

        // Gradient Descent
        for (int epoch = 0; epoch < epochs; epoch++) {
            double dw = 0, db = 0;

            for (int i = 0; i < n; i++) {
                double z = w * X[i] + b;
                double pred = sigmoid(z);

                // Gradients
                dw += (pred - Y[i]) * X[i];
                db += (pred - Y[i]);
            }

            // Update weights
            w -= lr * dw / n;
            b -= lr * db / n;
        }

        // Final model parameters
        System.out.println("Trained Weight (w): " + w);
        System.out.println("Trained Bias (b): " + b);

        // Predictions
        System.out.println("\nPredictions:");
        for (int i = 0; i < n; i++) {
            double z = w * X[i] + b;
            double prob = sigmoid(z);
            int prediction = prob >= 0.5 ? 1 : 0;
            System.out.println("X = " + X[i] + " | Actual = " + Y[i] +
                               " | Predicted Prob = " + prob +
                               " | Class = " + prediction);
        }

        // Test new input
        double newX = 2.5;
        double prob = sigmoid(w * newX + b);
        int prediction = prob >= 0.5 ? 1 : 0;
        System.out.println("\nTest Input: X = " + newX +
                           " | Predicted Prob = " + prob +
                           " | Predicted Class = " + prediction);
    }
}
