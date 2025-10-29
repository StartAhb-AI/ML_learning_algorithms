import java.util.*;

public class KNNExample {

    // Data Point
    static class Point {
        double x, y; // coordinates
        String label; // class label

        Point(double x, double y, String label) {
            this.x = x;
            this.y = y;
            this.label = label;
        }
    }

    // Calculate Euclidean Distance
    public static double distance(Point a, Point b) {
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }

    // KNN Prediction
    public static String predict(List<Point> trainingData, Point testPoint, int k) {
        // Store distances
        List<Map.Entry<Double, String>> distances = new ArrayList<>();
        for (Point train : trainingData) {
            double d = distance(train, testPoint);
            distances.add(new AbstractMap.SimpleEntry<>(d, train.label));
        }

        // Sort by distance
        distances.sort(Comparator.comparingDouble(Map.Entry::getKey));

        // Count votes for k nearest neighbors
        Map<String, Integer> votes = new HashMap<>();
        for (int i = 0; i < k; i++) {
            String label = distances.get(i).getValue();
            votes.put(label, votes.getOrDefault(label, 0) + 1);
        }

        // Return label with max votes
        return Collections.max(votes.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    public static void main(String[] args) {
        // Training dataset
        List<Point> trainingData = new ArrayList<>();
        trainingData.add(new Point(1, 2, "A"));
        trainingData.add(new Point(2, 3, "A"));
        trainingData.add(new Point(3, 3, "A"));
        trainingData.add(new Point(6, 5, "B"));
        trainingData.add(new Point(7, 7, "B"));
        trainingData.add(new Point(8, 6, "B"));

        // Test point
        Point testPoint = new Point(5, 5, "?");

        // Choose k
        int k = 3;
        String prediction = predict(trainingData, testPoint, k);

        System.out.println("Test Point (" + testPoint.x + ", " + testPoint.y + ")");
        System.out.println("Predicted Class: " + prediction);
    }
}
