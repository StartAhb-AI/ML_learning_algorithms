import java.util.*;

public class DecisionTreeExample {

    // Node of the Decision Tree
    static class Node {
        String feature;
        String label;
        Map<String, Node> children = new HashMap<>();
    }

    // Calculate Entropy
    public static double entropy(List<String> labels) {
        Map<String, Integer> freq = new HashMap<>();
        for (String label : labels) {
            freq.put(label, freq.getOrDefault(label, 0) + 1);
        }

        double entropy = 0.0;
        int total = labels.size();
        for (int count : freq.values()) {
            double p = (double) count / total;
            entropy -= p * (Math.log(p) / Math.log(2));
        }
        return entropy;
    }

    // Build Decision Tree using ID3
    public static Node buildTree(List<Map<String, String>> data, List<String> features, List<String> labels) {
        Node node = new Node();

        // If all labels are same → leaf node
        Set<String> uniqueLabels = new HashSet<>(labels);
        if (uniqueLabels.size() == 1) {
            node.label = labels.get(0);
            return node;
        }

        // If no features left → majority class
        if (features.isEmpty()) {
            node.label = majorityLabel(labels);
            return node;
        }

        // Find best feature using information gain
        String bestFeature = null;
        double bestGain = -1;
        double baseEntropy = entropy(labels);

        for (String feature : features) {
            // Partition data by feature value
            Map<String, List<String>> partitions = new HashMap<>();
            for (int i = 0; i < data.size(); i++) {
                String value = data.get(i).get(feature);
                partitions.putIfAbsent(value, new ArrayList<>());
                partitions.get(value).add(labels.get(i));
            }

            // Calculate weighted entropy
            double newEntropy = 0.0;
            for (List<String> subset : partitions.values()) {
                double weight = (double) subset.size() / data.size();
                newEntropy += weight * entropy(subset);
            }

            double gain = baseEntropy - newEntropy;
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = feature;
            }
        }

        node.feature = bestFeature;

        // Remove used feature
        List<String> remainingFeatures = new ArrayList<>(features);
        remainingFeatures.remove(bestFeature);

        // Create branches
        Map<String, List<Map<String, String>>> newData = new HashMap<>();
        Map<String, List<String>> newLabels = new HashMap<>();

        for (int i = 0; i < data.size(); i++) {
            String value = data.get(i).get(bestFeature);
            newData.putIfAbsent(value, new ArrayList<>());
            newLabels.putIfAbsent(value, new ArrayList<>());
            newData.get(value).add(data.get(i));
            newLabels.get(value).add(labels.get(i));
        }

        for (String value : newData.keySet()) {
            node.children.put(value, buildTree(newData.get(value), remainingFeatures, newLabels.get(value)));
        }

        return node;
    }

    // Majority vote
    public static String majorityLabel(List<String> labels) {
        Map<String, Integer> freq = new HashMap<>();
        for (String label : labels) {
            freq.put(label, freq.getOrDefault(label, 0) + 1);
        }
        return Collections.max(freq.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    // Prediction
    public static String predict(Node tree, Map<String, String> sample) {
        if (tree.label != null) {
            return tree.label;
        }
        String value = sample.get(tree.feature);
        Node child = tree.children.get(value);
        if (child == null) return "Unknown";
        return predict(child, sample);
    }

    public static void main(String[] args) {
        // Example dataset
        // Weather dataset: Outlook, Temperature, Play
        List<Map<String, String>> data = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        data.add(Map.of("Outlook", "Sunny", "Temp", "Hot")); labels.add("No");
        data.add(Map.of("Outlook", "Sunny", "Temp", "Mild")); labels.add("No");
        data.add(Map.of("Outlook", "Overcast", "Temp", "Hot")); labels.add("Yes");
        data.add(Map.of("Outlook", "Rainy", "Temp", "Mild")); labels.add("Yes");
        data.add(Map.of("Outlook", "Rainy", "Temp", "Cool")); labels.add("Yes");

        List<String> features = new ArrayList<>(Arrays.asList("Outlook", "Temp"));

        // Train decision tree
        Node tree = buildTree(data, features, labels);

        // Test sample
        Map<String, String> testSample = Map.of("Outlook", "Sunny", "Temp", "Cool");
        String prediction = predict(tree, testSample);
        System.out.println("Prediction for test sample: " + prediction);
    }
}
