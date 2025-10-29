import java.util.*;

public class RandomForestExample {

    // ----- Decision Tree Node -----
    static class Node {
        String feature;
        String label;
        Map<String, Node> children = new HashMap<>();
    }

    // ----- Entropy -----
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

    // ----- Majority Vote -----
    public static String majorityLabel(List<String> labels) {
        Map<String, Integer> freq = new HashMap<>();
        for (String label : labels) {
            freq.put(label, freq.getOrDefault(label, 0) + 1);
        }
        return Collections.max(freq.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    // ----- Build Tree (ID3 algorithm) -----
    public static Node buildTree(List<Map<String, String>> data, List<String> features, List<String> labels) {
        Node node = new Node();

        // All labels same → leaf
        Set<String> uniqueLabels = new HashSet<>(labels);
        if (uniqueLabels.size() == 1) {
            node.label = labels.get(0);
            return node;
        }

        // No features left → majority
        if (features.isEmpty()) {
            node.label = majorityLabel(labels);
            return node;
        }

        // Find best feature
        String bestFeature = null;
        double bestGain = -1;
        double baseEntropy = entropy(labels);

        for (String feature : features) {
            Map<String, List<String>> partitions = new HashMap<>();
            for (int i = 0; i < data.size(); i++) {
                String value = data.get(i).get(feature);
                partitions.putIfAbsent(value, new ArrayList<>());
                partitions.get(value).add(labels.get(i));
            }

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

        // Remaining features
        List<String> remainingFeatures = new ArrayList<>(features);
        remainingFeatures.remove(bestFeature);

        // Partition data by best feature
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

    // ----- Predict -----
    public static String predict(Node tree, Map<String, String> sample) {
        if (tree.label != null) return tree.label;
        String value = sample.get(tree.feature);
        Node child = tree.children.get(value);
        if (child == null) return "Unknown";
        return predict(child, sample);
    }

    // ----- Random Forest -----
    static class RandomForest {
        List<Node> trees = new ArrayList<>();
        int numTrees;

        RandomForest(int numTrees) {
            this.numTrees = numTrees;
        }

        void train(List<Map<String, String>> data, List<String> features, List<String> labels) {
            Random rand = new Random();
            for (int t = 0; t < numTrees; t++) {
                // Bootstrap sample
                List<Map<String, String>> sampleData = new ArrayList<>();
                List<String> sampleLabels = new ArrayList<>();
                for (int i = 0; i < data.size(); i++) {
                    int idx = rand.nextInt(data.size());
                    sampleData.add(data.get(idx));
                    sampleLabels.add(labels.get(idx));
                }
                Node tree = buildTree(sampleData, features, sampleLabels);
                trees.add(tree);
            }
        }

        String predict(Map<String, String> sample) {
            List<String> votes = new ArrayList<>();
            for (Node tree : trees) {
                votes.add(predict(tree, sample));
            }
            return majorityLabel(votes);
        }
    }

    // ----- Main -----
    public static void main(String[] args) {
        // Simple dataset
        List<Map<String, String>> data = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        data.add(Map.of("Outlook", "Sunny", "Temp", "Hot")); labels.add("No");
        data.add(Map.of("Outlook", "Sunny", "Temp", "Mild")); labels.add("No");
        data.add(Map.of("Outlook", "Overcast", "Temp", "Hot")); labels.add("Yes");
        data.add(Map.of("Outlook", "Rainy", "Temp", "Mild")); labels.add("Yes");
        data.add(Map.of("Outlook", "Rainy", "Temp", "Cool")); labels.add("Yes");

        List<String> features = new ArrayList<>(Arrays.asList("Outlook", "Temp"));

        // Train Random Forest
        RandomForest rf = new RandomForest(3); // 3 trees
        rf.train(data, features, labels);

        // Test sample
        Map<String, String> testSample = Map.of("Outlook", "Sunny", "Temp", "Cool");
        String prediction = rf.predict(testSample);

        System.out.println("Random Forest Prediction: " + prediction);
    }
}
