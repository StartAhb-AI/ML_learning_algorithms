package com.example.ml;

import smile.clustering.HierarchicalClustering;
import smile.clustering.linkage.WardLinkage;
import smile.math.distance.EuclideanDistance;

public class HierarchicalExample {
    public static void run() {
        System.out.println("\n--- Hierarchical Clustering (Ward's Method) ---");
        double[][] data = {
                {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0},
                {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}
        };

        int n = data.length; // 6 data points
        int dim = data[0].length; // 2 features/dimensions
        double[] condensed = new double[n * (n - 1) / 2];
        int index = 0;

        // Compute condensed distance array manually
        EuclideanDistance distance = new EuclideanDistance();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                condensed[index++] = distance.d(data[i], data[j]);
            }
        }

        // FIX: Using the condensed array, total points (n), and DIMENSION (dim=2).
        // For older versions, this signature often expects the dimension.
        WardLinkage linkage = new WardLinkage(condensed, n, dim);

        // Fit hierarchical clustering
        HierarchicalClustering hc = HierarchicalClustering.fit(linkage);

        // Partition into 2 clusters
        int[] labels = hc.partition(2);

        System.out.println("Hierarchical Clustering assignments:");
        for (int i = 0; i < labels.length; i++) {
            System.out.println("Point " + i + " (" + data[i][0] + ", " + data[i][1] + ") -> Cluster " + labels[i]);
        }
    }
}