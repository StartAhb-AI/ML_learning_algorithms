package com.example.ml;

import smile.clustering.KMeans;
import smile.clustering.CentroidClustering;

public class KMeansExample {
    public static void run() {
        System.out.println("--- K-Means Clustering ---");
        double[][] data = {
                {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0},
                {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}
        };

        int k = 2; // number of clusters

        CentroidClustering<double[], double[]> kmeans = KMeans.fit(data, k);

        // FIX: Replaced kmeans.getClusterLabel() with direct access to the 'y' field
        // This is often required for older/specific versions of SMILE API
        int[] labels = kmeans.y; // <--- Corrected method/field access

        System.out.println("Cluster assignments:");
        for (int i = 0; i < labels.length; i++) {
            System.out.println("Point " + i + " (" + data[i][0] + ", " + data[i][1] + ") -> Cluster " + labels[i]);
        }
    }
}