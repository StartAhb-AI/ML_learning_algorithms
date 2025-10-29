package com.example.ml;

import smile.projection.PCA;
import java.util.Arrays;

public class PCAExample {
    public static void run() {
        System.out.println("\n--- Principal Component Analysis (PCA) ---");
        double[][] data = {
                {2.5, 2.4}, {0.5, 0.7}, {2.2, 2.9},
                {1.9, 2.2}, {3.1, 3.0}, {2.3, 2.7},
                {2, 1.6}, {1, 1.1}, {1.5, 1.6}, {1.1, 0.9}
        };

        // Fit PCA
        PCA pca = PCA.fit(data);

        // FIX 1: Set the target dimension for projection (k=1)
        pca.setProjection(1);

        // FIX 2: Transform the data using the new projection
        double[][] transformed = pca.project(data);

        System.out.println("PCA transformed data (1D):");
        for (int i = 0; i < transformed.length; i++) {
            System.out.println("Point " + i + ": " + transformed[i][0]);
        }

        double cumulativeVariance = pca.getCumulativeVarianceProportion()[0];
        System.out.printf("Variance Explained by 1st Principal Component: %.4f%n", cumulativeVariance);
    }
}