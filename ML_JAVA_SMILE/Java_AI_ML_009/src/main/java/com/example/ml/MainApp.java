package com.example.ml;

public class MainApp {
    public static void main(String[] args) {
        System.out.println("=========================================");
        System.out.println("Java ML Unsupervised Tasks Demo");
        System.out.println("=========================================");

        KMeansExample.run();

        HierarchicalExample.run();

        PCAExample.run();

        System.out.println("=========================================");
    }
}