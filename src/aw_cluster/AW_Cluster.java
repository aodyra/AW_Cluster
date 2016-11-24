/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aw_cluster;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Scanner;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;

/**
 *
 * @author aodyra
 */
public class AW_Cluster {

    public static String masukanFile(Scanner sc){
        int pilihanFile;
        System.out.println("Masukan file latih: ");
        System.out.println("1. iris.arff");
        System.out.println("2. unseen.arff");
        System.out.println("3. weather.nominal.arff");
        System.out.println("4. weather.numeric.arff");
        System.out.print("Pilihan: ");
        pilihanFile = sc.nextInt();
        String path = "resources/";
        switch(pilihanFile){
            case 1: path += "iris.arff";
                    break;
            case 2: path += "unseen.arff";
                    break;
            case 3: path += "weather.nominal.arff";
                    break;
            case 4: path += "weather.numeric.arff";
                    break;
        }
        return path;
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception{
        // TODO code application logic here
        Scanner sc = new Scanner(System.in);
        Instances trainingData;
        ClusterEvaluation eval;
        String path;
        int pilihan;
        int jumlahCluster;
        int maxIter;
        
        do{
            System.out.println("Masukan pilihan algoritma: ");
            System.out.println("1. MyKMeans");
            System.out.println("2. MyAgnes");
            System.out.println("3. Exit");
            System.out.print("Pilihan: ");
            pilihan = sc.nextInt();
            if(pilihan == 1){
                path = masukanFile(sc);
                System.out.println("Masukan jumlah cluster: ");
                jumlahCluster = sc.nextInt();
                System.out.println("Masukan jumlah maksimum iterasi: ");
                maxIter = sc.nextInt();
                BufferedReader data = new BufferedReader(new FileReader(path));
                trainingData = new Instances(data);
                myKMeans kmeans = new myKMeans();
                kmeans.setNumCluster(jumlahCluster);
                kmeans.setMaxIteration(maxIter);
                kmeans.buildClusterer(trainingData);
                eval = new ClusterEvaluation();
                eval.setClusterer(kmeans);
                eval.evaluateClusterer(trainingData);
                System.out.println("Cluster Evaluation: " + eval.clusterResultsToString());
                System.out.println("");
            } else if(pilihan == 2){
                path = masukanFile(sc);
            }
            
        }while(pilihan != 3);
        
        
        
    }
    
}
