/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aw_cluster;

import java.util.HashSet;
import java.util.Random;
import weka.clusterers.AbstractClusterer;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author aodyra
 */
public class myKMeans extends AbstractClusterer {
    
    private int numCluster = 2;
    private Instances centroid;
    private Instances instances;
    private int[] sizeEachCluster;
    private int[] assignments;
    private int numIteration = 0;
    private int seedRandom;
    private int maxIteration = 500;
    private DistanceFunction distanceFunction = new EuclideanDistance();
   
    public myKMeans(){
        // default seed = 10
        seedRandom = 10;
    }
    
    public myKMeans(int seed){
        seedRandom = 10;
    }
    
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NO_CLASS);
        
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        
        return result;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        
        Instances instances = new Instances(data);
        instances.setClassIndex(-1);
        
        assignments = new int[instances.numInstances()];
        
        distanceFunction.setInstances(instances);
        
        // Initialize Centroid Random From seed
        Random random = new Random(getSeedRandom());
        Instances tempInstances = new Instances(instances);
        
        int tI = tempInstances.numInstances();
        while(tI >= 0 && centroid.numInstances() < numCluster){
            int indexCentroid = random.nextInt(tI + 1);
            centroid.add(tempInstances.get(indexCentroid));
            tempInstances.swap(tI, indexCentroid);
            tI--;
        }
        
        tempInstances = null;
        
        boolean converged = false;
        boolean firstIteration = true;
        while(!converged){
            converged = true;
            for (int i = 0; i < instances.numInstances(); ++i){
                Instance toCluster = instances.instance(i);
                int clusterResult = clusterInstanceTraining(toCluster);
                if (clusterResult != assignments[i]) converged = false;
                assignments[i] = clusterResult;
            }
        }
        
    }
    
    public int clusterInstanceTraining(Instance toCluster){
        double distance;
        double distanceMin = 0;
        int indexCluster = 0;
        for(int i = 0; i < numCluster; ++i){
            distance = distanceFunction.distance(toCluster, centroid.instance(i));
            if(i == 0) {
                distanceMin = distance;
                indexCluster = i;
            }
            if(distance < distanceMin){
                distanceMin = distance;
                indexCluster = i;
            }
        }
        return indexCluster;
    }
    
    public int clusterInstance(Instance instance){
        int cluster = 0;
        return cluster;
    }
    
    public void moveCentroid(){
        
    }
    
    @Override
    public int numberOfClusters() throws Exception {
        return numCluster;
    }
    
    public int[] getAssignment() {
        return assignments;
    }

    public Instances getCentroid() {
        return centroid;
    }

    public int[] getSizeEachCluster() {
        return sizeEachCluster;
    }

    public int getNumIteration() {
        return numIteration;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }
    
    public int getMaxIteration() {
        return maxIteration;
    }
    
    public int getSeedRandom() {
        return seedRandom;
    }
    
    public void setNumCluster(int numCluster) {
        this.numCluster = numCluster;
    }

    public void setDistanceFunction(DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }

    public void setSeedRandom(int seedRandom) {
        this.seedRandom = seedRandom;
    }

    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }
}
