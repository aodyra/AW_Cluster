/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aw_cluster;

import java.util.HashSet;
import java.util.Random;
import weka.clusterers.AbstractClusterer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author aodyra
 */
public class myKMeans extends AbstractClusterer {
    
    private int numCluster = 2;
    private Instances centroid;
    private int[] sizeEachCluster;
    private int[] assignments;
    private int numIteration = 0;
    private int seedRandom;
    private int maxIteration = 500;
    private double[] squaredError;
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
        
        if(instances.numInstances() == 0){
            throw new RuntimeException("Dataset should not be empty");
        }
        
        assignments = new int[instances.numInstances()];
        centroid = new Instances(instances, numCluster);
        distanceFunction.setInstances(instances);
        squaredError = new double[numCluster];
        
        // Initialize Centroid Random From seed
        Random random = new Random(getSeedRandom());
        Instances tempInstances = new Instances(instances);
        
        int tI = tempInstances.numInstances() - 1;
        while(tI >= 0 && centroid.numInstances() < numCluster){
            int indexCentroid = random.nextInt(tI + 1);
            centroid.add(tempInstances.instance(indexCentroid));
            tempInstances.swap(tI, indexCentroid);
            tI--;
        }
        
        tempInstances = null;
        
        boolean converged = false;
        boolean firstIteration = true;
        while(!converged){
            converged = true;
            numIteration++;
            for (int i = 0; i < instances.numInstances(); ++i){
                Instance toCluster = instances.instance(i);
                int clusterResult = clusterInstanceProcess(toCluster, true);
                if (clusterResult != assignments[i]) converged = false;
                assignments[i] = clusterResult;
            }
            
            // update centroid
            Instances[] TempI = new Instances[numCluster];
            centroid = new Instances(instances, numCluster);
            for(int i = 0; i < TempI.length; ++i){
                TempI[i] = new Instances(instances, 0);
            }
            for(int i = 0; i < instances.numInstances(); ++i){
                TempI[assignments[i]].add(instances.instance(i));
            }
            for(int i = 0; i < TempI.length; ++i){
                moveCentroid(TempI[i]);
            }
            if(converged) squaredError = new double[numCluster];
            if(numIteration == maxIteration) converged = true;
            sizeEachCluster = new int[numCluster];
            for(int i = 0; i < numCluster; ++i){
                sizeEachCluster[i] = TempI[i].numInstances();
            }
            
        }
    }

    protected double[] moveCentroid(Instances members){
        double [] vals = new double[members.numAttributes()];

        for (int j = 0; j < members.numAttributes(); j++) {
            vals[j] = members.meanOrMode(j);
        }
        centroid.add(new Instance(1.0, vals));

        return vals;
    }
    
    public int clusterInstanceProcess(Instance toCluster, boolean updateError){
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
        if(updateError){
            distanceMin *= distanceMin;
            squaredError[indexCluster] += distanceMin;
        }
        return indexCluster;
    }
    
    public int clusterInstance(Instance instance){
        return clusterInstanceProcess(instance, false);
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

    public double[] getSquaredError() {
        return squaredError;
    }

    public void setSquaredError(double[] squaredError) {
        this.squaredError = squaredError;
    }
    
    public void setNumCluster(int numCluster) throws Exception{
        if (numCluster <= 0){
            throw new Exception("Number of clusters must be > 0");
        }
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

    @Override
    public String toString() {
        if (centroid == null) {
            return "No clusterer built yet!";
        }

        int maxWidth = 0;
        int maxAttWidth = 0;
        boolean containsNumeric = false;
        for (int i = 0; i < numCluster; i++) {
            for (int j = 0 ;j < centroid.numAttributes(); j++) {
                if (centroid.attribute(j).name().length() > maxAttWidth) {
                    maxAttWidth = centroid.attribute(j).name().length();
                }
                if (centroid.attribute(j).isNumeric()) {
                    containsNumeric = true;
                    double width = Math.log(Math.abs(centroid.instance(i).value(j))) /
                            Math.log(10.0);
                    //          System.err.println(m_ClusterCentroids.instance(i).value(j)+" "+width);
                    if (width < 0) {
                        width = 1;
                    }
                    // decimal + # decimal places + 1
                    width += 6.0;
                    if ((int)width > maxWidth) {
                        maxWidth = (int)width;
                    }
                }
            }
        }

        for (int i = 0; i < centroid.numAttributes(); i++) {
            if (centroid.attribute(i).isNominal()) {
                Attribute a = centroid.attribute(i);
                for (int j = 0; j < centroid.numInstances(); j++) {
                    String val = a.value((int)centroid.instance(j).value(i));
                    if (val.length() > maxWidth) {
                        maxWidth = val.length();
                    }
                }
                for (int j = 0; j < a.numValues(); j++) {
                    String val = a.value(j) + " ";
                    if (val.length() > maxAttWidth) {
                        maxAttWidth = val.length();
                    }
                }
            }
        }

        // check for size of cluster sizes
        for (int i = 0; i < sizeEachCluster.length; i++) {
            String size = "(" + sizeEachCluster[i] + ")";
            if (size.length() > maxWidth) {
                maxWidth = size.length();
            }
        }

        String plusMinus = "+/-";
        maxAttWidth += 2;
        if (maxAttWidth < "Attribute".length() + 2) {
            maxAttWidth = "Attribute".length() + 2;
        }

        if (maxWidth < "Full Data".length()) {
            maxWidth = "Full Data".length() + 1;
        }

        if (maxWidth < "missing".length()) {
            maxWidth = "missing".length() + 1;
        }



        StringBuffer temp = new StringBuffer();
        //    String naString = "N/A";


    /*    for (int i = 0; i < maxWidth+2; i++) {
          naString += " ";
          } */
        temp.append("\nkMeans\n======\n");
        temp.append("\nNumber of iterations: " + numIteration + "\n");

        if(distanceFunction instanceof EuclideanDistance){
            temp.append("Within cluster sum of squared errors: " + Utils.sum(squaredError));
        }else{
            temp.append("Sum of within cluster distances: " + Utils.sum(squaredError));
        }

        temp.append("\n\nCluster centroid:\n");
        temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2)) - "Cluster#".length(), true));

        temp.append("\n");
        temp.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));


//        temp.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

        // cluster numbers
        for (int i = 0; i < numCluster; i++) {
            String clustNum = "" + i;
            temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
        }
        temp.append("\n");

        // cluster sizes
        String cSize = "";
        temp.append(pad(cSize, " ", maxAttWidth - cSize.length(), true));
        for (int i = 0; i < numCluster; i++) {
            cSize = "(" + sizeEachCluster[i] + ")";
            temp.append(pad(cSize, " ",maxWidth + 1 - cSize.length(), true));
        }
        temp.append("\n");

        temp.append(pad("", "=", maxAttWidth +
                (maxWidth * (centroid.numInstances())
                        + centroid.numInstances()), true));
        temp.append("\n");

        for (int i = 0; i < centroid.numAttributes(); i++) {
            String attName = centroid.attribute(i).name();
            temp.append(attName);
            for (int j = 0; j < maxAttWidth - attName.length(); j++) {
                temp.append(" ");
            }

            String strVal;
            String valMeanMode;
            // full data
//            if (centroid.attribute(i).isNominal()) {
//                if (m_FullMeansOrMediansOrModes[i] == -1) { // missing
//                    valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
//                } else {
//                    valMeanMode =
//                            pad((strVal = centroid.attribute(i).value((int)m_FullMeansOrMediansOrModes[i])),
//                                    " ", maxWidth + 1 - strVal.length(), true);
//                }
//            } else {
//                if (Double.isNaN(m_FullMeansOrMediansOrModes[i])) {
//                    valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
//                } else {
//                    valMeanMode =  pad((strVal = Utils.doubleToString(m_FullMeansOrMediansOrModes[i],
//                            maxWidth,4).trim()),
//                            " ", maxWidth + 1 - strVal.length(), true);
//                }
//            }
//            temp.append(valMeanMode);

            for (int j = 0; j < numCluster; j++) {
                if (centroid.attribute(i).isNominal()) {
                    if (centroid.instance(j).isMissing(i)) {
                        valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode =
                                pad((strVal = centroid.attribute(i).value((int)centroid.instance(j).value(i))),
                                        " ", maxWidth + 1 - strVal.length(), true);
                    }
                } else {
                    if (centroid.instance(j).isMissing(i)) {
                        valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode = pad((strVal = Utils.doubleToString(centroid.instance(j).value(i),
                                maxWidth,4).trim()),
                                " ", maxWidth + 1 - strVal.length(), true);
                    }
                }
                temp.append(valMeanMode);
            }
            temp.append("\n");
        }

        temp.append("\n\n");
        return temp.toString();
    }
    
    private String pad(String source, String padChar,
                       int length, boolean leftPad) {
        StringBuffer temp = new StringBuffer();

        if (leftPad) {
            for (int i = 0; i< length; i++) {
                temp.append(padChar);
            }
            temp.append(source);
        } else {
            temp.append(source);
            for (int i = 0; i< length; i++) {
                temp.append(padChar);
            }
        }
        return temp.toString();
    }
}
