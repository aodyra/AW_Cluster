/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aw_cluster;

import java.util.ArrayList;
import java.util.Collections;
import weka.clusterers.AbstractClusterer;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Wiwit Rifa'i
 */
public class myAgnes extends AbstractClusterer
{
    
    private Instances instances;
    private DistanceFunction distanceFunction = new EuclideanDistance();

    public static final int SINGLE_LINKAGE = 1;
    public static final int COMPLETE_LINKAGE = 2;
    private int linkage = 1;

    private Double[][] distanceMatrix;
    private ArrayList< Integer > aliveIndexes;
    private ArrayList< MergePair > mergePairs;

    private int numCluster = 2;
    private int[] assignments;
    private ArrayList< Integer >[] clusteredIndex;

    public class MergePair implements Comparable< MergePair >  {
        int i, j;
        double dist;
        MergePair(int i, int j, double dist) {
            this.i = i;
            this.j = j;
            this.dist = dist;
        }

        @Override
        public int compareTo(MergePair other) {
            double d = this.dist - other.dist;
            if (d < 0) {
                return -1;
            } else if (d > 0) {
                return 1;
            } else {
                return 0;
            }
        }
    }

    public myAgnes(){
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
        
        instances = new Instances(data);

        instances.setClassIndex(-1);
        aliveIndexes = new ArrayList();
        for (int i = 0; i < instances.numInstances(); i++)
            aliveIndexes.add(i);
        mergePairs = new ArrayList();

        distanceFunction.setInstances(instances);

        // Distance Matrix Inititalization
        distanceMatrix = new Double[instances.numInstances()][instances.numInstances()];
        for (int i = 0; i < instances.numInstances(); i++) {
            for (int j = 0; j < instances.numInstances(); j++) {
                distanceMatrix[i][j] = distanceFunction.distance(instances.instance(i), instances.instance(j));
            }
        }
        while (aliveIndexes.size() > 1) {

            // Find Two Nearest Cluster
            MergePair bestPair = new MergePair(-1, -1, 0);
            for (int i = 0; i < aliveIndexes.size(); i++) {
                for (int j = i+1; j < aliveIndexes.size(); j++) {
                    int index_i = aliveIndexes.get(i), index_j = aliveIndexes.get(j);
                    MergePair currentPair = new MergePair(index_i, index_j, distanceMatrix[index_i][index_j]);
                    if (bestPair.i < 0 || bestPair.compareTo(currentPair) > 0)
                        bestPair = currentPair;
                    else if (bestPair.compareTo(currentPair) == 0 && Math.random() < 0.5)
                        bestPair = currentPair; 
                }
            }

            // Merge Two Nearest Cluster
            mergePairs.add(bestPair);
            int index_j = aliveIndexes.indexOf(bestPair.j);
            aliveIndexes.remove(index_j);

            // Update Distance Matrix
            for (int i = 0; i < aliveIndexes.size(); i++) {
                int index = aliveIndexes.get(i);
                if (index == bestPair.i)
                    continue;
                double dist = Math.min(distanceMatrix[index][bestPair.i], distanceMatrix[index][bestPair.j]);
                if (this.linkage == COMPLETE_LINKAGE)
                    dist = Math.max(distanceMatrix[index][bestPair.i], distanceMatrix[index][bestPair.j]);
                distanceMatrix[index][bestPair.i] = dist;
                distanceMatrix[bestPair.i][index] = dist;
            }
        }

        // Construct Cluster
        constuctCluster(numCluster);
    }

    public class DisjoinSetUnion
    {
        private int[] par;
        private int[] set;
        private ArrayList< Integer > [] elements;
        public DisjoinSetUnion(int n) {
            par = new int[n];
            set = new int[n];
            elements = new ArrayList[n];
            for (int i = 0; i < n; i++) {
                par[i] = -1;
                elements[i] = new ArrayList< Integer > ();
                elements[i].add(i);
            }
        }
        public int find(int x) {
            if (par[x] < 0) return x;
            else {
                par[x] = find(par[x]);
                return par[x];
            }
        }
        public boolean merge(int u, int v) {
            u = find(u);
            v = find(v);
            if (u == v)
                return false;
            if (par[u] < par[v]) {
                par[u] += par[v];
                par[v] = u;
                elements[u].addAll(elements[v]);
            }
            else {
                par[v] += par[u];
                par[u] = v;
                elements[v].addAll(elements[u]);
            }
            return true;
        }
        public ArrayList< Integer > getElements(int index) {
            return par[index] < 0 ? elements[index] : null;
        }
    }

    public void constuctCluster(int noCluster) {
        DisjoinSetUnion dsu = new DisjoinSetUnion(instances.numInstances());
        assignments = new int[instances.numInstances()];
        for (int i = 0; i < instances.numInstances() - noCluster; i++) {
            MergePair pair = mergePairs.get(i);
            dsu.merge(pair.i, pair.j);
        }
        clusteredIndex = new ArrayList[noCluster];
        int index = 0;
        for (int i = 0; i < instances.numInstances(); i++)
            if (dsu.find(i) == i)
                clusteredIndex[index++] = dsu.getElements(i); 
        for (int i = 0; i < noCluster; i++) if (clusteredIndex[i] != null) {
            for (Integer e : clusteredIndex[i])
                assignments[e] = i;
        }
    }

    @Override
    public int clusterInstance(Instance instance){
        int cluster = -1;
        double dist = 0;
        for (int i = 0; i < instances.numInstances(); i++) {
            double curDist = distanceFunction.distance(instance, instances.instance(i));
            if (cluster == -1 || dist > curDist) {
                cluster = assignments[i];
                dist = curDist;
            }
        }
        return cluster;
    }
    
    @Override
    public int numberOfClusters() throws Exception {
        return numCluster;
    }
    
    public int[] getAssignment() {
        return assignments;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }
    
    public void setNumCluster(int numCluster) {
        this.numCluster = numCluster;
        if (mergePairs != null)
            constuctCluster(numCluster);
    }

    public void setDistanceFunction(DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }
    public ArrayList< Integer >[] getClusteredIndex() {
        return this.clusteredIndex;
    }
    
    public int getLinkage() {
    	return linkage;
    }

    public void setLinkage(int linkage) {
        this.linkage = linkage;
    }
    
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("myAgnes\n=======\n");
        DisjoinSetUnion dsu = new DisjoinSetUnion(instances.numInstances());
        for (int i = 0; i < instances.numInstances(); i++)
            sb.append("["+i+"]");
        for (int i = 0; i < mergePairs.size(); i++) {
            dsu.merge(mergePairs.get(i).i, mergePairs.get(i).j);
            for (int j = 0; j < instances.numInstances(); j++) if (dsu.getElements(j) != null) {
                ArrayList< Integer > arr = dsu.getElements(j);
                Collections.sort(arr);
                sb.append("[");
                for (int k = 0; k < arr.size(); k++) {
                    if (k > 0)
                        sb.append("  ");
                    sb.append(arr.get(k).toString());
                }
                sb.append("]");
            }
            sb.append("\n");
        }
        sb.append("\n");
        return sb.toString();
    }
}
