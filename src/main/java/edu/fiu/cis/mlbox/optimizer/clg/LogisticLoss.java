package edu.fiu.cis.mlbox.optimizer.clg;

import edu.fiu.cis.mlbox.data.SparseVector;
import edu.fiu.cis.mlbox.utils.MathFunctions;

/**
 * The logistic loss (the negative log-likelihood of logistic regression).
 * @author Liang Tang
 */
public class LogisticLoss extends CoordinateLipschitzGradientLoss {

    final static double EPS = 1E-5;
    
    // Column store index, where each column is a dimension.
    final double[][] colValArrs;
    final int[][] colIndexArrs;

    // the sum of the inner product between the beta_j and x_j
    final double[] innerProducts;

    final int dimension;
    
    final boolean hasIntercept;

    final SparseVector[] features;
        
    final double[] labels;
    
    final double[] weights;

    public LogisticLoss(int dimension, final SparseVector[] features, 
    		final double[] labels, 
    		final double[] weights,
    		final boolean hasIntercept) {
    	this.features = features;
        this.labels = labels;
        this.weights = weights;
        this.hasIntercept = hasIntercept;        
        this.dimension = hasIntercept ? (dimension+1) : dimension;
        this.colIndexArrs = new int[this.dimension][];
        this.colValArrs = new double[this.dimension][];
        
        // Create the cache of the sum of the inner product between the beta_j and x_j 
        this.innerProducts = new double[features.length];
        
        // Check the input training instances
        checkTrainData();
 
        // Create the column based store
        createColumnStore();
    }
    
    private void updateInnerProducts(int dimIndex, double delta) {
        if (Math.abs(delta) < EPS) {
            return;
        }
        int[] colIndices = this.colIndexArrs[dimIndex];
        double[] colValues = this.colValArrs[dimIndex];
        for (int i = 0; i < colIndices.length; i++) {
            int instIndex = colIndices[i];
            double x_j = colValues[i];
            innerProducts[instIndex] += delta * x_j;
        }
    }

    private void checkTrainData() {
        if (features == null || labels == null) {
            throw new IllegalArgumentException("The training data set is empty!");
        }
        
        int numInsts = features.length;
        if (labels.length != numInsts) {
        	throw new IllegalArgumentException("The number of labels is not equal to "
        			+ "the number of data instances!");
        }
        if (features.length != numInsts) {
        	throw new IllegalArgumentException("The number of data feature dimensions is not equal to "
        			+ "the number of data instances!");
        }
        if (weights != null && weights.length != numInsts) {
        	throw new IllegalArgumentException("The number of weights is not equal to "
        			+ "the number of data instances!");
        }
        
        for (int instIndex = 0; instIndex<numInsts; instIndex++) {
        	if (!MathFunctions.almostEqual(labels[instIndex], 0) 
        			&& !MathFunctions.almostEqual(labels[instIndex], 1)) {
        		throw new IllegalArgumentException("The label of the "+instIndex
        				+"th data can only be 0 or 1");
        	}
           
        }
    }

    private void createColumnStore() {
        // First Scan: Count the number of entries for each dimension
        final int[] maxNumEntries = new int[dimension];
        int numInsts = features.length;
        for (int instIndex = 0; instIndex < numInsts; instIndex++) {
            double weight = weights == null ?  1.0 : weights[instIndex];
            if (MathFunctions.almostEqual(weight, 0)) {
                continue;
            }
            int[] dimIndices = features[instIndex].dims;
            for (int j = 0; j < dimIndices.length; j++) {
                int dimIndex = dimIndices[j];
                maxNumEntries[dimIndex]++;
            }
        }
        if (hasIntercept) {
        	maxNumEntries[dimension-1] = numInsts;
        }
        
        // Allocate the memory for the column store
        for (int dim = 0; dim < dimension; dim++) {
        	colIndexArrs[dim] = new int[maxNumEntries[dim]];
        	colValArrs[dim] = new double[maxNumEntries[dim]];
        }
       
        // Second Scan : Build the column store
        final int[] entryIndices = new int[dimension];
        for (int instIndex = 0; instIndex < features.length; instIndex++) {
            double weight = weights == null ?  1.0 : weights[instIndex];
            if (MathFunctions.almostEqual(weight, 0)) {
                continue;
            }
            int[] dimIndices = features[instIndex].dims;
            double[] x = features[instIndex].vals;
            for (int j = 0; j < dimIndices.length; j++) {
                int dimIndex = dimIndices[j];
                int entryIndex = entryIndices[dimIndex];
                colIndexArrs[dimIndex][entryIndex] = instIndex;
                colValArrs[dimIndex][entryIndex] = x[j];
                entryIndices[dimIndex]++;
            }
            if (hasIntercept) {
            	colIndexArrs[dimension-1][instIndex] = instIndex;
                colValArrs[dimension-1][instIndex] = 1;
            }
        }
    }

    /**
     * Get the gradient of the logistic loss (the negative of the log-likelihood)
     *
     * @param dimIndex
     * @return
     */
    @Override
    public double getGradient(int dimIndex) {
        double grad = 0;
        final int[] colIndices = this.colIndexArrs[dimIndex];
        final double[] colValues = this.colValArrs[dimIndex];
        for (int i = 0; i < colIndices.length; i++) {
            int instIndex = colIndices[i];
            double x_j = colValues[i];
            double y = labels[instIndex];
            double weight = weights == null ? 1.0 : weights[instIndex];
            double v = innerProducts[instIndex];
            double pred = MathFunctions.sigmoid(v);
            pred = pred < EPS ? EPS : pred;
            pred = pred > 1 - EPS ? (1 - EPS) : pred;
            grad += -(y - pred) * x_j * weight;
        }
        return grad;
    }

    @Override
    public double getMaxSecondDerivative(int dimIndex) {
        double maxSecondDerivative = 0;
        final int[] colIndices = this.colIndexArrs[dimIndex];
        final double[] colValues = this.colValArrs[dimIndex];
        for (int i = 0; i < colIndices.length; i++) {
            int instIndex = colIndices[i];
            double x_j = colValues[i];
            double weight = weights == null ? 1.0 : weights[instIndex];
            maxSecondDerivative += 0.25 * x_j * x_j * weight;
        }        
        return maxSecondDerivative;
    }

    @Override
    public void coefficientUpdate(int dimIndex, double delta, double[] newBeta) {
        updateInnerProducts(dimIndex, delta);
    }

    @Override
    public double cost(final double[] beta) {
        final double EPS = 1E-6;
        double cost = 0;
        for (int instIndex=0; instIndex<features.length; instIndex++) {
            double y = labels[instIndex];
            double pred = expected(beta, features[instIndex]);
            pred = pred < EPS ? EPS : pred;
            pred = pred > 1 - EPS ? (1 - EPS) : pred;
            double logLikelihood = y * Math.log(pred) + (1 - y) * Math.log(1 - pred);
            double weight = weights == null ? 1.0 : weights[instIndex];
            cost += -logLikelihood * weight;
        }
        return cost;
    }
    
    private double expected(final double[] beta,  final SparseVector feature) {
    	double sum = feature.innerProduct(beta);
    	sum = hasIntercept ? (sum+beta[beta.length-1]) : sum;
        return MathFunctions.sigmoid(sum);
    }

   
}
