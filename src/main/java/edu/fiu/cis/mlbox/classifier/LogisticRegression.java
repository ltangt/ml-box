package edu.fiu.cis.mlbox.classifier;

import edu.fiu.cis.mlbox.data.SparseVector;
import edu.fiu.cis.mlbox.optimizer.clg.CoordinateLipschitzGradientOptimizer;
import edu.fiu.cis.mlbox.optimizer.clg.L2RegularizerLoss;
import edu.fiu.cis.mlbox.optimizer.clg.LinearCombineLoss;
import edu.fiu.cis.mlbox.optimizer.clg.LogisticLoss;
import edu.fiu.cis.mlbox.utils.MathFunctions;

/**
 * The implementation of logistic regression
 * @author Liang Tang
 *
 */
public class LogisticRegression {
	
	// The weight for the regularizer
	double lambda = 1.0;
	
	// The trained coefficients
	double[] beta = null;
	
	boolean hasIntercept = true;
	
	public LogisticRegression() {
		this(1.0);
	}
	
	public LogisticRegression(boolean hasIntercept) {
		this(1.0, hasIntercept);
	}
	
	public LogisticRegression(double lambda) {
		this(lambda, false);
	}
	
	public LogisticRegression(double lambda, boolean hasIntercept) {
		this.lambda = lambda;
		this.hasIntercept = hasIntercept;
	}
	
	/**
	 * Train the logistic regression using dense vector data
	 * @param features
	 * @param labels
	 */
	public void train(final double[][] features, final double[] labels) {
		if (features == null || features.length == 0) {
			throw new IllegalArgumentException("The training data is empty!");
		}
		// Convert the dense vectors into sparse representation
		int dimension = features[0].length;
		SparseVector[] sparseFeatures = new SparseVector[features.length];
		for (int instIndex=0; instIndex<features.length; instIndex++) {
			sparseFeatures[instIndex] = new SparseVector(features[instIndex]);
		}
		train(dimension, sparseFeatures, labels);
	}
	
	
	/**
	 * Train the logistic regression using sparse data
	 * @param featureVals
	 * @param featureDims
	 * @param labels
	 */
	public void train(int dimension, final SparseVector[] features, final double[] labels) {
		train(dimension, features, labels, null);
	}
	
	/**
	 * Train the logistic regression using sparse data
	 * @param featureVals
	 * @param featureDims
	 * @param labels
	 */
	public void train(int dimension, final SparseVector[] features, 
			final double[] labels, final double[] weights) {
		// Create the logistic loss function
		LogisticLoss logLoss = new LogisticLoss(dimension, features, 
				labels, weights, hasIntercept);
		// Create the L2 loss
		L2RegularizerLoss l2loss = new L2RegularizerLoss(dimension, hasIntercept);
		// Combine the two loss functions
		LinearCombineLoss loss = new LinearCombineLoss();
		loss.add(logLoss);
		loss.add(l2loss, lambda);
		
		// Create the optimizer
		CoordinateLipschitzGradientOptimizer optimizer = null;
		if (hasIntercept) {
			optimizer =	new CoordinateLipschitzGradientOptimizer(dimension+1, loss);
		}
		else {
			optimizer = new CoordinateLipschitzGradientOptimizer(dimension, loss); 
		}
		// Start the training algorithm to minimize the loss
		optimizer.train();
		beta = optimizer.getCofficients();
	}
	
	public double[] getCoefficients() {
		return beta;
	}
	
	public double predict(final SparseVector feature) {
		if (this.beta == null) {
			throw new IllegalStateException("The coefficients have not been trained!");
		}
		double sum = feature.innerProduct(this.beta);
    	sum = hasIntercept ? (sum+beta[beta.length-1]) : sum;
        return MathFunctions.sigmoid(sum);
	}
	
	public double predict(double[] feature) {
		return predict(new SparseVector(feature));
	}

}
