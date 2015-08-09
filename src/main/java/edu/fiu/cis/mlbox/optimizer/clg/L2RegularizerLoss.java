package edu.fiu.cis.mlbox.optimizer.clg;

import edu.fiu.cis.mlbox.utils.VectorUtil;

/**
 * The L2 regularization loss
 * @author Liang Tang
 */
public class L2RegularizerLoss extends CoordinateLipschitzGradientLoss {

    final static double EPS = 1E-5;

    int dimension = -1;

    double[] priorBeta = null;
    
    // Whether the last feature is the intercept or not (intercept has no L2 loss).
    boolean hasIntercept = true;

    transient double[] beta = null;
    
    public L2RegularizerLoss(int dimension) {
    	this(dimension, null, false);
    }

    public L2RegularizerLoss(double[] priorBeta) {
    	this(priorBeta.length, priorBeta, false);
    }
    
    public L2RegularizerLoss(int dimension, boolean isLastFeatureIntercept) {
    	this(dimension, null, isLastFeatureIntercept);
    }

    public L2RegularizerLoss(double[] priorBeta, boolean isLastFeatureIntercept) {
    	this(priorBeta.length, priorBeta, isLastFeatureIntercept);
    }
    
    public L2RegularizerLoss(int dimension, double[] priorBeta, 
    		boolean isLastFeatureIntercept) {
    	if (priorBeta != null) {
    		this.priorBeta = VectorUtil.copyNew(priorBeta);
        	this.dimension = priorBeta.length;
    	}
    	else {
    		this.priorBeta = null;
    		this.dimension = dimension;
    	}
    	this.beta = new double[dimension];
    	this.hasIntercept = isLastFeatureIntercept;
    }
    
    
    /**
     * Get the gradient of the L2 regularizer loss
     *
     * @param dimIndex
     * @return
     */
    @Override
    public double getGradient(int dimIndex) {
    	if (hasIntercept == false || dimIndex < dimension-1) {
	    	if (priorBeta == null) {
	            return beta[dimIndex];
	        } else {
	            return beta[dimIndex] - priorBeta[dimIndex];
	        }
    	}
    	else {
    		return 0;
    	}
    }

    @Override
    public double getMaxSecondDerivative(int dimIndex) {
    	return 1;
    }

    @Override
    public void coefficientUpdate(int dimIndex, double delta, double[] newBeta) {
        this.beta = newBeta;
    }

	@Override
	public double cost(double[] beta) {
		double allCost = VectorUtil.innerProduct(beta, beta) / 2;
		if (!hasIntercept) {
			return allCost;
		}
		else {
			return allCost - beta[dimension-1]*beta[dimension-1]/2;
		}
	}
	
}
