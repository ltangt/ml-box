package edu.fiu.cis.mlbox.optimizer.clg;

import edu.fiu.cis.mlbox.utils.MathFunctions;
import edu.fiu.cis.mlbox.utils.VectorUtil;

/**
 * The implementation of coordinate Lipschitz constant gradient optimization
 * algorithm
 * 
 * @author Liang Tang
 */
public class CoordinateLipschitzGradientOptimizer {

	final double[] beta;

	final int dimension;

	int MAX_ITER = 500;

	final double[] maxSecondDerivaties;

	final CoordinateLipschitzGradientLoss loss;

	transient int DEBUG = 0;

	final static double EPS = 1E-5;

	public CoordinateLipschitzGradientOptimizer(int dimension,
			CoordinateLipschitzGradientLoss loss) {
		this.dimension = dimension;
		this.loss = loss;
		this.beta = new double[dimension];
		this.maxSecondDerivaties = new double[dimension];
		this.loss.coefficientUpdate(0, 0, this.beta);
	}

	public void setDebug(int debug) {
		this.DEBUG = debug;
	}

	public void setMaxNumIteration(int maxIter) {
		this.MAX_ITER = maxIter;
	}

	/**
	 *
	 * @param iter
	 * @return The number of dimensions been updated
	 */
	private int updateBeta(final int iter) {
		int nBetaUpdated = 0;
		double grad = 0;
		double maxSecondDerivative = 0;
		double delta;
		for (int dimIndex = 0; dimIndex < dimension; dimIndex++) {
			grad = loss.getGradient(dimIndex);
			maxSecondDerivative = this.maxSecondDerivaties[dimIndex];

			delta = -1.0 / maxSecondDerivative * grad;
			double newBeta = beta[dimIndex] + delta;
			if (MathFunctions.almostEqual(newBeta, beta[dimIndex], EPS) == false) {
				nBetaUpdated++;
				beta[dimIndex] = newBeta;
				loss.coefficientUpdate(dimIndex, delta, beta);
			}
		}

		return nBetaUpdated;
	}

	public void train() {
		// Initialize the coefficients and costs for each loss objective
		
		for (int dimIndex = 0; dimIndex < dimension; dimIndex++) {
			double maxSecDev = loss.getMaxSecondDerivative(dimIndex);
			if (maxSecDev < EPS) {
				maxSecDev = EPS;
			}
			this.maxSecondDerivaties[dimIndex] = maxSecDev;
		}

		if (DEBUG >= 1) {
			System.out.println("Initial cost: " + loss.cost(beta));
		}

		// Start optimization
		int iter = 0;
		for (iter = 0; iter < MAX_ITER; iter++) {
			int nCoefficientsUpdated = updateBeta(iter);
			if (nCoefficientsUpdated == 0) {
				break; // all Converged
			}

			/////////////////// debug //////////////////////
			if (DEBUG == 1) {
				if (iter % 2 == 0) {
					System.out.println("iter : " + iter + ",  cost : " + loss.cost(beta) + ", #cofficientUpdated: "
							+ nCoefficientsUpdated);
				}
			} else if (DEBUG == 2) {
				System.out.println(
						"iter : " + iter + ",  cost : " + loss.cost(beta) + ", #cofficientUpdated: " + nCoefficientsUpdated);
			} else if (DEBUG == 3) {
				System.out.println(
						"iter : " + iter + ", cost : " + loss.cost(beta) + ", #cofficientUpdated: " + nCoefficientsUpdated);
				printParameters();
			}
			////////////////////////////////////////////////
		}

		if (DEBUG >= 1) {
			System.out.println("Total #iter : " + iter + ",  final cost : " + loss.cost(beta));
			printParameters();
		}
	}
	

	public double[] getCofficients() {
		return this.beta;
	}

	public void printParameters() {		
		System.out.println(VectorUtil.toString(beta));
	}

}
