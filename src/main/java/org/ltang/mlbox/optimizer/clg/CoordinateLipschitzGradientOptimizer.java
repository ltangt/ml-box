package org.ltang.mlbox.optimizer.clg;

import org.ltang.mlbox.utils.MathFunctions;
import org.ltang.mlbox.utils.VectorUtil;


/**
 * The implementation of coordinate Lipschitz constant gradient optimization
 * algorithm
 * 
 * @author Liang Tang
 */
public class CoordinateLipschitzGradientOptimizer {

	final double[] _beta;

	final int _dimension;

	int _max_iter = 500;

	final double[] _maxSecondDerivaties;

	final CoordinateLipschitzGradientLoss _loss;

	transient int DEBUG = 0;

	final static double EPS = 1E-5;

	public CoordinateLipschitzGradientOptimizer(int dimension,
			CoordinateLipschitzGradientLoss loss) {
		this._dimension = dimension;
		this._loss = loss;
		this._beta = new double[dimension];
		this._maxSecondDerivaties = new double[dimension];
		this._loss.coefficientUpdate(0, 0, this._beta);
	}

	public void setDebug(int debug) {
		this.DEBUG = debug;
	}

	public void setMaxNumIteration(int maxIter) {
		this._max_iter = maxIter;
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
		for (int dimIndex = 0; dimIndex < _dimension; dimIndex++) {
			grad = _loss.getGradient(dimIndex);
			maxSecondDerivative = this._maxSecondDerivaties[dimIndex];

			delta = -1.0 / maxSecondDerivative * grad;
			double newBeta = _beta[dimIndex] + delta;
			if (MathFunctions.almostEqual(newBeta, _beta[dimIndex], EPS) == false) {
				nBetaUpdated++;
				_beta[dimIndex] = newBeta;
				_loss.coefficientUpdate(dimIndex, delta, _beta);
			}
		}

		return nBetaUpdated;
	}

	public void train() {
		// Initialize the coefficients and costs for each _loss objective
		
		for (int dimIndex = 0; dimIndex < _dimension; dimIndex++) {
			double maxSecDev = _loss.getMaxSecondDerivative(dimIndex);
			if (maxSecDev < EPS) {
				maxSecDev = EPS;
			}
			this._maxSecondDerivaties[dimIndex] = maxSecDev;
		}

		if (DEBUG >= 1) {
			System.out.println("Initial cost: " + _loss.cost(_beta));
		}

		// Start optimization
		int iter = 0;
		for (iter = 0; iter < _max_iter; iter++) {
			int nCoefficientsUpdated = updateBeta(iter);
			if (nCoefficientsUpdated == 0) {
				break; // all Converged
			}

			/////////////////// debug //////////////////////
			if (DEBUG == 1) {
				if (iter % 2 == 0) {
					System.out.println("iter : " + iter + ",  cost : " + _loss.cost(_beta) + ", #cofficientUpdated: "
							+ nCoefficientsUpdated);
				}
			} else if (DEBUG == 2) {
				System.out.println(
						"iter : " + iter + ",  cost : " + _loss.cost(_beta) + ", #cofficientUpdated: " + nCoefficientsUpdated);
			} else if (DEBUG == 3) {
				System.out.println(
						"iter : " + iter + ", cost : " + _loss.cost(_beta) + ", #cofficientUpdated: " + nCoefficientsUpdated);
				printParameters();
			}
			////////////////////////////////////////////////
		}

		if (DEBUG >= 1) {
			System.out.println("Total #iter : " + iter + ",  final cost : " + _loss.cost(_beta));
			printParameters();
		}
	}
	

	public double[] getCofficients() {
		return this._beta;
	}

	public void printParameters() {		
		System.out.println(VectorUtil.toString(_beta));
	}

}
