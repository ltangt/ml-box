package org.ltang.mlbox.optimizer.clg;

import org.apache.log4j.Logger;
import org.ltang.mlbox.optimizer.LipschitzConstantGradientLoss;
import org.ltang.mlbox.utils.MathFunctions;
import org.ltang.mlbox.utils.VectorUtil;


/**
 * The implementation of coordinate Lipschitz constant gradient optimization
 * algorithm
 *
 * @author Liang Tang
 */
public class CoordinateLipschitzGradientOptimizer {

  private static final Logger log = Logger.getLogger(CoordinateLipschitzGradientOptimizer.class);

  final double[] _beta;

  // The number of features plus 1 (intercept term)
  final int _dimension;

  int _max_iter = 500;

  // norm of the objective's gradient is less than tol times its initial value
  double _tolerance = 1E-4;

  int DEBUG = 0;

  final double[] _maxSecondDerivaties;

  final LipschitzConstantGradientLoss _loss;

  final static double EPS = 1E-7;

  public CoordinateLipschitzGradientOptimizer(LipschitzConstantGradientLoss loss) {
    _dimension = loss.getDimension();
    _loss = loss;
    _beta = new double[_dimension+1];
    _maxSecondDerivaties = new double[_dimension+1];
    _loss.coefficientUpdate(0, 0, _beta);
  }

  public void setDebug(int debug) {
    DEBUG = debug;
  }

  public void setMaxNumIteration(int maxIter) {
    _max_iter = maxIter;
  }

  public void setToleranceForStopCriterion(final double tol) {
    _tolerance = tol;
  }

  /**
   * Update one dimension of the coefficients
   * @return The norm of the gradient
   */
  private double updateBeta() {
    double grad = 0;
    double maxSecondDerivative = 0;
    double delta;
    double gradNorm = 0;
    for (int dimIndex = 0; dimIndex < _dimension+1; dimIndex++) {
      grad = _loss.getGradient(dimIndex, _beta);
      maxSecondDerivative = _maxSecondDerivaties[dimIndex];
      delta = -1.0 / maxSecondDerivative * grad;
      double newBeta = _beta[dimIndex] + delta;
      if (MathFunctions.almostEqual(newBeta, _beta[dimIndex], EPS) == false) {
        _beta[dimIndex] = newBeta;
        _loss.coefficientUpdate(dimIndex, delta, _beta);
      }
      gradNorm += grad*grad;
    }

    return gradNorm;
  }

  public void train() {

    for (int dimIndex = 0; dimIndex < _dimension+1; dimIndex++) {
      double maxSecDev = _loss.getMaxSecondDerivative(dimIndex);
      if (maxSecDev < EPS) {
        maxSecDev = EPS;
      }
      _maxSecondDerivaties[dimIndex] = maxSecDev;
    }

    if (DEBUG >= 1) {
      log.info("Initial cost: " + _loss.cost(_beta));
    }

    // Compute the initial gradient's norm
    double initGradNorm = 0;
    for (int dimIndex = 0; dimIndex < _dimension+1; dimIndex++) {
      double grad = _loss.getGradient(dimIndex, _beta);
      initGradNorm += grad*grad;
    }

    // Start optimization
    int iter;
    for (iter = 0; iter < _max_iter; iter++) {
      double gradNorm = updateBeta();
      if (gradNorm <= initGradNorm * _tolerance) {
        break; // all Converged
      }

      /////////////////// debug //////////////////////
      if (DEBUG == 1) {
        if (iter % 2 == 0) {
          log.info("iter : " + iter + ",  cost : " + _loss.cost(_beta) + ", #gradnorm: " + gradNorm);
        }
      } else if (DEBUG == 2) {
        log.info("iter : " + iter + ",  cost : " + _loss.cost(_beta) + ", #gradnorm: " + gradNorm);
      } else if (DEBUG == 3) {
        log.info("iter : " + iter + ", cost : " + _loss.cost(_beta) + ", #gradNorm: " + gradNorm);
        printParameters();
      }
      ////////////////////////////////////////////////
    }

    if (DEBUG >= 1) {
      log.info("Total #iter : " + iter + ",  final cost : " + _loss.cost(_beta));
      printParameters();
    }
  }

  public double[] getCofficients() {
    return _beta;
  }

  public void printParameters() {
    log.info(VectorUtil.toString(_beta));
  }
}
