package org.ltang.mlbox.optimizer.clg;

import org.ltang.mlbox.utils.VectorUtil;


/**
 * The L2 regularization _loss
 * @author Liang Tang
 */
public class L2RegularizerLoss extends CoordinateLipschitzGradientLoss {

  int _dimension = -1;

  double[] _priorBeta = null;

  // Whether the last feature is the intercept or not (intercept has no L2 _loss).
  boolean _hasIntercept = true;

  double[] _beta = null;

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

  public L2RegularizerLoss(int dimension, double[] priorBeta, boolean isLastFeatureIntercept) {
    if (priorBeta != null) {
      _priorBeta = VectorUtil.copyNew(priorBeta);
      _dimension = priorBeta.length;
    } else {
      _priorBeta = null;
      _dimension = dimension;
    }
    _beta = new double[dimension];
    _hasIntercept = isLastFeatureIntercept;
  }

  /**
   * Get the gradient of the L2 regularizer _loss
   *
   * @param dimIndex
   * @return
   */
  @Override
  public double getGradient(int dimIndex) {
    if (_hasIntercept == false || dimIndex < _dimension - 1) {
      if (_priorBeta == null) {
        return _beta[dimIndex];
      } else {
        return _beta[dimIndex] - _priorBeta[dimIndex];
      }
    } else {
      return 0;
    }
  }

  @Override
  public double getMaxSecondDerivative(int dimIndex) {
    return 1;
  }

  @Override
  public void coefficientUpdate(int dimIndex, double delta, double[] newBeta) {
    _beta = newBeta;
  }

  @Override
  public double cost(double[] beta) {
    double allCost = VectorUtil.innerProduct(beta, beta) / 2;
    if (!_hasIntercept) {
      return allCost;
    } else {
      return allCost - beta[_dimension - 1] * beta[_dimension - 1] / 2;
    }
  }
}
