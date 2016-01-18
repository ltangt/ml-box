package org.ltang.mlbox.optimizer.clg;

import org.ltang.mlbox.utils.VectorUtil;


/**
 * The L2 regularization _loss
 * @author Liang Tang
 */
public class L2RegularizerLoss extends CoordinateLipschitzGradientLoss {

  int _dimension = -1;

  // Has _dimension+1 elements, and the last dimension is the intercept
  double[] _priorBeta = null;

  // Has _dimension+1 elements, and the last dimension is the intercept
  double[] _beta = null;

  public L2RegularizerLoss(int dimension) {
    this(dimension, null);
  }

  public L2RegularizerLoss(double[] priorBeta) {
    this(priorBeta.length, priorBeta);
  }


  public L2RegularizerLoss(int dimension, double[] priorBeta) {
    if (priorBeta != null) {
      _priorBeta = VectorUtil.copyNew(priorBeta);
      _dimension = priorBeta.length -1;
    } else {
      _priorBeta = null;
      _dimension = dimension;
    }
    _beta = new double[dimension + 1];
  }

  /**
   * Get the gradient of the L2 regularizer _loss
   *
   * @param dimIndex
   * @return
   */
  @Override
  public double getGradient(int dimIndex) {
    if (dimIndex < _dimension) {
      if (_priorBeta == null) {
        return _beta[dimIndex];
      }
      else {
        return _beta[dimIndex] - _priorBeta[dimIndex];
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
    _beta = newBeta;
  }

  @Override
  public double cost(double[] beta) {
    if (_priorBeta == null) {
      double allCost = VectorUtil.innerProduct(beta, beta) / 2;
      return allCost - beta[_dimension] * beta[_dimension] / 2;
    }
    else {
      double[] diffVec = VectorUtil.minus(beta, _priorBeta);
      double allCost = VectorUtil.innerProduct(diffVec, diffVec) / 2;
      return allCost - diffVec[_dimension] * diffVec[_dimension] / 2;
    }
  }
}
