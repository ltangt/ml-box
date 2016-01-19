package org.ltang.mlbox.optimizer.clg;

import org.ltang.mlbox.optimizer.LipschitzConstantGradientLoss;
import org.ltang.mlbox.utils.VectorUtil;


/**
 * The L2 regularization _loss
 * @author Liang Tang
 */
public class L2RegularizerLoss implements LipschitzConstantGradientLoss {

  int _dimension = -1;

  // Has _dimension+1 elements, and the last dimension is the intercept
  double[] _priorBeta = null;

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
  }

  /**
   * Get the gradient of the L2 regularizer _loss
   *
   * @param dimIndex
   * @return
   */
  @Override
  public double getGradient(final int dimIndex, final double[] beta) {
    if (dimIndex < _dimension) {
      if (_priorBeta == null) {
        return beta[dimIndex];
      }
      else {
        return beta[dimIndex] - _priorBeta[dimIndex];
      }
    }
    else {
      return 0;
    }
  }

  @Override
  public double getMaxSecondDerivative(final int dimIndex) {
    return 1;
  }

  @Override
  public double cost(final double[] beta) {
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

  @Override
  public int getDimension() {
    return _dimension;
  }

  @Override
  public void coefficientUpdate(int dimIndex, double delta, double[] newBeta) {

  }
}
