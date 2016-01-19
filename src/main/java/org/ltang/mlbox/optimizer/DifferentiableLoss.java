package org.ltang.mlbox.optimizer;

/**
 * Created by ltang on 1/18/16.
 */
public interface DifferentiableLoss extends Loss {

  double getGradient(final int dimIndex, final double[] beta);
}
