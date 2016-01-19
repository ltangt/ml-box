package org.ltang.mlbox.optimizer;

/**
 * Abstract class for loss function
 * Created by ltang on 1/18/16.
 */
public interface Loss {

  double cost(final double[] beta);

  int getDimension();

  void coefficientUpdate(final int dimIndex, final double delta, final double[] newBeta);

}
