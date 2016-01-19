package org.ltang.mlbox.optimizer.clg;

import java.util.ArrayList;
import java.util.List;
import org.ltang.mlbox.optimizer.LipschitzConstantGradientLoss;


/**
 * The weighted linear combination of a set of _loss functions
 * @author Liang Tang
 *
 */
public class LinearCombineLoss implements LipschitzConstantGradientLoss {

  List<LipschitzConstantGradientLoss> lossList = new ArrayList<LipschitzConstantGradientLoss>();

  List<Double> lossWeights = new ArrayList<Double>();

  public LinearCombineLoss() {

  }

  public void add(LipschitzConstantGradientLoss loss) {
    add(loss, 1.0);
  }

  public void add(LipschitzConstantGradientLoss loss, double weight) {
    if (lossList.size() > 0) {
      // Check the dimension is consistent or not
      if (lossList.get(0).getDimension() != loss.getDimension()) {
        throw new IllegalArgumentException("The loss function's dimension is not consistent. "
            + "The new loss function dimension is : "+loss.getDimension() +", but the old "
            + "loss function dimension is : "+lossList.get(0).getDimension());
      }
    }
    this.lossList.add(loss);
    this.lossWeights.add(weight);
  }

  @Override
  public double getGradient(final int dimIndex, final double[] beta) {
    double grad = 0;
    for (int lossIndex = 0; lossIndex < lossList.size(); lossIndex++) {
      grad += lossList.get(lossIndex).getGradient(dimIndex, beta) * lossWeights.get(lossIndex);
    }
    return grad;
  }

  @Override
  public double getMaxSecondDerivative(final int dimIndex) {
    double maxSecondDerivative = 0;
    for (int lossIndex = 0; lossIndex < lossList.size(); lossIndex++) {
      maxSecondDerivative += lossList.get(lossIndex).getMaxSecondDerivative(dimIndex) * lossWeights.get(lossIndex);
    }
    return maxSecondDerivative;
  }

  @Override
  public void coefficientUpdate(final int dimIndex, final double delta, final double[] newBeta) {
    for (LipschitzConstantGradientLoss loss : lossList) {
      loss.coefficientUpdate(dimIndex, delta, newBeta);
    }
  }

  @Override
  public double cost(final double[] beta) {
    double cost = 0;
    for (int lossIndex = 0; lossIndex < lossList.size(); lossIndex++) {
      cost += lossList.get(lossIndex).cost(beta) * lossWeights.get(lossIndex);
    }
    return cost;
  }

  @Override
  public int getDimension() {
    return lossList.get(0).getDimension();
  }
}
