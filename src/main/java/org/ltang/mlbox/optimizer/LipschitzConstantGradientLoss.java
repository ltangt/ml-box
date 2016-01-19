package org.ltang.mlbox.optimizer;

/**
 * The abstract class of the _loss function for the coordinate Lipschitz constant gradient algorithm
 * @author Liang Tang
 */
public interface LipschitzConstantGradientLoss extends DifferentiableLoss {

	double getMaxSecondDerivative(final int dimIndex);

}
