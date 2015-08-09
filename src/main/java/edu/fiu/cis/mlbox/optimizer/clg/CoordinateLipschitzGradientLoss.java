package edu.fiu.cis.mlbox.optimizer.clg;

/**
 * The abstract class of the loss function for the coordinate Lipschitz constant gradient algorithm
 * @author Liang Tang
 */
public abstract class CoordinateLipschitzGradientLoss {

	public CoordinateLipschitzGradientLoss() {
	}

	abstract public double cost(double[] beta);

	abstract public double getGradient(int dimIndex);

	abstract public double getMaxSecondDerivative(int dimIndex);
	
	public void coefficientUpdate(int dimIndex, double delta, double[] newBeta) {
    }

}
