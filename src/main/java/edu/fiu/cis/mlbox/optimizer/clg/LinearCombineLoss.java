package edu.fiu.cis.mlbox.optimizer.clg;

import java.util.ArrayList;
import java.util.List;

/**
 * The weighted linear combination of a set of loss functions
 * @author Liang Tang
 *
 */
public class LinearCombineLoss extends CoordinateLipschitzGradientLoss {

	List<CoordinateLipschitzGradientLoss> lossList = 
			new ArrayList<CoordinateLipschitzGradientLoss>();
	
	List<Double> lossWeights = new ArrayList<Double>();

	public LinearCombineLoss() {

	}
	
	public void add(CoordinateLipschitzGradientLoss loss) {
		add(loss, 1.0);
	}

	public void add(CoordinateLipschitzGradientLoss loss, double weight) {
		this.lossList.add(loss);
		this.lossWeights.add(weight);
	}

	@Override
	public double getGradient(int dimIndex) {
		double grad = 0;
		for (int lossIndex=0; lossIndex<lossList.size(); lossIndex++) {
			grad += lossList.get(lossIndex).getGradient(dimIndex) * lossWeights.get(lossIndex);
		}
		return grad;
	}

	@Override
	public double getMaxSecondDerivative(int dimIndex) {
		double maxSecondDerivative = 0;
		for (int lossIndex=0; lossIndex<lossList.size(); lossIndex++) {
			maxSecondDerivative += lossList.get(lossIndex).getMaxSecondDerivative(dimIndex)
					* lossWeights.get(lossIndex);
		}
		return maxSecondDerivative;
	}
	
	
	@Override
	public void coefficientUpdate(int dimIndex, double delta, double[] newBeta) {
		for (CoordinateLipschitzGradientLoss loss : lossList) {
			loss.coefficientUpdate(dimIndex, delta, newBeta);
		}
	}

	@Override
	public double cost(double[] beta) {
		double cost = 0;
		for (int lossIndex=0; lossIndex<lossList.size(); lossIndex++) {
			cost += lossList.get(lossIndex).cost(beta) * lossWeights.get(lossIndex);
		}
		return cost;
	}

}
