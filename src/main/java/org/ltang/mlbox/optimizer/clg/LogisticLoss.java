package org.ltang.mlbox.optimizer.clg;

import org.ltang.mlbox.data.Instance;
import org.ltang.mlbox.data.SparseVector;
import org.ltang.mlbox.utils.MathFunctions;


/**
 * The logistic _loss (the negative log-likelihood of logistic regression).
 * @author Liang Tang
 */
public class LogisticLoss extends CoordinateLipschitzGradientLoss {

  final static double EPS = 1E-5;

  // Column store index, where each column is a _dimension.
  final double[][] _colValArrs;
  final int[][] _colIndexArrs;

  // the sum of the inner product between the beta_j and x_j
  final double[] _innerProducts;

  // the number of features without considering the intercept term
  final int _dimension;

  final Instance[] _instances;

  public LogisticLoss(int dimension, final Instance[] instances) {
    _instances = instances;
    _dimension = dimension;
    _colIndexArrs = new int[_dimension+1][];
    _colValArrs = new double[_dimension+1][];

    // Create the cache of the sum of the inner product between the beta_j and x_j
    _innerProducts = new double[_instances.length];

    // Check the input training instances
    checkTrainData();

    // Create the column based store
    createColumnStore();
  }

  private void updateInnerProducts(int dimIndex, double delta) {
    if (Math.abs(delta) < EPS) {
      return;
    }
    int[] colIndices = this._colIndexArrs[dimIndex];
    double[] colValues = this._colValArrs[dimIndex];
    for (int i = 0; i < colIndices.length; i++) {
      int instIndex = colIndices[i];
      double x_j = colValues[i];
      _innerProducts[instIndex] += delta * x_j;
    }
  }

  private void checkTrainData() {
    if (_instances == null) {
      throw new IllegalArgumentException("The training data set is empty!");
    }

    int numInsts = _instances.length;
    for (int instIndex = 0; instIndex < numInsts; instIndex++) {
      if (!MathFunctions.almostEqual(_instances[instIndex].getLabel(), 0f)
          && !MathFunctions.almostEqual(_instances[instIndex].getLabel(), 1f)) {
        throw new IllegalArgumentException("The label of the " + instIndex + "th data can only be 0 or 1");
      }
    }
  }

  private void createColumnStore() {
    // First Scan: Count the number of entries for each _dimension
    final int[] maxNumEntries = new int[_dimension];
    int numInsts = _instances.length;
    for (int instIndex = 0; instIndex < numInsts; instIndex++) {
      Instance inst = _instances[instIndex];
      float weight = inst.getWeight();
      SparseVector features = inst.getFeatures();
      if (MathFunctions.almostEqual(weight, 0)) {
        continue;
      }

      int[] dimIndices = features.dims;
      for (int j = 0; j < dimIndices.length; j++) {
        int dimIndex = dimIndices[j];
        maxNumEntries[dimIndex]++;
      }
    }
    maxNumEntries[_dimension - 1] = numInsts;

    // Allocate the memory for the column store
    for (int dim = 0; dim < _dimension; dim++) {
      _colIndexArrs[dim] = new int[maxNumEntries[dim]];
      _colValArrs[dim] = new double[maxNumEntries[dim]];
    }

    // Second Scan : Build the column store
    final int[] entryIndices = new int[_dimension];
    for (int instIndex = 0; instIndex < _instances.length; instIndex++) {
      Instance inst = _instances[instIndex];
      float weight = inst.getWeight();
      SparseVector features = inst.getFeatures();
      if (MathFunctions.almostEqual(weight, 0)) {
        continue;
      }
      int[] dimIndices = features.dims;
      float[] x = features.vals;
      for (int j = 0; j < dimIndices.length; j++) {
        int dimIndex = dimIndices[j];
        int entryIndex = entryIndices[dimIndex];
        _colIndexArrs[dimIndex][entryIndex] = instIndex;
        _colValArrs[dimIndex][entryIndex] = x[j];
        entryIndices[dimIndex]++;
      }
      _colIndexArrs[_dimension - 1][instIndex] = instIndex;
      _colValArrs[_dimension - 1][instIndex] = 1;
    }
  }

  /**
   * Get the gradient of the logistic _loss (the negative of the log-likelihood)
   *
   * @param dimIndex
   * @return
   */
  @Override
  public double getGradient(int dimIndex) {
    double grad = 0;
    final int[] colIndices = this._colIndexArrs[dimIndex];
    final double[] colValues = this._colValArrs[dimIndex];
    for (int i = 0; i < colIndices.length; i++) {
      int instIndex = colIndices[i];
      Instance inst = _instances[instIndex];
      double x_j = colValues[i];
      double y = inst.getLabel();
      double weight = inst.getWeight();
      double v = _innerProducts[instIndex];
      double pred = MathFunctions.sigmoid(v);
      pred = pred < EPS ? EPS : pred;
      pred = pred > 1 - EPS ? (1 - EPS) : pred;
      grad += -(y - pred) * x_j * weight;
    }
    return grad;
  }

  @Override
  public double getMaxSecondDerivative(int dimIndex) {
    double maxSecondDerivative = 0;
    final int[] colIndices = this._colIndexArrs[dimIndex];
    final double[] colValues = this._colValArrs[dimIndex];
    for (int i = 0; i < colIndices.length; i++) {
      int instIndex = colIndices[i];
      Instance inst = _instances[instIndex];
      double x_j = colValues[i];
      double weight = inst.getWeight();
      maxSecondDerivative += 0.25 * x_j * x_j * weight;
    }
    return maxSecondDerivative;
  }

  @Override
  public void coefficientUpdate(int dimIndex, double delta, double[] newBeta) {
    updateInnerProducts(dimIndex, delta);
  }

  @Override
  public double cost(final double[] beta) {
    final double EPS = 1E-6;
    double cost = 0;
    for (int instIndex = 0; instIndex < _instances.length; instIndex++) {
      Instance inst = _instances[instIndex];
      double y = inst.getLabel();
      double pred = expected(beta, inst.getFeatures());
      pred = pred < EPS ? EPS : pred;
      pred = pred > 1 - EPS ? (1 - EPS) : pred;
      double logLikelihood = y * Math.log(pred) + (1 - y) * Math.log(1 - pred);
      double weight = inst.getWeight();
      cost += -logLikelihood * weight;
    }
    return cost;
  }

  private double expected(final double[] beta, final SparseVector feature) {
    double sum = feature.innerProduct(beta);
    sum = sum + beta[beta.length - 1];
    return MathFunctions.sigmoid(sum);
  }
}
