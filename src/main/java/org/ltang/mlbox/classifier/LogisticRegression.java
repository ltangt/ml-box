package org.ltang.mlbox.classifier;

import org.ltang.mlbox.data.Instance;
import org.ltang.mlbox.data.SparseVector;
import org.ltang.mlbox.optimizer.clg.CoordinateLipschitzGradientOptimizer;
import org.ltang.mlbox.optimizer.clg.L2RegularizerLoss;
import org.ltang.mlbox.optimizer.clg.LinearCombineLoss;
import org.ltang.mlbox.optimizer.clg.LogisticLoss;
import org.ltang.mlbox.utils.MathFunctions;


/**
 * The implementation of logistic regression
 * @author Liang Tang
 *
 */
public class LogisticRegression {

  // The weight for the regularizer
  double _lambda = 1.0;

  // The trained coefficients
  double[] _beta = null;

  // The prior model coefficients
  double[] _prior = null;

  // The maximum iteration for the optimization algorithm
  int _maxIter = -1;

  // debug mode
  int _debug = 0;

  public LogisticRegression() {
    this(1.0);
  }

  public LogisticRegression(double lambda) {
    this._lambda = lambda;
  }

  public LogisticRegression(final double lambda, final int maxIter) {
    this._lambda = lambda;
    this._maxIter = maxIter;
  }

  public void setPrior(final double[] priorBeta) {
    _prior = new double[priorBeta.length];
    System.arraycopy(priorBeta, 0, _prior, 0, priorBeta.length);
  }

  public void setDebug(final int debug) {
    _debug = debug;
  }


  /**
   * Train the logistic regression using dense vector data
   * @param features
   * @param labels
   */
  public void train(final float[][] features, final float[] labels) {
    if (features == null || features.length == 0) {
      throw new IllegalArgumentException("The training data is empty!");
    }
    // Convert the dense vectors into sparse representation
    int dimension = features[0].length;
    Instance[] instances = new Instance[features.length];
    SparseVector[] sparseFeatures = new SparseVector[features.length];
    for (int instIndex = 0; instIndex < features.length; instIndex++) {
      SparseVector f = new SparseVector(features[instIndex]);
      instances[instIndex] = new Instance(f, labels[instIndex]);
    }
    train(dimension, instances);
  }

  /**
   * Train the logistic regression using sparse data
   * @param dimension
   * @param instances
   */
  public void train(int dimension, final Instance[] instances) {
    // Create the logistic loss function
    final LogisticLoss logLoss = new LogisticLoss(dimension, instances);
    // Create the L2 loss
    final L2RegularizerLoss l2loss;
    if (_prior != null) {
      l2loss = new L2RegularizerLoss(_prior);
    } else {
      l2loss = new L2RegularizerLoss(dimension);
    }
    // Combine the two loss functions
    final LinearCombineLoss loss = new LinearCombineLoss();
    loss.add(logLoss);
    loss.add(l2loss, _lambda);

    // Create the optimizer
    final CoordinateLipschitzGradientOptimizer optimizer = new CoordinateLipschitzGradientOptimizer(dimension, loss);
    optimizer.setDebug(_debug);
    if (_maxIter > 0) {
      optimizer.setMaxNumIteration(_maxIter);
    }

    // Start the training algorithm to minimize the loss
    optimizer.train();
    _beta = optimizer.getCofficients();
  }

  public double[] getCoefficients() {
    return _beta;
  }

  public double predict(final SparseVector feature) {
    if (this._beta == null) {
      throw new IllegalStateException("The coefficients have not been trained!");
    }
    double sum = feature.innerProduct(this._beta);
    sum = sum + _beta[_beta.length - 1];
    return MathFunctions.sigmoid(sum);
  }

  public double predict(float[] feature) {
    return predict(new SparseVector(feature));
  }
}
