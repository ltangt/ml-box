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
  double lambda = 1.0;

  // The trained coefficients
  double[] beta = null;

  boolean hasIntercept = true;

  public LogisticRegression() {
    this(1.0);
  }

  public LogisticRegression(boolean hasIntercept) {
    this(1.0, hasIntercept);
  }

  public LogisticRegression(double lambda) {
    this(lambda, false);
  }

  public LogisticRegression(double lambda, boolean hasIntercept) {
    this.lambda = lambda;
    this.hasIntercept = hasIntercept;
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
    LogisticLoss logLoss = new LogisticLoss(dimension, instances, hasIntercept);
    // Create the L2 loss
    L2RegularizerLoss l2loss = new L2RegularizerLoss(dimension, hasIntercept);
    // Combine the two loss functions
    LinearCombineLoss loss = new LinearCombineLoss();
    loss.add(logLoss);
    loss.add(l2loss, lambda);

    // Create the optimizer
    CoordinateLipschitzGradientOptimizer optimizer = null;
    if (hasIntercept) {
      optimizer = new CoordinateLipschitzGradientOptimizer(dimension + 1, loss);
    } else {
      optimizer = new CoordinateLipschitzGradientOptimizer(dimension, loss);
    }
    // Start the training algorithm to minimize the loss
    optimizer.train();
    beta = optimizer.getCofficients();
  }

  public double[] getCoefficients() {
    return beta;
  }

  public double predict(final SparseVector feature) {
    if (this.beta == null) {
      throw new IllegalStateException("The coefficients have not been trained!");
    }
    double sum = feature.innerProduct(this.beta);
    sum = hasIntercept ? (sum + beta[beta.length - 1]) : sum;
    return MathFunctions.sigmoid(sum);
  }

  public double predict(float[] feature) {
    return predict(new SparseVector(feature));
  }
}
