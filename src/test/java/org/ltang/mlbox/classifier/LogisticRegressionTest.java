package org.ltang.mlbox.classifier;

import org.ltang.mlbox.data.Instance;
import java.io.File;

import java.io.IOException;

import org.ltang.mlbox.classifier.eval.AUC;
import org.ltang.mlbox.utils.LIBSVMDataLoader;
import org.testng.Assert;
import org.testng.annotations.Test;

import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import org.ltang.mlbox.utils.VectorUtil;


public class LogisticRegressionTest {

  @Test()
  public void testToy1() {
    LogisticRegression LR = new LogisticRegression();
    float[][] data = new float[4][];
    float[] labels = new float[4];
    data[0] = new float[]{0f, -1f};
    labels[0] = 1;
    data[1] = new float[]{1f, -1f};
    labels[1] = 0;
    data[2] = new float[]{1f, 1f};
    labels[2] = 0;
    data[3] = new float[]{0f, 1f};
    labels[3] = 1;
    LR.train(data, labels);
    System.out.println(VectorUtil.toString(LR.getCoefficients()));

    Assert.assertTrue(LR.predict(new float[]{0.3f, 0.5f}) > 0.5);
    System.out.println(LR.predict(new float[]{0.3f, 0.5f}));

    Assert.assertTrue(LR.predict(new float[]{100f, -100f}) < 0.5);
    System.out.println(LR.predict(new float[]{100f, -100f}));

    Assert.assertTrue(LR.predict(new float[]{0f, 100f}) > 0.5);
    System.out.println(LR.predict(new float[]{0f, 100f}));

    Assert.assertTrue(LR.predict(new float[]{-10f, 0f}) > 0.5);
    System.out.println(LR.predict(new float[]{-10f, 0f}));
  }

  /**
   * The test data set is here :
   * 	 http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
   * @throws Exception
   */
  @Test
  public void testA1A()
      throws IOException, Exception {
    int dimension = 123;
    LIBSVMDataLoader trainLoader = new LIBSVMDataLoader("data/libsvm/a1a.train.txt", dimension);
    LogisticRegression LR = new LogisticRegression();
    LR.train(dimension, trainLoader.getInstances());

    LIBSVMDataLoader testLoader = new LIBSVMDataLoader("data/libsvm/a1a.test.txt", dimension);

    int numTests = testLoader.getNumInstances();
    double[] scores = new double[numTests];
    boolean[] labels = new boolean[numTests];
    int numAcc = 0;
    Instance[] testInsts = testLoader.getInstances();
    for (int i = 0; i < numTests; i++) {
      Instance inst = testInsts[i];
      scores[i] = LR.predict(inst.getFeatures());
      labels[i] = inst.getLabel() > 0.5;
      boolean predictedLabel = scores[i] > 0.5;
      if (labels[i] == predictedLabel) {
        numAcc++;
      }
    }
    AUC auc = new AUC();
    double aucScore = auc.calc(scores, labels);
    System.out.println("AUC = " + aucScore + " , Accuracy = " + ((double) numAcc) / numTests);

    // Build another model using Liblinear: http://liblinear.bwaldvogel.de
    Problem problem = Problem.readFromFile(new File("data/libsvm/a1a.train.txt"), 1);
    SolverType solver = SolverType.L2R_LR; // -s 0
    double C = 1.0;    // cost of constraints violation
    double eps = 0.0001; // stopping criteria

    Parameter parameter = new Parameter(solver, C, eps);
    Model model = Linear.train(problem, parameter);

    Problem testProblem = Problem.readFromFile(new File("data/libsvm/a1a.test.txt"), 1);
    numAcc = 0;
    for (int i = 0; i < numTests; i++) {
      scores[i] = Linear.predict(model, testProblem.x[i]);
      boolean predictedLabel = scores[i] > 0.5;
      if (labels[i] == predictedLabel) {
        numAcc++;
      }
    }
    double LibLinear_aucScore = auc.calc(scores, labels);
    System.out.println("Liblinear AUC = " + aucScore + " , Accuracy = " + ((double) numAcc) / numTests);
    Assert.assertTrue(aucScore > LibLinear_aucScore*0.95);
  }
}
