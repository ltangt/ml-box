package org.ltang.mlbox.utils;

import java.util.Random;

/**
 * A set of mathematical functions for collection objects
 * @author Liang Tang
 *
 */
public class MathFunctions {

  public static final double EPS = 1.0E-13;

  public static double sigmoid(double v) {
    return 1.0 / (1 + Math.exp(-v));
  }

  public static boolean almostEqual(double x, double y) {
    double s = x - y;
    return (-EPS < s) && (s < EPS);
  }

  public static boolean almostEqual(double x, double y, double eps) {
    double s = x - y;
    return (-eps < s) && (s < eps);
  }

  public static void normalizeProbabilites(double[] probs) {
    // Normalize the probabilities (make sure the summation to be 1)
    double sum = 0;
    for (int i = 0; i < probs.length; i++) {
      sum += probs[i];
    }
    if (almostEqual(sum, 0)) {
      throw new IllegalArgumentException("The sum of probabilities is 0");
    }

    for (int i = 0; i < probs.length; i++) {
      probs[i] /= sum;
    }
  }

  public static int randomSample(double[] probs, Random rand) {
    // Convert to cumulative probs
    double[] cumProbs = new double[probs.length];
    for (int i = 0; i < probs.length; i++) {
      if (i == 0) {
        cumProbs[i] = probs[i];
      } else {
        cumProbs[i] = cumProbs[i - 1] + probs[i];
      }
    }
    double v = rand.nextDouble();
    for (int i = 0; i < cumProbs.length; i++) {
      if (i == 0) {
        if (v <= cumProbs[i]) {
          return i;
        }
      } else {
        if (v > cumProbs[i - 1] && v <= cumProbs[i]) {
          return i;
        }
      }
    }
    return cumProbs.length - 1;
  }

  public static boolean isNumeric(String str) {
    try
    {
      double d = Double.parseDouble(str);
    }
    catch(NumberFormatException nfe)
    {
      return false;
    }
    return true;
  }

    public static double sum(double [] arr) {
        double s = 0;
        for (double val : arr) {
            s += val;
        }
        return s;
    }

//  /**
//  * Generate the beta distribution sample via gamma distribution.
//  *
//  * @param alpha
//  * @param beta
//  * @return
//  */
//  public static double randomBeta(double alpha, double beta) {
//    GammaDistribution gamma1 = new GammaDistribution(alpha, 1);
//    GammaDistribution gamma2 = new GammaDistribution(beta, 1);
//    double x = gamma1.sample();
//    double y = gamma2.sample();
//    return x / (x + y);
//  }

}
