package org.ltang.mlbox.utils;

import java.util.Arrays;


/**
 * A set of utility functions for vector operations
 * @author Liang Tang
 */
public final class VectorUtil {

  public static double[] FloatToDouble(final float[] v) {
    double[] dv = new double[v.length];
    for (int i=0; i<v.length; i++) {
      dv[i] = (double)v[i];
    }
    return dv;
  }

  public static float[] DoubleToFloat(final double[] v) {
    float[] fv = new float[v.length];
    for (int i=0; i<v.length; i++) {
      fv[i] = (float)v[i];
    }
    return fv;
  }

  public static double[] add(final double[] v1, final double[] v2) {
    if (v1.length != v2.length) {
      throw new IllegalArgumentException("Two vectors' dimensions are not identical !");
    }
    double[] ret = new double[v1.length];
    for (int i=0; i<v1.length; i++) {
      ret[i] = v1[i]+v2[i];
    }
    return ret;
  }

  public static double[] minus(final double[] v1, final double[] v2) {
    if (v1.length != v2.length) {
      throw new IllegalArgumentException("Two vectors' dimensions are not identical !");
    }
    double[] ret = new double[v1.length];
    for (int i=0; i<v1.length; i++) {
      ret[i] = v1[i]-v2[i];
    }
    return ret;
  }

  public static double[] copyNew(final double[] v) {
    return Arrays.copyOf(v, v.length);
  }

  public static double[] mul(final double[] v, double scalar) {
    double[] ret = new double[v.length];
    for (int i=0; i<v.length; i++) {
      ret[i] = v[i]*scalar;
    }
    return ret;
  }

  public static double[] div(final double[] v, double scalar) {
    return mul(v, 1.0/scalar);
  }

  public static double innerProduct(final double[] v1, final double[] v2) {
    if (v1.length != v2.length) {
      throw new IllegalArgumentException("Two vectors' dimensions are not identical !");
    }
    double ret= 0;
    for (int i=0; i<v1.length; i++) {
      ret += v1[i]*v2[i];
    }
    return ret;
  }

  public static double[] subVector(final double[] v, int len) {
    double[] sub = new double[len];
    for (int i=0; i<len; i++) {
      sub[i] = v[i];
    }
    return sub;
  }

  public static double[] concateVector(final double[] v1, final double[] v2) {
    double[] v = new double[v1.length+v2.length];
    System.arraycopy(v1, 0, v, 0, v1.length);
    System.arraycopy(v2, 0, v, v1.length, v2.length);
    return v;
  }

  /**
   * Compute the norm of a dense vector. The scale-version is more accurate than takeing the sum of square and get the square root.
   * @param x
   * @return
   */
  public static double norm2(final double[] x) {
    int n = x.length;

    if (n < 1) {
      return 0;
    }

    if (n == 1) {
      return Math.abs(x[0]);
    }

    double scale = 0; // scaling factor that is factored out
    double sum = 1; // basic sum of squares from which scale has been factored out
    for (int i = 0; i < n; i++) {
      if (x[i] != 0) {
        double abs = Math.abs(x[i]);
        // try to get the best scaling factor
        if (scale < abs) {
          double t = scale / abs;
          sum = 1 + sum * (t * t);
          scale = abs;
        } else {
          double t = abs / scale;
          sum += t * t;
        }
      }
    }
    return scale * Math.sqrt(sum);
  }

  public static String toString(double[] v) {
    return CollectionUtil.asList(v).toString();
  }

}
