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

  public static double norm2(final double[] v) {
    return Math.sqrt(innerProduct(v,v));
  }

  public static String toString(double[] v) {
    return CollectionUtil.asList(v).toString();
  }

}
