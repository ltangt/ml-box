package org.ltang.mlbox.data;

import java.io.DataInputStream;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;


/**
 * Sparse vector implementation
 *
 * @author Liang Tang
 * @date Aug 27, 2014 10:10:47 PM
 */
public final class SparseVector {

  public final int[] dims;

  public final float[] vals;

  static final float EPS = 1.0E-13f;

  public SparseVector(final int[] dims, final float[] values) {
    if (dims.length != values.length) {
      throw new IllegalArgumentException("The numbers of dimensions " + "and values are not identical!");
    }
    this.dims = Arrays.copyOf(dims, dims.length);
    this.vals = Arrays.copyOf(values, values.length);
    sortDimensionIndices();
  }

  private SparseVector(final List<float[]> valuePairs, int dimension) {
    List<float[]> sortedValuePairs = new ArrayList<float[]>(valuePairs.size());
    sortedValuePairs.addAll(valuePairs);
    Collections.sort(sortedValuePairs, new Comparator<float[]>() {
      @Override
      public int compare(float[] o1, float[] o2) {
        if ((int) o1[0] > (int) o2[0]) {
          return 1;
        } else if ((int) o1[0] < (int) o2[0]) {
          return -1;
        } else {
          return 0;
        }
      }
    });
    this.dims = new int[sortedValuePairs.size()];
    this.vals = new float[sortedValuePairs.size()];
    createFromSortedValuePairs(sortedValuePairs);
  }

  public SparseVector(float[] v) {
    List<float[]> sortedValuePairs = new ArrayList<float[]>();
    for (int dimIndex = 0; dimIndex < v.length; dimIndex++) {
      float val = v[dimIndex];
      float absVal = val > 0 ? val : (-val);
      if (absVal > EPS) {
        sortedValuePairs.add(new float[]{dimIndex, val});
      }
    }
    this.dims = new int[sortedValuePairs.size()];
    this.vals = new float[sortedValuePairs.size()];
    createFromSortedValuePairs(sortedValuePairs);
  }

  private SparseVector(final Collection<Integer> dims, final Collection<Float> values) {
    if (dims.size() != values.size()) {
      throw new IllegalArgumentException("The numbers of dimensions and values " + "are not identical!");
    }
    Iterator<Integer> dimIter = dims.iterator();
    Iterator<Float> valueIter = values.iterator();
    this.dims = new int[dims.size()];
    this.vals = new float[dims.size()];
    int index = 0;
    while (dimIter.hasNext()) {
      this.dims[index] = dimIter.next();
      this.vals[index] = valueIter.next();
      index++;
    }
    sortDimensionIndices();
  }

  protected SparseVector(int numValues) {
    this.dims = new int[numValues];
    this.vals = new float[numValues];
  }

  private void createFromSortedValuePairs(final List<float[]> sortedValuePairs) {
    if (this.dims.length < sortedValuePairs.size() || this.vals.length < sortedValuePairs.size()) {
      throw new IllegalArgumentException("The number of values is greater than " + "the capacity of the sparse vector");
    }
    for (int i = 0; i < sortedValuePairs.size(); i++) {
      float[] valuePair = sortedValuePairs.get(i);
      this.dims[i] = (int) valuePair[0];
      this.vals[i] = valuePair[1];
    }
  }

  private void sortDimensionIndices() {
    List<float[]> valuePairs = new ArrayList<float[]>(this.dims.length);
    for (int i = 0; i < dims.length; i++) {
      valuePairs.add(new float[]{dims[i], vals[i]});
    }
    Collections.sort(valuePairs, new Comparator<float[]>() {
      @Override
      public int compare(float[] o1, float[] o2) {
        if ((int) o1[0] > (int) o2[0]) {
          return 1;
        } else if ((int) o1[0] < (int) o2[0]) {
          return -1;
        } else {
          return 0;
        }
      }
    });
    createFromSortedValuePairs(valuePairs);
  }

  public SparseVector(final SparseVector copy) {
    this(copy.dims, copy.vals);
  }

  public SparseVector copyNew() {
    return new SparseVector(this);
  }

  /**
   * Create a sparse vector from a string description, like {"1:10",
   * "2:11",...}
   *
   * @param descriptions
   * @return
   */
  public static SparseVector create(String[] descriptions) {
    List<Integer> dims = new ArrayList<Integer>();
    List<Float> values = new ArrayList<Float>();
    for (String entry : descriptions) {
      entry = entry.trim();
      String[] token = entry.split(":");
      dims.add(Integer.parseInt(token[0]));
      values.add(Float.parseFloat(token[1]));
    }
    return new SparseVector(dims, values);
  }

  public float get(int dimIndex) {
    int index = Arrays.binarySearch(this.dims, dimIndex);
    if (index >= 0) {
      return vals[index];
    } else {
      return 0;
    }
  }

  public int[] getDimensions() {
    return this.dims;
  }

  public float getEntryValue(int index) {
    return this.vals[index];
  }

  public SparseVector add(final SparseVector v) {
    List<float[]> resultValuePairs = new ArrayList<float[]>();
    int i = 0;
    int j = 0;
    while (i < this.dims.length || j < v.vals.length) {
      int dimIndex1 = i < this.dims.length ? this.dims[i] : -1;
      int dimIndex2 = j < v.vals.length ? v.dims[j] : -1;

      if (dimIndex1 == -1 && dimIndex2 == -1) {
        throw new IllegalStateException("Error for both dimIndex1 and dimIndex2 are -1");
      } else if (dimIndex1 == -1) { // i went to the end
        resultValuePairs.add(new float[]{dimIndex2, v.vals[j]});
        j++;
      } else if (dimIndex2 == -1) { // j went to the end
        resultValuePairs.add(new float[]{dimIndex1, this.vals[i]});
        i++;
      } else if (dimIndex1 < dimIndex2) {
        resultValuePairs.add(new float[]{dimIndex1, this.vals[i]});
        i++;
      } else if (dimIndex1 > dimIndex2) {
        resultValuePairs.add(new float[]{dimIndex2, v.vals[j]});
        j++;
      } else {
        resultValuePairs.add(new float[]{dimIndex1, this.vals[i] + v.vals[j]});
        i++;
        j++;
      }
    }
    SparseVector result = new SparseVector(resultValuePairs.size());
    result.createFromSortedValuePairs(resultValuePairs);
    return result;
  }

  public SparseVector neg() {
    SparseVector ret = new SparseVector(this);
    for (int i = 0; i < ret.vals.length; i++) {
      ret.vals[i] = -ret.vals[i];
    }
    return ret;
  }

  public SparseVector subtract(final SparseVector v) {
    SparseVector tmp = v.copyNew();
    tmp.neg();
    return this.add(tmp);
  }

  public SparseVector mul(float scalar) {
    SparseVector ret = new SparseVector(this);
    for (int i = 0; i < ret.vals.length; i++) {
      ret.vals[i] *= scalar;
    }
    return ret;
  }

  public SparseVector div(float scalar) {
    return mul(1.0f / scalar);
  }

  public SparseVector pow(float p) {
    SparseVector ret = new SparseVector(this);
    for (int i = 0; i < ret.vals.length; i++) {
      ret.vals[i] = (float) Math.pow(ret.vals[i], p);
    }
    return ret;
  }

  public SparseVector square() {
    SparseVector ret = new SparseVector(this);
    for (int i = 0; i < ret.vals.length; i++) {
      ret.vals[i] = ret.vals[i] * ret.vals[i];
    }
    return ret;
  }

  public SparseVector inverse() {
    SparseVector ret = new SparseVector(this);
    for (int i = 0; i < ret.vals.length; i++) {
      ret.vals[i] = (float) (1.0 / ret.vals[i]);
    }
    return ret;
  }

  public float innerProduct(final SparseVector v) {
    float ret = 0;
    int i = 0;
    int j = 0;
    while (i < this.dims.length && j < v.vals.length) {
      int dimIndex1 = this.dims[i];
      int dimIndex2 = v.dims[j];
      if (dimIndex1 < dimIndex2) {
        i++;
      } else if (dimIndex1 > dimIndex2) {
        j++;
      } else {
        ret += this.vals[i] * v.vals[j];
        i++;
        j++;
      }
    }
    return ret;
  }

  public float innerProduct(final float[] v) {
    return innerProduct(v, v.length);
  }

  public double innerProduct(final double[] v) {
    return innerProduct(v, v.length);
  }

  public float innerProduct(final float[] v, int vlen) {
    float ret = 0;
    for (int i = 0; i < this.dims.length; i++) {
      int dimIndex = this.dims[i];
      ret += this.vals[i] * v[dimIndex];
    }
    return ret;
  }

  public double innerProduct(final double[] v, int vlen) {
    double ret = 0;
    for (int i = 0; i < this.dims.length; i++) {
      int dimIndex = this.dims[i];
      ret += this.vals[i] * v[dimIndex];
    }
    return ret;
  }

  public float norm2() {
    return (float)Math.sqrt(innerProduct(this));
  }

  public SparseVector elementwiseMultiply(SparseVector v) {
    List<float[]> resultValuePairs = new ArrayList<float[]>();
    int i = 0;
    int j = 0;
    while (i < this.dims.length && j < v.vals.length) {
      int dimIndex1 = this.dims[i];
      int dimIndex2 = v.dims[j];
      if (dimIndex1 < dimIndex2) {
        i++;
      } else if (dimIndex1 > dimIndex2) {
        j++;
      } else {
        resultValuePairs.add(new float[]{dimIndex1, this.vals[i] * v.vals[j]});
        i++;
        j++;
      }
    }
    SparseVector result = new SparseVector(resultValuePairs.size());
    result.createFromSortedValuePairs(resultValuePairs);
    return result;
  }

  public static float cosine(SparseVector v1, SparseVector v2) {
    float inner = v1.innerProduct(v2);
    return inner / v1.norm2() / v2.norm2();
  }

  @Override
  public String toString() {
    StringBuffer buf = new StringBuffer();
    buf.append("{");
    for (int i = 0; i < dims.length; i++) {
      buf.append(dims[i] + ":" + vals[i]);
      if (i < dims.length - 1) {
        buf.append(",");
      }
    }
    buf.append("}");
    return buf.toString();
  }

  public void serialize(DataOutputStream os)
      throws IOException {
    int valueSize = this.dims.length;
    os.writeInt(valueSize);
    for (int i = 0; i < this.dims.length; i++) {
      os.writeInt(this.dims[i]);
      os.writeFloat(this.vals[i]);
    }
  }

  public static SparseVector deserialize(DataInputStream dis)
      throws IOException {
    int numValues = dis.readInt();
    int[] dims = new int[numValues];
    float[] vals = new float[numValues];
    for (int i = 0; i < numValues; i++) {
      dims[i] = dis.readInt();
      vals[i] = dis.readFloat();
    }
    return new SparseVector(dims, vals);
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + java.util.Arrays.hashCode(dims);
    result = prime * result + java.util.Arrays.hashCode(vals);
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    SparseVector other = (SparseVector) obj;
    if (!java.util.Arrays.equals(dims, other.dims)) {
      return false;
    }
    if (!java.util.Arrays.equals(vals, other.vals)) {
      return false;
    }
    return true;
  }
}
