package org.ltang.mlbox.utils;

import org.ltang.mlbox.data.Instance;
import java.io.BufferedReader;

import java.io.FileReader;
import java.io.IOException;

import org.ltang.mlbox.data.SparseVector;


public class LIBSVMDataLoader {

  private Instance[] _instances = null;

  private int _dimension = -1;

  public LIBSVMDataLoader(String fileName, int dimension) {
    this(fileName, dimension, Integer.MAX_VALUE);
  }

  public LIBSVMDataLoader(String fileName, int dimension, int maxNumInstances) {
    try {
      this._dimension = dimension;
      BufferedReader reader = new BufferedReader(new FileReader(fileName));
      preScan(reader, maxNumInstances);
      reader.close();

      reader = new BufferedReader(new FileReader(fileName));
      load(reader);
      reader.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  // Pre-scan the file to count the number of lines and the _dimension of the data space
  private void preScan(BufferedReader reader, int maxNumInstances)
      throws IOException {
    String line = reader.readLine();
    int maxNumLines = 0;
    int maxUsedDimension = -1;
    while (line != null && isBlankLine(line) == false && maxNumLines < maxNumInstances) {
      String[] tokens = line.split("\\s+");
      for (int i = 1; i < tokens.length; i++) {
        String[] subTokens = tokens[i].split(":");
        int dimIndex = Integer.parseInt(subTokens[0]) - 1;
        maxUsedDimension = Math.max(dimIndex + 1, maxUsedDimension);
      }
      maxNumLines++;
      line = reader.readLine();
    }
    if (this._dimension == -1) {
      this._dimension = maxUsedDimension;
    }

    // Allocate the memory for data instances
    _instances = new Instance[maxNumLines];
  }

  private void load(BufferedReader reader)
      throws IOException {
    // Read the first line
    String line = reader.readLine();
    int instIndex = 0;
    while (line != null && isBlankLine(line) == false && instIndex < _instances.length) {
      String[] tokens = line.split("\\s+");
      float[] vals = new float[tokens.length - 1];
      int[] dims = new int[tokens.length - 1];
      float y = Float.parseFloat(tokens[0]);
      y = y < 0 ? 0 : 1;
      for (int i = 1; i < tokens.length; i++) {
        String[] subTokens = tokens[i].split(":");
        int dimIndex = Integer.parseInt(subTokens[0]) - 1;
        float val = Float.parseFloat(subTokens[1]);
        vals[i - 1] = val;
        dims[i - 1] = dimIndex;
      }
      SparseVector features = new SparseVector(dims, vals);
      _instances[instIndex] = new Instance(features, y);
      instIndex++;
      line = reader.readLine();
    }
  }

  private static boolean isBlankLine(String line) {
    line = line.trim();
    return line.length() == 0;
  }

  public int getDimension() {
    return _dimension;
  }

  public Instance[] getInstances() {
    return _instances;
  }

  public int getNumInstances() {
    return _instances.length;
  }

}
