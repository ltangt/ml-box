package edu.fiu.cis.mlbox.classifier.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Compute the AUC of a classifier (Area under ROC Curve)
 * @author Liang Tang
 *
 */
public class AUC {

  double[] probs = null;
  boolean[] labels = null;
  long numPositives = 0;
  long numNegatives = 0;

  public AUC() {

  }

  public double calc(double[] probs, boolean[] labels) {
    this.probs = probs;
    this.labels = labels;
    if (probs.length != labels.length) {
      throw new IllegalArgumentException("The length of probs is not equal to the length of labels");
    }

    // Sort the instances by their probabilities in descending order
    List<Integer> sortedIndices = sortByProb();

    // Compute the total number of positive and negative instances
    numPositives = 0;
    numNegatives = 0;
    for (int i = 0; i < labels.length; i++) {
      if (labels[i]) {
        numPositives++;
      } else {
        numNegatives++;
      }
    }

    // Calculate the area under the ROC curve
    // --------------------------------------------

    // The current number of true positives
    long lastTP = 0;

    // The current number of true
    long lastT = 0;

    // Last updated TPR (not last threshold's TPR)
    double lastTPR = 0;

    // Last updated FPR (not last threshold's FPR)
    double lastFPR = 0;

    double area = 0;
    for (int instIndex : sortedIndices) {
      long T = lastT + 1;
      if (labels[instIndex]) { // positive instance
        // TP and TPR are updated, FP, FPR does not change
        long TP = lastTP + 1; // TP increases by 1
        double TPR = ((double) TP) / numPositives;
        lastTPR = TPR;
        lastTP = TP;
      } else {
        // FP and FPR are updated, TP, TPR does not change
        long TP = lastTP;
        long FP = T - TP; // FP increases by 1
        double FPR = ((double) FP) / numNegatives;

        // Add the region's area
        area += lastTPR * (FPR - lastFPR);

        lastFPR = FPR;
      }
      lastT = T;
    }

    return area;
  }

  private List<Integer> sortByProb() {
    List<Integer> instIndices = new ArrayList<Integer>(probs.length);
    for (int i = 0; i < probs.length; i++) {
      instIndices.add(i);
    }
    Collections.sort(instIndices, new InstanceIndexComparator());
    Collections.reverse(instIndices);
    return instIndices;
  }

  /**
   * Compare the instance by their probability
   *
   */
  class InstanceIndexComparator implements Comparator<Integer> {


    public int compare(Integer instIndex1, Integer instIndex2) {
      // TODO Auto-generated method stub
      if (probs[instIndex1] > probs[instIndex2]) {
        return 1;
      } else if (probs[instIndex1] < probs[instIndex2]) {
        return -1;
      } else {
        return 0;
      }
    }

  }

}
