package edu.fiu.cis.mlbox.utils;

import java.util.ArrayList;

import java.util.Collection;
import java.util.List;

/**
 * A set of utility functions for collection objects
 * 
 * @author Liang Tang
 * @date 08/08/2015
 *
 */
public class CollectionUtil {

	public static int[] asIntArray(Collection<Integer> l) {
		if (l.size() == 0) {
			return new int[0];
		} else {
			int[] arr = new int[l.size()];
			int i = 0;
			for (Integer val : l) {
				arr[i] = val;
				i++;
			}
			return arr;
		}
	}

	public static double[] asDoubleArray(Collection<Double> l) {
		if (l.size() == 0) {
			return new double[0];
		} else {
			double[] arr = new double[l.size()];
			int i = 0;
			for (Double val : l) {
				arr[i] = val;
				i++;
			}
			return arr;
		}
	}

	public static float[] asFloatArray(Collection<Float> l) {
		if (l.size() == 0) {
			return new float[0];
		} else {
			float[] arr = new float[l.size()];
			int i = 0;
			for (Float val : l) {
				arr[i] = val;
				i++;
			}
			return arr;
		}
	}

	public static boolean[] asBoolArray(Collection<Boolean> l) {
		if (l.size() == 0) {
			return new boolean[0];
		} else {
			boolean[] arr = new boolean[l.size()];
			int i = 0;
			for (Boolean val : l) {
				arr[i] = val;
				i++;
			}
			return arr;
		}
	}

	public static List<Double> asList(double[] arr) {
		List<Double> l = new ArrayList<Double>(arr.length);
		for (double val : arr) {
			l.add(val);
		}
		return l;
	}

	public static List<Float> asList(float[] arr) {
		List<Float> l = new ArrayList<Float>(arr.length);
		for (float val : arr) {
			l.add(val);
		}
		return l;
	}

	public static int findLargestNumberIndex(long[] arr) {
		int largestIndex = -1;
		for (int i = 0; i < arr.length; i++) {
			if (largestIndex == -1 || arr[largestIndex] < arr[i]) {
				largestIndex = i;
			}
		}
		return largestIndex;
	}

}
