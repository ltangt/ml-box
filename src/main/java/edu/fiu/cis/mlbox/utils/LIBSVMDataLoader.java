package edu.fiu.cis.mlbox.utils;

import java.io.BufferedReader;

import java.io.FileReader;
import java.io.IOException;

import edu.fiu.cis.mlbox.data.SparseVector;

public class LIBSVMDataLoader {

	private SparseVector[] features = null;

	private double[] labels = null;

	private int dimension = -1;
	
	public LIBSVMDataLoader(String fileName, int dimension) {
		this(fileName, dimension, Integer.MAX_VALUE);
	}


	public LIBSVMDataLoader(String fileName, int dimension, int maxNumInstances) {
		try {
			this.dimension = dimension;
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
	
	// Pre-scan the file to count the number of lines and the dimension of the data space
	private void preScan(BufferedReader reader, int maxNumInstances) throws IOException {
		String line = reader.readLine();
		int maxNumLines = 0;
		int maxUsedDimension = -1;
		while (line != null && isBlankLine(line) == false && maxNumLines < maxNumInstances) {
			String[] tokens = line.split("\\s+");
			for (int i = 1; i < tokens.length; i++) {
				String[] subTokens = tokens[i].split(":");
				int dimIndex = Integer.parseInt(subTokens[0]) - 1;
				maxUsedDimension = Math.max(dimIndex+1, maxUsedDimension);
			}
			maxNumLines++;
			line = reader.readLine();
		}
		if (this.dimension == -1) {
			this.dimension = maxUsedDimension;
		}
		
		// Allocate the memory for data instances
		this.features = new SparseVector[maxNumLines];
		this.labels = new double[maxNumLines];
	}

	private void load(BufferedReader reader) throws IOException {
		// Read the first line
		String line = reader.readLine();
		int instIndex = 0;
		while (line != null && isBlankLine(line) == false 
				&& instIndex < features.length) {
			String[] tokens = line.split("\\s+");
			double[] vals = new double[tokens.length - 1];
			int[] dims = new int[tokens.length - 1];
			double y = Double.parseDouble(tokens[0]);
			labels[instIndex] = y < 0 ? 0 : 1;
			for (int i = 1; i < tokens.length; i++) {
				String[] subTokens = tokens[i].split(":");
				int dimIndex = Integer.parseInt(subTokens[0]) - 1;
				double val = Double.parseDouble(subTokens[1]);
				vals[i-1] = val;
				dims[i-1] = dimIndex;
			}
			this.features[instIndex] = new SparseVector(dims, vals);
			instIndex++;
			line = reader.readLine();
		}
	}

	private static boolean isBlankLine(String line) {
		line = line.trim();
		return line.length() == 0;
	}
	
	public int getDimension() {
		return dimension;
	}
	
	public SparseVector[] getFeatures() {
		return features;
	}
	
	public int getNumInstances() {
		return features.length;
	}
		
	public double[] getLabels() {
		return labels;
	}
	

}
