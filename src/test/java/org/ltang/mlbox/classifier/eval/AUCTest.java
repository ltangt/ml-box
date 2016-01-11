package org.ltang.mlbox.classifier.eval;

import java.util.Random;

import org.testng.Assert;
import org.testng.annotations.Test;

public class AUCTest {
	
	@Test()
	public void testRandomAUC() {
		Random rand = new Random(System.nanoTime());
		final int numTests = 10000;
		double[] scores = new double[numTests];
		boolean[] labels = new boolean[numTests];
		for (int i=0; i<numTests; i++) {
			labels[i] = rand.nextBoolean();
			scores[i] = rand.nextDouble();
		}
		AUC auc = new AUC();
		double error = Math.abs(auc.calc(scores, labels) - 0.5);
		Assert.assertTrue(error < 0.1);
	}
	
	@Test
	public void testBestAUC() {
		Random rand = new Random(System.nanoTime());
		final int numTests = 10000;
		double[] scores = new double[numTests];
		boolean[] labels = new boolean[numTests];
		for (int i=0; i<numTests; i++) {
			labels[i] = rand.nextBoolean();
			scores[i] = labels[i] ? 1.0 : 0.0;
		}
		AUC auc = new AUC();
		double error = Math.abs(auc.calc(scores, labels) - 1.0);
		Assert.assertTrue(error < 0.00001);
	}
	
	@Test
	public void testSyntheticData() {
		// The positive sample should have 0.7 probability to be ranked
		//    higher than the negative sample		
		final double synAuc = 0.7;
		Random rand = new Random(System.nanoTime());
		final int numTests = 50000;
		double[] scores = new double[numTests];
		boolean[] labels = new boolean[numTests];
		for (int i=0; i<numTests; i++) {
			labels[i] = rand.nextBoolean();
			if (rand.nextDouble() <= synAuc) {
				scores[i] = labels[i] ? 1.0 : 0.0;
			}
			else {
				scores[i] = labels[i] ? 0.0 : 1.0;
			}
		}
		AUC auc = new AUC();
		double error = Math.abs(auc.calc(scores, labels) - synAuc);
		Assert.assertTrue(error < 0.01);
	}
	

}
