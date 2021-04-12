package gpneuralnet;

/* This class represents a single entry of training data */

public class TrainingData {
	final public double[] inputs;
	final public double[] expectedOutputs;
	
	public TrainingData(double[] i, double[] o) {
		this.inputs = i;
		this.expectedOutputs = o;
	}
	
	//Calculates the differences between the output values and the expected output values
	private double[] outputDifferences(double[] n) throws Exception {
		if (n.length != this.expectedOutputs.length) throw new Exception("ERROR: Training data incompatible with network (expected size: " + n.length + ", got: " + this.expectedOutputs.length + ")");
		else {
			double[] dif = new double[n.length];
			
			for (int i = 0; i < n.length; i ++) {
				dif[i] = this.expectedOutputs[i] - n[i];
			}
			
			return dif;
		}
	}
	
	//Returns the sum of the squares of the differences between values in n and the expectedOutputs
	protected double getCost(double[] n) throws Exception {
		double[] da = this.outputDifferences(n);
		double o = 0;
		for (double d : da) o += d*d;
		return o;
	}
	
	//Returns the index of the largest value in expectedOutputs
	public int getLabel() throws Exception {		
		int l = 0;
		for (int i = 0; i < this.expectedOutputs.length; i ++) {
			if (this.expectedOutputs[i] > this.expectedOutputs[l]) l = i;
		}
		
		return l;
	}
	
	public String toString() {
		return String.format("TrainingData( %d Inputs, %d outputs )",this.inputs.length,this.expectedOutputs.length);
	}
}
