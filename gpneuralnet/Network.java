package gpneuralnet;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/* 
 * The Network class represents a neural network.
 * It is supposed to be used a black box, with which you can only
 * interact by changing the inputs and querying the outputs.
*/

public class Network implements java.io.Serializable {
	private static final long serialVersionUID = -2959307874865895882L;
	
	//Layer array's
	public Node[] inputLayer;
	public Node[][] middleLayer;
	public Node[] outputLayer;
	
	//Settings
	public double learningRate = .0001d;
	public int batchSize = 100;
	public double delta = .00000001d;
	
	private Network(int ih, int mw, int mh, int oh, boolean isZero) {
		//Setting up the array's
		this.inputLayer = new Node[ih];
		this.middleLayer = new Node[mw][mh];
		this.outputLayer = new Node[oh];
		
		//Filling the array's with nodes (from input to output)
		for (int i = 0; i < this.inputLayer.length; i ++) {
			this.inputLayer[i] = new Node();
		}
		
		for (int i = 0; i < this.middleLayer.length; i ++) {
			switch (i) {
			case 0:
				for (int j = 0; j < this.middleLayer[i].length; j ++) {
					this.middleLayer[i][j] = new Node(this.inputLayer,isZero);
				}
				break;
			default:
				for (int j = 0; j < this.middleLayer[i].length; j ++) {
					this.middleLayer[i][j] = new Node(this.middleLayer[i-1],isZero);
				}
				break;
			}
		}
		
		for (int i = 0; i < this.outputLayer.length; i ++) { //Connecting to the last middle layer
			this.outputLayer[i] = new Node(this.middleLayer[this.middleLayer.length-1],isZero);
		}
	}
	
	public Network(int ih, int mw, int mh, int oh) { //Constructor for non-zero networks
		this(ih,mw,mh,oh,false);
	}
	
	//Sets the value of a specific input node
	public void setInput(int i, double d) throws Exception {
		if (i < 0 || i >= this.inputLayer.length) throw new NullPointerException("ERROR: Index " + i + " not found in input layer.");
		else {
			this.inputLayer[i].setActivation(d);
		}
	}
	
	//Sets the value of all input nodes
	public void setInputs(double[] d) throws Exception {
		if (d.length != this.inputLayer.length) throw new NullPointerException("ERROR: Input layer and provided inputs are not of the same size.");
		else {
			for (int i = 0; i < d.length; i ++) {
				this.inputLayer[i].setActivation(d[i]);
			}
		}
	}
	
	//Returns all values from the output nodes
	public double[] getOutputs() throws Exception {
		return this.getOutputsPartially(0);
	}
	
	//Returns all outputs, but only calculates the activation after the layer l
	private double[] getOutputsPartially(int l) throws Exception {
		double[] out = new double[this.outputLayer.length];
		
		for (int i = l; i < this.middleLayer.length; i ++) {
			for (int j = 0; j < this.middleLayer[i].length; j ++) {
				this.middleLayer[i][j].calculateActivation();
			}
		}
		
		for (int i = 0; i < this.outputLayer.length; i ++) {
			this.outputLayer[i].calculateActivation();
			out[i] = this.outputLayer[i].getActivation();
		}
		
		return out;
	}
	
	//Returns the value of a specific output node
	public double getOutput(int index) throws Exception {
		if (index < 0 || index >= this.outputLayer.length) throw new NullPointerException("ERROR: Index " + index + " not found in output layer.");
		else {
			for (int i = 0; i < this.middleLayer.length; i ++) {
				for (int j = 0; j < this.middleLayer[i].length; j ++) {
					this.middleLayer[i][j].calculateActivation();
				}
			}
			
			this.outputLayer[index].calculateActivation();
			return this.outputLayer[index].getActivation();
		}
	}
	
	//Returns the index of the greatest value in the output layer
	public int getLabel() throws Exception {
		double[] out = this.getOutputs();
		
		int l = 0;
		for (int i = 0; i < out.length; i ++) {
			if (out[i] > out[l]) l = i;
		}
		
		return l;
	}
	
	//Tries to teach the Network to return values similar to the data provided
	public void learnFromData(TrainingData[] data) throws Exception {
		//Shuffles the data
		List<TrainingData> asList = Arrays.asList(data);
		Collections.shuffle(asList);
		for (int i = 0; i < asList.size(); i ++) data[i] = asList.get(i);
		
		double costRecord = Double.MAX_VALUE;
		
		for (int i = 0; i < data.length; i += this.batchSize) {
			int piecesUsed = 0;
			double totalCost = 0;
			
			for (int j = 0; j < this.batchSize && i+j < data.length; j ++) {
				TrainingData piece = data[i+j];
				Network deltaNetwork = this.getZeroNetwork();
				this.setInputs(piece.inputs);
				double baseCost = piece.getCost(this.getOutputs());
				totalCost += baseCost;
				
				//Calculating middle layer delta
				for (int k = 0; k < this.middleLayer.length; k ++) {
					for (int l = 0; l < this.middleLayer[k].length; l ++) {
						for (int m = 0; m < this.middleLayer[k][l].connections; m ++) {
							this.middleLayer[k][l].weights[m] += (this.delta * baseCost * baseCost);
							double alteredCost = piece.getCost(this.getOutputsPartially(k));
							this.middleLayer[k][l].weights[m] -= (this.delta * baseCost * baseCost);
							
							double dcdw = (alteredCost-baseCost)/(this.delta * baseCost * baseCost);
							
							deltaNetwork.middleLayer[k][l].weights[m] -= dcdw;
						}
						
						this.middleLayer[k][l].bias += (this.delta * baseCost * baseCost);
						double alteredCost = piece.getCost(this.getOutputsPartially(k));
						this.middleLayer[k][l].bias -= (this.delta * baseCost * baseCost);
						double dcdb = (alteredCost-baseCost)/(this.delta * baseCost * baseCost);
						deltaNetwork.middleLayer[k][l].bias -= dcdb;
					}
				}
				
				//Calculating output layer delta
				for (int k = 0; k < this.outputLayer.length; k ++) {
					for (int l = 0; l < this.outputLayer[k].connections; l ++) {
						this.outputLayer[k].weights[l] += (this.delta * baseCost * baseCost);
						double alteredCost = piece.getCost(this.getOutputsPartially(this.middleLayer.length));
						this.outputLayer[k].weights[l] -= (this.delta * baseCost * baseCost);
						
						double dcdw = (alteredCost-baseCost)/(this.delta * baseCost * baseCost);
						
						deltaNetwork.outputLayer[k].weights[l] -= dcdw;
					}
					
					this.outputLayer[k].bias += (this.delta * baseCost * baseCost);
					double alteredCost = piece.getCost(this.getOutputsPartially(this.middleLayer.length));
					this.outputLayer[k].bias -= (this.delta * baseCost * baseCost);
					double dcdb = (alteredCost-baseCost)/(this.delta * baseCost * baseCost);
					deltaNetwork.outputLayer[k].bias -= dcdb;
				}
				
				//Adjusting middle layer
				for (int k = 0; k < this.middleLayer.length; k ++) {
					for (int l = 0; l < this.middleLayer[k].length; l ++) {
						for (int m = 0; m < this.middleLayer[k][l].connections; m ++) {
							this.middleLayer[k][l].weights[m] += this.learningRate*deltaNetwork.middleLayer[k][l].weights[m]*baseCost;
						}
						
						this.middleLayer[k][l].bias += this.learningRate*deltaNetwork.middleLayer[k][l].bias*baseCost;
					}
				}
				
				//Adjusting output layer
				for (int k = 0; k < this.outputLayer.length; k ++) {
					for (int l = 0; l < this.outputLayer[k].connections; l ++) {
						this.outputLayer[k].weights[l] += this.learningRate*deltaNetwork.outputLayer[k].weights[l]*baseCost;
					}
					
					this.outputLayer[k].bias += this.learningRate*deltaNetwork.outputLayer[k].bias*baseCost;
				}
				
				piecesUsed ++;
			}
		}
	}
	
	/* Reading and writing Network to a file */
	
	//Writes the Network to a text file to be used again later
	public void saveToFile(String location) throws Exception {
		FileOutputStream f = new FileOutputStream(new File(System.getProperty("user.dir") + "\\" + location + ".mdl"));
        ObjectOutputStream o = new ObjectOutputStream(f);
        
        o.writeObject(this);
        
        o.close();
        f.close();
	}
	
	//Allows for the loading of saved Networks
	public static Network loadFromFile(String location) throws ClassNotFoundException, IOException {
		FileInputStream fi = new FileInputStream(new File(System.getProperty("user.dir") + "\\" + location + ".mdl"));
        ObjectInputStream oi = new ObjectInputStream(fi);
        
        Network out = (Network) oi.readObject();
        
        fi.close();
        oi.close();
        
		return out;
	}
	
	//Returns a network with the same size, but with all weights and biases set to 0
	private Network getZeroNetwork() {
		return new Network(this.inputLayer.length,this.middleLayer.length,this.middleLayer[0].length,this.outputLayer.length,true);
	}
	
	public String toString() {
		return String.format("Network( %d inputs, %dx%d middle, %d outputs)", this.inputLayer.length, this.middleLayer.length, this.middleLayer[0].length, this.outputLayer.length);
	}
}