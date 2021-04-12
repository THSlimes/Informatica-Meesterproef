package gpneuralnet;

import java.util.Random;

/* 
 * The Node class represents a node in a Network, like you'll find in the input, middle and output layers.
*/

public class Node implements java.io.Serializable {
	private static final long serialVersionUID = -7892282436879782705L;
	
	/* When talking about connections, it refers to only the connections in the layer before it. */
	public int connections = 0;
	public Node[] connectedNodes = new Node[0];
	public double[] weights = new double[0];
	private boolean isInputNode = false;
	
	public double bias = 0;
	
	protected double activation; //The calcActivation() function stores the nodes calculated value is this variable
	
	private static Random rand = new Random();
	
	//Used only for input nodes, which takes no inputs
	protected Node() {
		this.isInputNode = true;
	}
	
	//Constructor used for creating a new network	
	public Node(Node[] c, boolean iz) {
		this.connectedNodes = c;
		this.connections = c.length;
		weights = new double[this.connections];
		
		//Generates random weights for all node connections
		for (int i = 0; i < this.connections; i ++) this.weights[i] = iz ? 0 : 2*Node.rand.nextDouble()-1; //Random double from -1.0 to 1.0
		
		this.bias = iz ? 0 : 2*Node.rand.nextDouble()-1;
	}
	
	//Constructor for a node with specified connection weights
	public Node(Node[] c, double[] w) throws Exception {
		if (c.length != w.length) throw new Exception("ERROR: Connected nodes array is not of the same size as weights array.");
		else {
			this.connectedNodes = c;
			this.connections = c.length;
			this.weights = w;
			this.bias = 2*Node.rand.nextDouble()-1;
		}
	}
	
	//Sets the node's activation directly
	public void setActivation(double d) throws Exception {
		if (this.isInputNode) this.activation = d;
		else throw new Exception("ERROR: Only input nodes can have their values set directly");
	}
	
	//Returns the stored activation
	public double getActivation() {
		return this.activation;
	}
	
	//Calculates the node's Activation and stores it
	public void calculateActivation() throws Exception {
		double weightedSum = 0;
		
		for (int i = 0; i < connections; i ++) {
			weightedSum += this.weights[i] * this.connectedNodes[i].getActivation();
		}
		
		this.activation = Node.sigmoid(weightedSum + this.bias);
	}
	
	//Returns sigmoid value of d
	protected static double sigmoid(double d) {
		return 1.0d / (1 + Math.pow(Math.E, -d));
	}
	
	public String toString() {
		return String.format("Node( %d connections, Bias = %f )", this.connections, this.bias);
	}
}
