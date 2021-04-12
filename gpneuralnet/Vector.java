package gpneuralnet;

/*
 * The Vector class represents a Vector of any dimension.
 * It also includes some basic Vector math functions.
 * It is only used for easily calculating Node and InputNode outputs.
*/

public class Vector {
	public int size;
	public double[] values;
	
	//Creates a Vector object using an array of values
	public Vector(double[] v) {
		this.values = v;
		this.size = v.length;
	}
	
	/* Mathematical Operations */
	
	//Adds a Vector to this
	public Vector add(Vector v2) {
		for (int i = 0; i < Math.min(this.size,v2.size); i ++) {
			this.values[i] += v2.values[i];
		}
		
		return this;
	}
	
	//Subtracts a Vector to this
	public Vector sub(Vector v2) {
		for (int i = 0; i < Math.min(this.size,v2.size); i ++) {
			this.values[i] -= v2.values[i];
		}
		
		return this;
	}
	
	//Returns the dot product of this and another Vector
	public double dot(Vector v2) throws Exception {
		if (this.size != v2.size) throw new Exception("ERROR: Cannot calculate dot product of two Vector with different sizes.");
		else {
			double out = 0;
			
			for (int i = 0; i < this.size; i ++) {
				out += this.values[i] * v2.values[i];
			}
			
			return out;
		}
	}
	
	//Returns the dot product of two Vectors
	public static double dot(Vector v1, Vector v2) throws Exception {
		if (v1.size != v2.size) throw new Exception("ERROR: Cannot calculate dot product of two Vector with different sizes.");
		else {
			double out = 0;
			
			for (int i = 0; i < v1.size; i ++) {
				out += v1.values[i] * v2.values[i];
			}
			
			return out;
		}
	}
	
	//Multiplies all values of this by the value of f
	public Vector scale(double f) {
		for (int i = 0; i < this.size; i ++) {
			this.values[i] *= f;
		}
		
		return this;
	}
	
	/* Getting of Properties */
	
	//Returns the magnitude or length of this
	public double mag() {
		return Math.sqrt(this.sqMag());
	}
	
	//Returns the square of the magnitude of this
	public double sqMag() {
		double sqSum = 0;
		for (int i = 0; i < this.size; i ++) sqSum += this.values[i]*this.values[i];
		
		return sqSum;
	}
	
	//Returns the distance between this and another Vector as if they were points
	public double distanceTo(Vector v2) throws CloneNotSupportedException {
		Vector v1 = (Vector) this.clone();
		v1.sub(v2);
		return v2.mag();
	}
	
	public String toString() {
		String out = String.format("Vector( %d dimensional, values = {");
		
		for (int i = 0; i < this.size; i ++) out += (i != 0 ? ", " : "") + this.values[i];	
		
		out += "})";
		
		return out;
	}
}
