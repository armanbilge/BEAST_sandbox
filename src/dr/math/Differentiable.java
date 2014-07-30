package dr.math;

import dr.inference.model.Variable;

/**
 * 
 * @author Arman D. Bilge <armanbilge@gmail.com>
 *
 */
public interface Differentiable {

	/**
	 * Partially differentiates a function.
	 * @param v variable to differentiate with respect to
	 * @param d dimension of the given variable to differentiate with respect to
	 * @return the partial derivative of the function
	 */
	public double differentiate(Variable<Double> v, int d);
	
	/**
	 * Partially differentiates a function.
	 * 
	 * Convenience method that should return the derivative with respect to the 
	 * first dimension of the variable.
	 * 
	 * @param v variable to differentiate with respect to
	 * @return the partial derivative of the function
	 */
	public double differentiate(Variable<Double> v);
	
}
