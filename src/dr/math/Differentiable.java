package dr.math;

import dr.inference.model.Parameter;

/**
 * 
 * @author Arman D. Bilge <armanbilge@gmail.com>
 *
 */
public interface Differentiable {

	/**
	 * Partially differentiates a function.
	 * @param p parameter to differentiate with respect to
	 * @param d specific dimension of the given parameter to with respect to
	 * @return the partial derivative of the function
	 */
	public double differentiate(Parameter p, int d);
	
	/**
	 * Partially differentiates a function.
	 * 
	 * Convenience method that should return the derivative with respect to the 
	 * element of the parameter.
	 * 
	 * @param p parameter to differentiate with respect to
	 * @return the partial derivative of the function
	 */
	public double differentiate(Parameter p);
	
}
