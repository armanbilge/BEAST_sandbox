package dr.inference.hamilton;

import dr.inference.model.Likelihood;
import dr.inference.model.Variable;
import dr.inference.operators.OperatorFailedException;
import dr.inference.operators.SimpleMCMCOperator;
import dr.math.MathUtils;

/**
 * 
 * @author Arman D. Bilge <armanbilge@gmail.com>
 *
 */
public class HamiltonUpdate extends SimpleMCMCOperator {

	private final Likelihood q;
	private final Variable<Double>[] variables;
	private final int totalDimensions;
	
	private double epsilon;
	private double L;
	
	public HamiltonUpdate(Likelihood q, Variable<Double>[] variables, double epsilon, double L) {
		this.q = q;
		this.variables = variables;
		int d = 0;
		for (Variable<Double> v : variables) d += v.getSize();
		totalDimensions = d;
		this.epsilon = epsilon;
		this.L = L;
	}
	
	public String getPerformanceSuggestion() {
		return "No performance suggestion.";
	}

	@Override
	public String getOperatorName() {
		return "HamiltonUpdate";
	}

	@Override
	public double doOperation() throws OperatorFailedException {
		
		double[] p = new double[totalDimensions];
		double[] storedP = new double[totalDimensions];
		for (int i = 0; i < p.length; ++i) {
			double ng = MathUtils.nextGaussian();
			p[i] = ng;
			storedP[i] = ng;
		}
		
		int i = 0;
		for (Variable<Double> v : variables) {
			for (int d = 0; d < v.getSize(); ++d)
				p[i++] += epsilon * q.differentiate(v, d) / 2;
		}
		
		for (int l = 0; l < L; ++l) {
			i = 0;
			for (Variable<Double> v : variables) {
				for (int d = 0; d < v.getSize(); ++d)
					v.setValue(d, v.getValue(d) - epsilon * p[i++]);
			}
			if (l != L-1) {
				i = 0;
				for (Variable<Double> v : variables) {
					for (int d = 0; d < v.getSize(); ++d)
						p[i++] += epsilon * q.differentiate(v, d);
				}
			}
		}
		
		i = 0;
		for (Variable<Double> v : variables) {
			for (int d = 0; d < v.getSize(); ++d) {
				p[i] += epsilon * q.differentiate(v, d) / 2;
				p[i] *= -1;
				++i;
			}
		}
		
		double storedK = 0.0;
		double proposedK = 0.0;
		for (i = 0; i < p.length; ++i) {
			storedK += storedP[i] * storedP[i];
			proposedK += p[i] * p[i];
		}
		storedK /= 2;
		proposedK /= 2;
		
		return storedK - proposedK;
	}

}
