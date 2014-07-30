package dr.inferencexml.hamilton;

import dr.inference.hamilton.HamiltonUpdate;
import dr.inference.model.Likelihood;
import dr.inference.model.Variable;
import dr.inference.operators.MCMCOperator;
import dr.xml.AbstractXMLObjectParser;
import dr.xml.AttributeRule;
import dr.xml.ElementRule;
import dr.xml.XMLObject;
import dr.xml.XMLParseException;
import dr.xml.XMLSyntaxRule;

public class HamiltonUpdateParser extends AbstractXMLObjectParser {

	public static final String HAMILTON_UPDATE = "hamiltonUpdate";
	
	private static final String EPSILON = "epsilon";
	private static final String ITERATIONS = "iterations";
	private static final String VARIABLES = "variables";
	
    private final XMLSyntaxRule[] rules = {
            AttributeRule.newIntegerRule(ITERATIONS),
            AttributeRule.newDoubleRule(EPSILON),
            AttributeRule.newDoubleRule(MCMCOperator.WEIGHT),
            new ElementRule(Likelihood.class, true),
            new ElementRule(VARIABLES, new ElementRule[]{new ElementRule(Variable.class, 1, Integer.MAX_VALUE)})
    };
	
	@Override
	public String getParserName() {
		return HAMILTON_UPDATE;
	}

	@Override
	@SuppressWarnings("unchecked")
	public Object parseXMLObject(XMLObject xo) throws XMLParseException {
		
		int L = 100;
		if (xo.hasAttribute(ITERATIONS))
			L = xo.getIntegerAttribute(ITERATIONS);

		double epsilon = 0.1;
		if (xo.hasAttribute(EPSILON))
			epsilon = xo.getDoubleAttribute(EPSILON);
		
		Likelihood q = (Likelihood) xo.getChild(Likelihood.class);
		
		XMLObject cxo = xo.getChild(VARIABLES);
		Variable<Double>[] variables = new Variable[cxo.getChildCount()];
		for (int i = 0; i < variables.length; ++i)
			variables[i] = (Variable<Double>) cxo.getChild(i);
		
		HamiltonUpdate hu = new HamiltonUpdate(q, variables, epsilon, L);
		hu.setWeight(xo.getDoubleAttribute(MCMCOperator.WEIGHT));
		return hu;
		
	}

	@Override
	public XMLSyntaxRule[] getSyntaxRules() {
		return rules;
	}

	@Override
	public String getParserDescription() {
		return "An operator that makes proposals using Hamiltonain dynamics.";
	}

	@Override
	public Class<HamiltonUpdate> getReturnType() {
		return HamiltonUpdate.class;
	}

}
