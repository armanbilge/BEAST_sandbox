/*
 * TreeLikelihoodGenerator.java
 *
 * Copyright (C) 2002-2009 Alexei Drummond and Andrew Rambaut
 *
 * This file is part of BEAST.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * BEAST is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * BEAST is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAST; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

package dr.app.beauti.generator;

import dr.app.beauti.components.ComponentFactory;
import dr.app.beauti.components.dnds.DnDsComponentOptions;
import dr.app.beauti.components.dollo.DolloComponentOptions;
import dr.app.beauti.options.*;
import dr.app.beauti.types.MicroSatModelType;
import dr.app.beauti.util.XMLWriter;
import dr.evolution.datatype.Nucleotides;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.sitemodel.GammaSiteModel;
import dr.evomodel.sitemodel.SiteModel;
import dr.evomodel.substmodel.AsymmetricQuadraticModel;
import dr.evomodel.substmodel.LinearBiasModel;
import dr.evomodel.substmodel.NucModelType;
import dr.evomodel.substmodel.TwoPhaseModel;
import dr.evomodel.tree.TreeModel;
import dr.evomodelxml.branchratemodel.DiscretizedBranchRatesParser;
import dr.evomodelxml.branchratemodel.RandomLocalClockModelParser;
import dr.evomodelxml.branchratemodel.StrictClockBranchRatesParser;
import dr.evomodelxml.tree.MicrosatelliteSamplerTreeModelParser;
import dr.evomodelxml.treelikelihood.MicrosatelliteSamplerTreeLikelihoodParser;
import dr.evomodelxml.treelikelihood.TreeLikelihoodParser;
import dr.evoxml.AlignmentParser;
import dr.evoxml.SitePatternsParser;
import dr.util.Attribute;
import dr.xml.XMLParser;

/**
 * @author Alexei Drummond
 * @author Andrew Rambaut
 * @author Walter Xie
 */
public class TreeLikelihoodGenerator extends Generator {

    public TreeLikelihoodGenerator(BeautiOptions options, ComponentFactory[] components) {
        super(options, components);
    }

    /**
     * Write the tree likelihood XML block.
     *
     * @param partition the partition  to write likelihood block for
     * @param writer    the writer
     */
    public void writeTreeLikelihood(PartitionData partition, XMLWriter writer) {

        PartitionSubstitutionModel model = partition.getPartitionSubstitutionModel();

        DolloComponentOptions stochasticDolloComponent = (DolloComponentOptions) options
                .getComponentOptions(DolloComponentOptions.class);

        boolean doStochasticDollo =  stochasticDolloComponent.doStochasticDollo(model);

        if (model.getDataType() == Nucleotides.INSTANCE && model.getCodonHeteroPattern() != null) {

			DnDsComponentOptions robustCountingComponent = (DnDsComponentOptions) options
					.getComponentOptions(DnDsComponentOptions.class);

			boolean doRobustCounting = robustCountingComponent.doRobustCounting();

			if (doRobustCounting) {

				for (int i = 1; i <= model.getCodonPartitionCount(); i++) {
					writeAncestralTreeLikelihood(
							TreeLikelihoodParser.TREE_LIKELIHOOD, i,
							partition, writer);
				}

			} else {

				for (int i = 1; i <= model.getCodonPartitionCount(); i++) {
					writeTreeLikelihood(TreeLikelihoodParser.TREE_LIKELIHOOD,
							i, partition, writer);
				}

			}// END: doRobustCounting

        } else {
            writeTreeLikelihood(TreeLikelihoodParser.TREE_LIKELIHOOD, -1, partition, writer);
        }
    }

	/**
	 * Write the ancestral tree likelihood XML block. Needed for robust dN/dS
	 * counting model
	 */
	private void writeAncestralTreeLikelihood(String id, int num,
			PartitionData partition, XMLWriter writer) {

		PartitionSubstitutionModel substModel = partition
				.getPartitionSubstitutionModel();
		PartitionTreeModel treeModel = partition.getPartitionTreeModel();
		PartitionClockModel clockModel = partition.getPartitionClockModel();

		writer
				.writeComment("Ancestral likelihood for tree given sequence data");

		writer.writeOpenTag(TreeLikelihoodParser.ANCESTRAL_TREE_LIKELIHOOD,
				new Attribute[] {
						new Attribute.Default<String>(XMLParser.ID, substModel
								.getPrefix(num)
								+ partition.getPrefix() + id),
						new Attribute.Default<Boolean>(
								TreeLikelihoodParser.USE_AMBIGUITIES,
								substModel.isUseAmbiguitiesTreeLikelihood()) });

		writeCodonPatternsRef(
				substModel.getPrefix(num) + partition.getPrefix(), num,
				substModel.getCodonPartitionCount(), writer);

		writer.writeIDref(TreeModel.TREE_MODEL, treeModel.getPrefix()
				+ TreeModel.TREE_MODEL);

		writer.writeIDref(GammaSiteModel.SITE_MODEL, substModel.getPrefix(num)
				+ SiteModel.SITE_MODEL);

		writer.writeIDref(
				StrictClockBranchRatesParser.STRICT_CLOCK_BRANCH_RATES,
				clockModel.getPrefix() + BranchRateModel.BRANCH_RATES);

		writer.writeIDref(NucModelType.HKY.getXMLName(), substModel
				.getPrefix(num)
				+ "hky");

		writer.writeCloseTag(TreeLikelihoodParser.ANCESTRAL_TREE_LIKELIHOOD);

	}// END: writeAncestralTreeLikelihood

    /**
     * Write the tree likelihood XML block.
     *
     * @param id        the id of the tree likelihood
     * @param num       the likelihood number
     * @param partition the partition to write likelihood block for
     * @param writer    the writer
     */
    private void writeTreeLikelihood(String id, int num, PartitionData partition, XMLWriter writer) {
        PartitionSubstitutionModel substModel = partition.getPartitionSubstitutionModel();
        PartitionTreeModel treeModel = partition.getPartitionTreeModel();
        PartitionClockModel clockModel = partition.getPartitionClockModel();

        writer.writeComment("Likelihood for tree given sequence data");

        if (num > 0) {
            writer.writeOpenTag(
                    TreeLikelihoodParser.TREE_LIKELIHOOD,
                    new Attribute[]{
                            new Attribute.Default<String>(XMLParser.ID, substModel.getPrefix(num) + partition.getPrefix() + id),
                            new Attribute.Default<Boolean>(TreeLikelihoodParser.USE_AMBIGUITIES, substModel.isUseAmbiguitiesTreeLikelihood())}
            );
        } else {
            writer.writeOpenTag(
                    TreeLikelihoodParser.TREE_LIKELIHOOD,
                    new Attribute[]{
                            new Attribute.Default<String>(XMLParser.ID, partition.getPrefix() + id),
                            new Attribute.Default<Boolean>(TreeLikelihoodParser.USE_AMBIGUITIES, substModel.isUseAmbiguitiesTreeLikelihood())}
            );
        }

        if (!options.samplePriorOnly) {
            if (num > 0) {
                writeCodonPatternsRef(substModel.getPrefix(num) + partition.getPrefix(), num, substModel.getCodonPartitionCount(), writer);
            } else {
                writer.writeIDref(SitePatternsParser.PATTERNS, partition.getPrefix() + SitePatternsParser.PATTERNS);
            }
        } else {
            // We just need to use the dummy alignment
            writer.writeIDref(AlignmentParser.ALIGNMENT, partition.getAlignment().getId());
        }

        writer.writeIDref(TreeModel.TREE_MODEL, treeModel.getPrefix() + TreeModel.TREE_MODEL);

        if (num > 0) {
            writer.writeIDref(GammaSiteModel.SITE_MODEL, substModel.getPrefix(num) + SiteModel.SITE_MODEL);
        } else {
            writer.writeIDref(GammaSiteModel.SITE_MODEL, substModel.getPrefix() + SiteModel.SITE_MODEL);
        }


        switch (clockModel.getClockType()) {
            case STRICT_CLOCK:
                writer.writeIDref(StrictClockBranchRatesParser.STRICT_CLOCK_BRANCH_RATES, clockModel.getPrefix()
                        + BranchRateModel.BRANCH_RATES);
                break;
            case UNCORRELATED:
                writer.writeIDref(DiscretizedBranchRatesParser.DISCRETIZED_BRANCH_RATES, options.noDuplicatedPrefix(clockModel.getPrefix(), treeModel.getPrefix())
                        + BranchRateModel.BRANCH_RATES);
                break;
            case RANDOM_LOCAL_CLOCK:
                writer.writeIDref(RandomLocalClockModelParser.LOCAL_BRANCH_RATES, clockModel.getPrefix()
                        + BranchRateModel.BRANCH_RATES);
                break;

            case AUTOCORRELATED:
                throw new UnsupportedOperationException("Autocorrelated relaxed clock model not implemented yet");
//            	writer.writeIDref(ACLikelihoodParser.AC_LIKELIHOOD, options.noDuplicatedPrefix(clockModel.getPrefix(), treeModel.getPrefix())
//                        + BranchRateModel.BRANCH_RATES);
//                break;

            default:
                throw new IllegalArgumentException("Unknown clock model");
        }

        /*if (options.clockType == ClockType.STRICT_CLOCK) {
            writer.writeIDref(StrictClockBranchRates.STRICT_CLOCK_BRANCH_RATES, BranchRateModel.BRANCH_RATES);
        } else {
           writer.writeIDref(DiscretizedBranchRatesParser.DISCRETIZED_BRANCH_RATES, BranchRateModel.BRANCH_RATES);
        }*/

        writer.writeCloseTag(TreeLikelihoodParser.TREE_LIKELIHOOD);
    }

    public void writeTreeLikelihoodReferences(XMLWriter writer) {
        for (AbstractPartitionData partition : options.dataPartitions) { // Each PD has one TreeLikelihood
            if (partition.getTaxonList() != null) {
                if (partition instanceof PartitionData && partition.getTraits() == null) {
                    // is an alignment data partition
                    PartitionSubstitutionModel substModel = partition.getPartitionSubstitutionModel();
                    if (substModel.getDataType() == Nucleotides.INSTANCE && substModel.getCodonHeteroPattern() != null) {
                        for (int i = 1; i <= substModel.getCodonPartitionCount(); i++) {
                            writer.writeIDref(TreeLikelihoodParser.TREE_LIKELIHOOD, substModel.getPrefix(i) + partition.getPrefix() + TreeLikelihoodParser.TREE_LIKELIHOOD);
                        }
                    } else {
                        writer.writeIDref(TreeLikelihoodParser.TREE_LIKELIHOOD, partition.getPrefix() + TreeLikelihoodParser.TREE_LIKELIHOOD);
                    }
                } else if (partition instanceof PartitionPattern) { // microsat
                    writer.writeIDref(MicrosatelliteSamplerTreeLikelihoodParser.TREE_LIKELIHOOD,
                            partition.getPrefix() + MicrosatelliteSamplerTreeLikelihoodParser.TREE_LIKELIHOOD);
                }
            }
        }
    }

    /**
     * Write the tree likelihood XML block.
     *
     * @param partition the partition  to write likelihood block for
     * @param writer    the writer
     */
    public void writeTreeLikelihood(PartitionPattern partition, XMLWriter writer) {
        PartitionSubstitutionModel substModel = partition.getPartitionSubstitutionModel();
        PartitionTreeModel treeModel = partition.getPartitionTreeModel();
        PartitionClockModel clockModel = partition.getPartitionClockModel();

        writer.writeComment("Microsatellite Sampler Tree Likelihood");

        writer.writeOpenTag(MicrosatelliteSamplerTreeLikelihoodParser.TREE_LIKELIHOOD,
                new Attribute[]{new Attribute.Default<String>(XMLParser.ID,
                        partition.getPrefix() + MicrosatelliteSamplerTreeLikelihoodParser.TREE_LIKELIHOOD)});

        writeMicrosatSubstModelRef(substModel, writer);

        writer.writeIDref(MicrosatelliteSamplerTreeModelParser.TREE_MICROSATELLITE_SAMPLER_MODEL,
                treeModel.getPrefix() + TreeModel.TREE_MODEL + ".microsatellite");

        switch (clockModel.getClockType()) {
            case STRICT_CLOCK:
                writer.writeIDref(StrictClockBranchRatesParser.STRICT_CLOCK_BRANCH_RATES, clockModel.getPrefix()
                        + BranchRateModel.BRANCH_RATES);
                break;
            case UNCORRELATED:
            case RANDOM_LOCAL_CLOCK:
            case AUTOCORRELATED:
                throw new UnsupportedOperationException("Microsatellite only supports strict clock model");

            default:
                throw new IllegalArgumentException("Unknown clock model");
        }

        writer.writeCloseTag(MicrosatelliteSamplerTreeLikelihoodParser.TREE_LIKELIHOOD);
    }

    public void writeMicrosatSubstModelRef(PartitionSubstitutionModel model, XMLWriter writer) {
        if (model.getPhase() != MicroSatModelType.Phase.ONE_PHASE) {
            writer.writeIDref(TwoPhaseModel.TWO_PHASE_MODEL, model.getPrefix() + TwoPhaseModel.TWO_PHASE_MODEL);
        } else if (model.getMutationBias() != MicroSatModelType.MutationalBias.UNBIASED) {
            writer.writeIDref(LinearBiasModel.LINEAR_BIAS_MODEL, model.getPrefix() + LinearBiasModel.LINEAR_BIAS_MODEL);
        } else {
            writer.writeIDref(AsymmetricQuadraticModel.ASYMQUAD_MODEL, model.getPrefix() + AsymmetricQuadraticModel.ASYMQUAD_MODEL);
        }
    }

}
