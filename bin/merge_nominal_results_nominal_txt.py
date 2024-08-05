#!/usr/bin/env python3
# Author: Francois Aguet

import argparse
import numpy as np
import pandas as pd
import os
import gzip
from datetime import datetime
import subprocess
import io

parser = argparse.ArgumentParser(description='Filter significant SNP-gene pairs from FastQTL results using FDR cutoff')
parser.add_argument('permutation_results', help='FastQTL output')
parser.add_argument('nominal_results', help='FastQTL output from nominal pass')
parser.add_argument('outfile', help='FastQTL output from nominal pass')
parser.add_argument('--fdr', type=np.double, help='False discovery rate (e.g., 0.05)', default=0.05)
# parser.add_argument('annotation_gtf', help='Annotation in GTF format')
# parser.add_argument('--snp_lookup', default='', help='Tab-delimited file with columns: chr, variant_pos, variant_id, ref, alt, num_alt_per_site, rs_id_dbSNP...')
# parser.add_argument('-o', '--output_dir', default='.', help='Output directory')
args = parser.parse_args()

#------------------------------------------------------------------------------
# 1. eGenes (permutation output): add gene and variant information
#------------------------------------------------------------------------------

print('['+datetime.now().strftime("%b %d %H:%M:%S")+'] Annotating permutation results (ePhenotypes)', flush=True)
phenotype_df = pd.read_table(args.permutation_results).set_index('gene_id')


#------------------------------------------------------------------------------
# 2. variant-gene pairs: output new file with all significant pairs
#------------------------------------------------------------------------------

print('['+datetime.now().strftime("%b %d %H:%M:%S")+'] Filtering significant variant-phenotype pairs', flush=True)

# eGenes (apply FDR threshold)
threshold_df = phenotype_df.query(f'qval <= {args.fdr}')[['pval_nominal_threshold', 'pval_nominal', 'pval_beta']].copy()
threshold_df.rename(columns={'pval_nominal': 'min_pval_nominal'}, inplace=True)
phenotype_ids = set(threshold_df.index)
threshold_dict = threshold_df['pval_nominal_threshold'].to_dict()

signif_df = pd.read_table(args.nominal_results)
print('['+datetime.now().strftime("%b %d %H:%M:%S")+']', 'Succesfully read nominal pvalues', signif_df.shape, flush=True)
signif_df = signif_df[signif_df["gene_id"].isin(phenotype_ids)]
signif_df['threshold_nominal_p'] = signif_df['gene_id'].map(threshold_dict)
signif_df.query('pval_nominal < threshold_nominal_p', inplace=True)

print(' * Numbers variant-phenotype pairs tested: {}'.format(signif_df[['gene_id', 'variant_id']].drop_duplicates().shape[0]))
print(' * QTLs @ FDR {}: {}'.format(args.fdr, signif_df.shape[0]))

with gzip.open(args.outfile, 'wt') as f:
    signif_df.to_csv(f, sep='\t', float_format='%.6g', index=False)

print('['+datetime.now().strftime("%b %d %H:%M:%S")+'] Completed annotation', flush=True)