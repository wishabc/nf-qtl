from pandas_plink import read_plink
import argparse
import numpy as np
import pandas as pd


class Residualizer:
    def __init__(self, C):
        # center and orthogonalize
        self.Q, _ = np.linalg.qr(C - C.mean(0))
        self.dof = C.shape[0] - C.shape[1]
        self.n = C.shape[1]

    def transform(self, M, center=True):
        """Residualize rows of M wrt columns of C"""
        M0 = M - M.mean(1, keepdims=True)
        if center:
            M0 = M0 - np.matmul(np.matmul(M0, self.Q), self.Q.T)
        else:
            M0 = M - np.matmul(np.matmul(M0, self.Q), self.Q.T)
        return M0

def unpack_region(s):
    chrom, coords = s.split(":")
    start, end = coords.split("-")
    return chrom, int(start), int(end)

def process_dhs():
    rows = []
    for snp_index in range(genotype_matrix.shape[0]):
        current_snp = snp_info.loc[snp_index]
        
        genotypes = genotype_matrix[snp_index, :]
        snp_used_samples_index = (genotypes != -1) & (~np.isnan(phenotypes))
        snp_ag_ids_used = samples_order[snp_used_samples_index]
        snp_indivs_used = ag_to_indiv.loc[snp_ag_ids_used, 'iid']
        snp_indivs_used = snp_indivs_used.to_numpy().reshape(-1, 1)
        snp_genotypes = genotypes[snp_used_samples_index].reshape(-1, 1)
        snp_pcs = sample_pcs[snp_used_samples_index, :] # add covariates here
        snp_phenotypes = phenotypes[snp_used_samples_index].reshape(-1, 1)
        
        genotype_residuals = residualizer.transform(snp_genotypes.T).T
        phenotype_residuals = residualizer.transform(snp_phenotypes.T).T
        design = np.concatenate([genotype_residuals], axis=1) # add predictors here
        
        hr, ht, ha = np.unique(snp_genotypes, return_counts=True)[1]
        indiv_genotypes = bed[snps_index, :][snp_index, :]
        hr_ind, ht_ind, ha_ind = np.unique(indiv_genotypes[np.unique(indices_order[snp_used_samples_index])],
                                        return_counts=True)[1]
        
        ols = sm.OLS(phenotype_residuals, design)
        res = ols.fit()
        
        bse = res.bse[0] * np.sqrt(design.shape[0] - 1) / np.sqrt(design.shape[0] - 1 - residualizer.n)
        f = (res.ess / (1 + residualizer.n)) / (res.ssr / (design.shape[0] - 1 - residualizer.n))
        f_pval = st.f.sf(f, dfn=1 + residualizer.n, dfd=design.shape[0] - 1 - residualizer.n)
        
        rows.append([
            dhs['chunk_id'],
            current_snp['variant_id'],
            current_snp['pos'] - dhs['summit'],
            (current_snp['pos'] - 1 < dhs['end_core']) & (current_snp['pos'] - 1 >= dhs['start_core']),
            (current_snp['pos'] - 1 < dhs['end']) & (current_snp['pos'] - 1 >= dhs['start']),
            design.shape[0],
            len(np.unique(snp_indivs_used)),
            np.nan, #cell types
            hr, ht, ha,
            hr_ind, ht_ind, ha_ind,
            f, f_pval,
            res.params[0],
            res.bse[0] * np.sqrt(design.shape[0] - 1) / np.sqrt(design.shape[0] - 1 - residualizer.n),
            res.rsquared,
            res.ess,
            res.ssr,
            1 + residualizer.n,
        ])
def main(phenotype_matrix, index_file, samples_order, plink_files):
    
    
    for dhs in dhss:
        phenotypes = normalized_counts[current_dhs.index, :].squeeze()
        process_dhs()

        
def get_testable_snps(gt_matrix, min_snps, gens=2):
    # todo: prettify
    homref = (gt_matrix == 0).sum(axis=1)
    het = (gt_matrix == 1).sum(axis=1)
    homalt = (gt_matrix == 2).sum(axis=1)
    return ((homref >= min_snps).astype(np.int8) 
        + (het >= min_snps).astype(np.int8) 
        + (homalt >= min_snps).astype(np.int8)) >= gens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run QTL regression')
    parser.add_argument('chunk_id', help='Path to normalized phenotype matrix, numpy format')
    parser.add_argument('metadata', help='Path to metadata file with ag_id and indiv_id columns')
    parser.add_argument('phenotype_matrix', help='Path to normalized phenotype matrix, numpy format')
    parser.add_argument('index_file', help='Path to file with rows identificators of phenotype matrix')
    parser.add_argument('samples_order', help='Path to file with columns identificators (sample_ids) of phenotype matrix')
    parser.add_argument('plink prefix', help='Plink prefix to read file with plink_pandas package')
    parser.add_argument('outpath', help='Path to fasta file with SNPs coded as IUPAC symbols')
    args = parser.parse_args()

    ## ---- Read data for specific region only -----
    chrom, start, end = unpack_region(args.chunk_id)

    ## ---- Read phenotype data -------
    masterlist = pd.read_table(args.index_file,     
        names=["chr", "start", "end", "chunk_id", "score", "n_samples",
        "n_peaks", "dhs_width", "summit", "start_core", "end_core", "avg_score"],
        header=None)
    dhs_chunk_idx = ((masterlist['#chr'] == chrom) 
        & (masterlist['start'] >= start) 
        & (masterlist['end'] < end)).to_numpy().astype(bool)
    
    with open(args.samples_order) as f:
        samples_order = np.array(f.readline().strip().split('\t'))
    # fixme
    phenotype_data = np.load(args.phenotype_matrix)[dhs_chunk_idx, :] # DHS x samples
    assert (~np.isfinite(phenotype_data)).sum() == 0
    ## ---- Read genotype data in a bigger window ------
    window = 500_000
    bed, bim, fam = read_plink(args.plink_prefix)
    bed = 2 - bed
    bed[np.isnan(bed)] = -1
    bed = bed.astype(np.int8, copy=False)
    # pos is 1-based, start is 0-based
    snps_index = bim.eval(f'chrom == {chrom} & pos >= {start + window + 1} & pos < {end + window}').to_numpy().astype(bool)
    bed = bed[snps_index, :].compute()
    
    # filter SNPs with homref, homalt and hets present
    testable_snps = get_testable_snps(bed, snps=3, gens=3)
    bed = bed[testable_snps, :] # SNPs x indivs

    bim = bim[snps_index & testable_snps].reset_index(drop=True)
    # use eval instead?
    bim['variant_id'] = bim.apply(
        lambda row: f"{row['chrom']}_{row['pos']}_{row['snp']}_{row['a0']}_{row['a1']}",
        axis=1
    )
    fam.rename(columns={'iid': 'indiv_id'}, inplace=True)

    ## ---- Read indiv to sample correspondence -----
    indiv2samples_idx = pd.read_table(args.metadata)[['ag_id', 'indiv_id']].merge(
        fam['indiv_id'].reset_index()
    ).set_index('ag_id').loc[samples_order, 'index'].to_numpy()
    # transform genotype matrix
    bed = bed[:, indiv2samples_idx]
    
    ## ---- Read covariates data ----
    sample_pcs = pd.read_table(
            f'{args.plink_prefix}.eigenvec'
        ).loc[indiv2samples_idx].iloc[:, 2:].to_numpy()
    # calc residualizer for each variant
    valid_samples = (bed != -1) # [SNPs x samples]
    residualizers = [Residualizer(sample_pcs[snp_samples_idx, :]) 
        for snp_samples_idx in valid_samples] # len(SNPs)

    
    read_covars = pd.read_table()
    result = main(phenotype_data, samples_order, plink_dasks)



#	python3 $moduleDir/bin/qtl_regression.py '${chunk_id}' \
#		${normalized_matrix} \
#		${params.index_file} \
#		${params.indivs_order} \
#		${plink_prefix} \
#		${name}