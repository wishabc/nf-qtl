from pandas_plink import read_plink
import argparse
import numpy as np
import pandas as pd
from scipy import stats as st
import statsmodels.api as sm
import h5py
import time
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

header = [
        "#chr", "start", "end", "chunk_id", "summit",
        'variant_id', 'distance_to_summit',
        'n_samples', 'n_cell_types',
        'n_hom_ref', 'n_het', 'n_hom_alt',
        'f', 'f_pval',
        'b', 'b_se',
        'r2',
        'sse', 'ssr', 'df_model',
    ]


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


def process_snp(snp_phenotypes, snp_genotypes, residualizer):
    genotype_residuals = residualizer.transform(snp_genotypes.T).T
    phenotype_residuals = residualizer.transform(snp_phenotypes.T).T
    design = np.concatenate([genotype_residuals], axis=1) # add predictors here
    
    res = sm.OLS(phenotype_residuals, design).fit()

    bse = res.bse[0] * np.sqrt(design.shape[0] - 1) / np.sqrt(design.shape[0] - 1 - residualizer.n)
    f = (res.ess / (1 + residualizer.n)) / (res.ssr / (design.shape[0] - 1 - residualizer.n))
    f_pval = st.f.sf(f, dfn=1 + residualizer.n, dfd=design.shape[0] - 1 - residualizer.n)
    
    n_hom_ref, n_het, n_hom_alt = np.unique(snp_genotypes, return_counts=True)[1]

    return [
        design.shape[0], # samples tested
        np.nan, # cell types
        n_hom_ref, n_het, n_hom_alt,
        f, f_pval,
        res.params[0], bse,
        res.rsquared, res.ess, res.ssr, 1 + residualizer.n
    ]


def process_dhs(phenotype_matrix, genotype_matrix, samples_per_snp, dhs_residualizers, snps_data, dhs_data):
    result = []
    dhs_data_as_list = dhs_data.to_list()
    for snp_index, genotypes in enumerate(genotype_matrix):
        # genotypes = genotype_matrix[snp_index, :]
        #snp_used_samples_index = (genotypes != -1) & (~np.isnan(phenotypes))
        valid_samples = samples_per_snp[snp_index]
        snp_genotypes = genotypes[valid_samples].reshape(-1, 1)
        snp_phenotypes = phenotype_matrix[valid_samples].reshape(-1, 1)
        residualizer = dhs_residualizers[snp_index]
        snp_stats = process_snp(snp_phenotypes=snp_phenotypes, 
            snp_genotypes=snp_genotypes, residualizer=residualizer)
        snp_id, snp_pos = snps_data.iloc[snp_index][['variant_id', 'pos']]
        result.append([
            *dhs_data_as_list,
            snp_id,
            snp_pos - dhs_data['summit'], # dist to "TSS"
            *snp_stats
        ])
    return result


def main(phenotype_matrix, snps_per_dhs,
    genotype_matrix, valid_samples, residualizers,
    snps_data, dhs_data):
    # optionally do in parallel
    result = []
    for dhs_idx, snps_indices in enumerate(tqdm(snps_per_dhs)):
        # subsetting matrices
        phenotype = phenotype_matrix[dhs_idx, :].squeeze()
        genotypes = genotype_matrix[snps_indices, :]
        samples_per_snp = valid_samples[snps_indices, :]
        dhs_residualizers = residualizers[snps_indices]
        current_dhs_data = dhs_data.iloc[dhs_idx]
        current_snps_data = snps_data.iloc[snps_indices]
        stats = process_dhs(phenotype_matrix=phenotype,
            genotype_matrix=genotypes,
            samples_per_snp=samples_per_snp, 
            dhs_residualizers=dhs_residualizers,
            snps_data=current_snps_data,
            dhs_data=current_dhs_data
            )
        result.extend(stats)

    return pd.DataFrame(result, columns=header)
        
def find_testable_snps(gt_matrix, min_snps, gens=2):
    # todo: prettify
    homref = (gt_matrix == 0).sum(axis=1)
    het = (gt_matrix == 1).sum(axis=1)
    homalt = (gt_matrix == 2).sum(axis=1)
    return ((homref >= min_snps).astype(np.int8) 
        + (het >= min_snps).astype(np.int8) 
        + (homalt >= min_snps).astype(np.int8)) >= gens


def find_snps_per_dhs(phenotype_df, variant_df, window):
    phenotype_len = len(phenotype_df.index)
    result = np.zeros((phenotype_len, len(variant_df.index)), dtype=bool)
    invalid_phens_indices = []
    unique_chrs = variant_df['chrom'].unique()
    per_chr_groups = None
    if len(unique_chrs) > 1:
        per_chr_groups = variant_df.reset_index().groupby('chrom')
    for phen_idx, row in phenotype_df.iterrows():
        chrom = row['#chr']
        if per_chr_groups is None:
            chr_df = variant_df.reset_index()
        else:
            chr_df = per_chr_groups.get_group(chrom)
        snp_positions = chr_df['pos'].to_numpy()

        lower_bound = np.searchsorted(snp_positions, row['start'] + 1 - window)
        upper_bound = np.searchsorted(snp_positions, row['end'] + window, side='right')
        if lower_bound != upper_bound:
            snps_indices = chr_df['index'].to_numpy()[lower_bound:upper_bound - 1] # returns one just before
            result[phen_idx, snps_indices] = True
        else:
            invalid_phens_indices.append(phen_idx)  

    if len(invalid_phens_indices) != 0:
        print(f'** dropping {len(invalid_phens_indices)} phenotypes without variants in cis-window')
    invalid_phens_mask = np.zeros(phenotype_len, dtype=bool)
    invalid_phens_mask[invalid_phens_indices] = True
    return result, invalid_phens_mask


def preprocess_data():
    # TODO: move preprocessing here
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run QTL regression')
    parser.add_argument('chunk_id', help='Path to normalized phenotype matrix, numpy format')
    parser.add_argument('metadata', help='Path to metadata file with ag_id and indiv_id columns.' +
        'Should contain cell_type column if run in cell_spec mode')
    parser.add_argument('phenotype_matrix', help='Path to normalized phenotype matrix, numpy format')
    parser.add_argument('mask', help='Path to nan masked DHSs')
    parser.add_argument('index_file', help='Path to file with rows identificators of phenotype matrix')
    parser.add_argument('samples_order', help='Path to file with columns identificators (sample_ids) of phenotype matrix')
    parser.add_argument('plink_prefix', help='Plink prefix to read file with plink_pandas package')
    parser.add_argument('outpath', help='Path to fasta file with SNPs coded as IUPAC symbols')
    parser.add_argument('--cell_spec', help='Specify to do cell-specifc caQTL analysis',
        default=False, action="store_true")

    args = parser.parse_args()

    t = time.perf_counter()
    ## ---- Read data for specific region only -----
    chrom, start, end = unpack_region(args.chunk_id)

    ## ---- Read phenotype data -------
    masterlist = pd.read_table(args.index_file,     
        names=["#chr", "start", "end", "chunk_id", "score", "n_samples",
        "n_peaks", "dhs_width", "summit", "start_core", "end_core", "avg_score"],
        header=None)
    non_nan_mask = np.loadtxt(args.mask, dtype=bool)
    masterlist = masterlist.iloc[non_nan_mask]
    dhs_chunk_idx = ((masterlist['#chr'] == chrom) 
        & (masterlist['start'] >= start) 
        & (masterlist['end'] < end)).to_numpy().astype(bool)
    
    masterlist = masterlist.iloc[dhs_chunk_idx].reset_index(drop=True)

    with h5py.File(args.phenotype_matrix, 'r') as f:
        phenotype_data = f['normalized_counts'][dhs_chunk_idx, :] # [DHS x samples]
    assert (~np.isfinite(phenotype_data)).sum() == 0
    
    # read samples order in the normalized matrix
    with open(args.samples_order) as f:
        samples_order = np.array(f.readline().strip().split('\t'))

    ## ---- Read genotype data in a bigger window ------
    window = 500_000
    bim, fam, bed = read_plink(args.plink_prefix)
    bed = 2 - bed
    bed[np.isnan(bed)] = -1
    bed = bed.astype(np.int8, copy=False)
    # pos is 1-based, start is 0-based
    snps_index = bim.eval(f'chrom == "{chrom}" & pos >= {start - window + 1} & pos < {end + window}').to_numpy().astype(bool)
    bed = bed[snps_index, :].compute()
    
    # filter SNPs with homref, homalt and hets present
    testable_snps = find_testable_snps(bed, min_snps=3, gens=3)
    bed = bed[testable_snps, :] # [SNPs x indivs]

    bim = bim.iloc[snps_index].iloc[testable_snps].reset_index(drop=True)
    # use eval instead?
    if bim.empty:
        print(f'No SNPs passing filters found for {args.chunk_id}, exiting')
        with open(args.outpath, 'w') as f:
            f.write('\t'.join(header))
        exit(0)
    bim['variant_id'] = bim.apply(
        lambda row: f"{row['chrom']}_{row['pos']}_{row['snp']}_{row['a0']}_{row['a1']}",
        axis=1
    )
    fam.rename(columns={'iid': 'indiv_id'}, inplace=True)
    
    ## ----------- Find snps per DHS ---------
    snps_per_dhs, invalid_phens = find_snps_per_dhs(masterlist, bim, window=window) # [DHS x SNPs] boolean matrix, [DHS] boolean vector
    phenotype_data = phenotype_data[~invalid_phens, :]
    snps_per_dhs = snps_per_dhs[~invalid_phens, :]
    masterlist = masterlist.iloc[~invalid_phens, :].reset_index(drop=True)
    print('SNP-DHS pairs -', snps_per_dhs.sum())
    print('DHS with > 2 SNPs -', (snps_per_dhs.sum(axis=1) > 2).sum())
    ## --------- Read indiv to sample correspondence ----------
    metadata = pd.read_table(args.metadata)
    ## Check if all required columns present
    is_cell_specific = args.cell_spec
    req_cols = ['ag_id', 'indiv_id']
    if is_cell_specific:
        req_cols.append('CT')
    try:
        assert set(req_cols).issubset(metadata.columns)
    except Exception as e:
        print(f'{req_cols} not in {metadata.columns}')
    ordered_meta = metadata[req_cols].merge(
        fam['indiv_id'].reset_index()
    ).set_index('ag_id').loc[samples_order, :]
    difference = len(metadata.index) - len(ordered_meta.index)
    if difference != 0:
        print(f'{difference} samples has been filtered out!')

    indiv2samples_idx = ordered_meta['index'].to_numpy()
    cell_types = ordered_meta['CT'].to_numpy() # cell_types enumerated by sample_index
    ohe_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ohe_cell_types = ohe_enc.fit_transform(cell_types.reshape(-1, 1))
    # transform genotype matrix from [SNP x indiv] to [SNP x sample] format
    bed = bed[:, indiv2samples_idx]
    
    ## --------- Read covariates data ---------
    covariates = pd.read_table(f'{args.plink_prefix}.eigenvec')
    assert covariates['IID'].tolist() == fam['indiv_id'].tolist()

    sample_pcs = covariates.loc[indiv2samples_idx].iloc[:, 2:].to_numpy()
    # calc residualizer for each variant
    valid_samples = (bed != -1) # [SNPs x samples]
    covariates_np = np.concatenate([sample_pcs, ohe_cell_types], axis=1)
    residualizers = np.array([Residualizer(covariates_np[snp_samples_idx, :]) 
        for snp_samples_idx in valid_samples]) # len(SNPs), add covariates here

    print(f"Preprocessing finished in {time.perf_counter() - t}s")
    ## ------------ Run regressions -----------
    result = main(phenotype_matrix=phenotype_data, 
        snps_per_dhs=snps_per_dhs,
        genotype_matrix=bed,
        valid_samples=valid_samples,
        residualizers=residualizers,
        snps_data=bim[['variant_id', 'pos']],
        dhs_data=masterlist[["#chr", "start", "end", "chunk_id", "summit"]]
    )
    print(f"Processing finished in {time.perf_counter() - t}")
    result.to_csv(args.outpath, sep='\t', index=False)
