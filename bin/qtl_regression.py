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
    'n_samples',
    'n_hom_ref', 'n_het', 'n_hom_alt',
    'ss_model', 'ss_residuals', 'df_model', 'df_residuals'
]


class Residualizer:
    def __init__(self, C):
        # center and orthogonalize
        self.n = np.linalg.matrix_rank(C)
        self.dof = C.shape[0] - self.n
        if self.dof == 0:
            self.Q = None
        else:
            M, _ = remove_redundant_columns(C) # to make qr more stable
            self.Q, _ = np.linalg.qr(M - M.mean(axis=0))

        

    def transform(self, M, center=True):
        """Residualize rows of M wrt columns of C"""
        if self.Q is None:
            return None
        M0 = M - M.mean(1, keepdims=True)
        if center:
            M0 = M0 - np.matmul(np.matmul(M0, self.Q), self.Q.T)
        else:
            M0 = M - np.matmul(np.matmul(M0, self.Q), self.Q.T)
        return M0


class QTLmapper:
    def __init__(self, phenotype_matrix, snps_per_phenotype,
                 genotype_matrix, samples_per_snps, residualizers,
                 snps_data, dhs_data, is_cell_specific=False,
                 include_interaction=False, cell_type_data=None):

        self.phenotype_matrix = phenotype_matrix
        self.snps_per_phenotype = snps_per_phenotype
        self.genotype_matrix = genotype_matrix
        self.samples_per_snps = samples_per_snps

        self.residualizers = residualizers

        self.snps_data = snps_data
        self.dhs_data = dhs_data

        self.is_cell_specific = is_cell_specific
        self.cell_type_data = cell_type_data
        self.include_interaction = is_cell_specific and include_interaction

    @staticmethod
    def fit_regression(X, Y, df_model, df_residuals):
        XtX = np.matmul(np.transpose(X), X)
        XtY = np.matmul(np.transpose(X), Y)
        XtXinv = np.linalg.inv(XtX)
        coeffs = np.matmul(XtXinv, XtY)

        Y_predicted = np.matmul(X, coeffs)

        # sum of squares 
        ss_residuals = np.square(Y - Y_predicted).sum(0)
        ss_model = np.square(Y_predicted - Y.mean(0, keepdims=True)).sum(0)
        
        # mean sum of squares
        ms_residuals = ss_residuals / df_residuals
        # coeffs standard error
        coeffs_se = np.sqrt(XtXinv[np.eye(X.shape[1], dtype=bool)][..., None] * ms_residuals)

        return [ss_model, ss_residuals, df_model, df_residuals], [coeffs[0, :], coeffs_se[0, :]]

    def process_snp(self, snp_phenotypes, snp_genotypes, residualizer):
        design = residualizer.transform(snp_genotypes.T).T
        phenotype_residuals = residualizer.transform(snp_phenotypes.T).T
        n_hom_ref, n_het, n_hom_alt = np.unique(snp_genotypes, return_counts=True)[1]
        df_model = design.shape[1]
        df_residuals = design.shape[0] - df_model - residualizer.n
        snp_stats, coeffs = self.fit_regression(design, phenotype_residuals, df_model, df_residuals)

        return [
            design.shape[0],  # samples tested
            n_hom_ref, n_het, n_hom_alt,
            *snp_stats # coeffs, coeffs_se, ss_model, ss_residuals, df_model, df_residals
        ], coeffs

    def process_dhs(self, phenotype_matrix, genotype_matrix, samples_per_snp,
                    dhs_residualizers, snps_data, dhs_data):
        res = []
        coeffs_res = []
        dhs_data_as_list = dhs_data.to_list()
        for snp_index, genotypes in enumerate(genotype_matrix):
            valid_samples = samples_per_snp[snp_index]
            snp_genotypes = genotypes[valid_samples][:, None]  # [samples x 1]
            residualizer = dhs_residualizers[snp_index]
            if residualizer.Q is None:
                continue
            if self.include_interaction:
                # calculate interaction
                interaction = snp_genotypes * self.cell_type_data[valid_samples, :]  # [samples x cell_types]
                snp_genotypes, valid_design_cols_mask = remove_redundant_columns(
                    np.concatenate([snp_genotypes, interaction], axis=1))
                valid_design_cols_indices = np.where(valid_design_cols_mask)
            else:
                valid_design_cols_indices = np.ones(1)
            if snp_genotypes.shape[0] - snp_genotypes.shape[1] - residualizer.n < 1:
                continue

            snp_phenotypes = phenotype_matrix[valid_samples][:, None]  # [samples x 1]

            snp_stats, coeffs = self.process_snp(snp_phenotypes=snp_phenotypes,
                                         snp_genotypes=snp_genotypes,
                                         residualizer=residualizer)


            snp_id, snp_pos = snps_data.iloc[snp_index][['variant_id', 'pos']]

            to_add = np.repeat(np.array([snp_id, dhs_data['chunk_id']])[None, ...],
                valid_design_cols_indices.shape[0], axis=0)
            stack = np.stack([valid_design_cols_indices, *coeffs]).T
            
            coeffs_res.append(np.concatenate([to_add, stack], axis=1))

            res.append([
                *dhs_data_as_list,
                snp_id,
                snp_pos - dhs_data['summit'],  # dist to "TSS"
                *snp_stats
            ])
        return res, coeffs_res

    @staticmethod
    def post_processing(df):
        # Do in vectorized manner
        df['f_stat'] = df.eval('(ss_model / df_model) / (ss_residuals / df_residuals)')
        df['log_f_pval'] = -st.f.logsf(df['f_stat'], dfn=df['df_model'], dfd=df['df_residuals'])
        df['minor_allele_count'] = df[['n_hom_ref', 'n_hom_alt']].min(axis=1) * 2 + df['n_het']
        return df


    def map_qtl(self):
        # optionally do in parallel
        stats_res = []
        coefs_res = []
        for dhs_idx, snps_indices in enumerate(tqdm(self.snps_per_phenotype)):
            # sub-setting matrices
            phenotype = np.squeeze(self.phenotype_matrix[dhs_idx, :])
            genotypes = self.genotype_matrix[snps_indices, :]
            samples_per_snp = self.samples_per_snps[snps_indices, :]
            dhs_residualizers = self.residualizers[snps_indices]
            current_dhs_data = self.dhs_data.iloc[dhs_idx]
            current_snps_data = self.snps_data.iloc[snps_indices]

            stats, coefs = self.process_dhs(phenotype_matrix=phenotype,
                                     genotype_matrix=genotypes,
                                     samples_per_snp=samples_per_snp,
                                     dhs_residualizers=dhs_residualizers,
                                     snps_data=current_snps_data,
                                     dhs_data=current_dhs_data,
                                     )
            stats_res.extend(stats)
            coefs_res.extend(coefs)

        stats_res = pd.DataFrame(stats_res, columns=header)
        coefs_res = pd.DataFrame(np.concatenate(coefs_res), 
            columns=['variant_id', 'chunk_id', 'cell_type_idx', 'coeff', 'coeff_se'])
        return self.post_processing(stats_res), coefs_res


def find_testable_snps(gt_matrix, min_snps, gens=2, ma_frac=0.05):
    # todo: prettify
    homref = (gt_matrix == 0).sum(axis=1)
    het = (gt_matrix == 1).sum(axis=1)
    homalt = (gt_matrix == 2).sum(axis=1)
    enough_gens = ((homref >= min_snps).astype(np.int8)
            + (het >= min_snps).astype(np.int8)
            + (homalt >= min_snps).astype(np.int8)) >= gens
    ma_passing = np.minimum(homref, homalt) * 2  + het >= ma_frac * gt_matrix.shape[1]
    return ma_passing * enough_gens


def find_snps_per_dhs(phenotype_df, variant_df, window):
    phenotype_len = len(phenotype_df.index)
    snps_per_dhs = np.zeros((phenotype_len, len(variant_df.index)), dtype=bool)
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

        # Change start/end to summit if needed
        lower_bound = np.searchsorted(snp_positions, row['summit'] + 1 - window)
        upper_bound = np.searchsorted(snp_positions, row['summit'] + window, side='right')
        if lower_bound != upper_bound:
            snps_indices = chr_df['index'].to_numpy()[lower_bound:upper_bound - 1]  # returns one just before
            snps_per_dhs[phen_idx, snps_indices] = True
        else:
            invalid_phens_indices.append(phen_idx)

    if len(invalid_phens_indices) != 0:
        print(f'** dropping {len(invalid_phens_indices)} phenotypes without variants in cis-window')
    invalid_phens_mask = np.zeros(phenotype_len, dtype=bool)
    invalid_phens_mask[invalid_phens_indices] = True
    return snps_per_dhs, invalid_phens_mask


def unpack_region(s):
    chrom, coords = s.split(":")
    start, end = coords.split("-")
    return chrom, int(start), int(end)


def remove_redundant_columns(matrix):
    cols_mask = np.any(matrix != 0, axis=0)
    return matrix[:, cols_mask], cols_mask


def n_unique_last_axis(matrix):
    a_s = np.sort(matrix, axis=-1)  # [SNP x cell_type x sample]
    return matrix.shape[-1] - ((a_s[..., :-1] == a_s[..., 1:]) | (a_s[..., :-1] == 0)).sum(axis=-1)


def find_valid_samples(genotypes, cell_types, threshold=2):
    # cell_types - [cell_type x sample]
    # genotypes # [SNP x sample]
    gen_pseudo = (genotypes + 1)[:, None, :]  # [SNP x 1 x sample]
    res = np.squeeze(n_unique_last_axis(cell_types[None, :, :] * gen_pseudo) >= threshold)  # [SNP x cell_type]
    return (np.matmul(res, cell_types) * (genotypes != -1)).astype(bool)  # [SNP x sample]


# Too large function! TODO: move preprocessing to smaller functions
def main(chunk_id, masterlist_path, non_nan_mask_path, phenotype_matrix_path,
         samples_order_path, plink_prefix, metadata_path, outpath, is_cell_specific=False,
         include_interaction=False):
    t = time.perf_counter()
    print('Processing started')
    # ---- Read data for specific region only -----
    chrom, start, end = unpack_region(chunk_id)

    # ---- Read phenotype data -------
    masterlist = pd.read_table(masterlist_path,
                               names=["#chr", "start", "end", "chunk_id", "score", "n_samples",
                                      "n_peaks", "dhs_width", "summit", "start_core", "end_core", "avg_score"],
                               header=None)
    non_nan_mask = np.loadtxt(non_nan_mask_path, dtype=bool)
    masterlist = masterlist.iloc[non_nan_mask]
    # Change summit to start/end if needed
    dhs_chunk_idx = ((masterlist['#chr'] == chrom)
                     & (masterlist['summit'] >= start)
                     & (masterlist['summit'] < end)).to_numpy().astype(bool)

    masterlist = masterlist.iloc[dhs_chunk_idx].reset_index(drop=True)

    with h5py.File(phenotype_matrix_path, 'r') as f:
        phenotype_data = f['normalized_counts'][dhs_chunk_idx, :]  # [DHS x samples]
    assert (~np.isfinite(phenotype_data)).sum() == 0

    # read samples order in the normalized matrix
    with open(samples_order_path) as f:
        samples_order = np.array(f.readline().strip().split('\t'))

    # ---- Read genotype data in a bigger window ------
    window = 500_000
    allele_frac = 0.05

    bim, fam, bed = read_plink(plink_prefix)
    bed = 2 - bed
    bed[np.isnan(bed)] = -1
    bed = bed.astype(np.int8, copy=False)
    # pos is 1-based, start is 0-based
    snps_index = bim.eval(f'chrom == "{chrom}" & pos >= {start - window + 1}'
                          f' & pos < {end + window}').to_numpy().astype(bool)
    bed = bed[snps_index, :].compute()

    # filter SNPs with homref, homalt and hets present
    testable_snps = find_testable_snps(bed, min_snps=3, gens=3, ma_frac=allele_frac)
    bed = bed[testable_snps, :]  # [SNPs x indivs]

    bim = bim.iloc[snps_index].iloc[testable_snps].reset_index(drop=True)
    # use eval instead?
    if bim.empty:
        print(f'No SNPs passing filters found for {chunk_id}, exiting')
        with open(outpath, 'w') as f:
            f.write('\t'.join(header))
        exit(0)
    bim['variant_id'] = bim.apply(
        lambda row: f"{row['chrom']}_{row['pos']}_{row['snp']}_{row['a0']}_{row['a1']}",
        axis=1
    )
    fam.rename(columns={'iid': 'indiv_id'}, inplace=True)

    # ----------- Find snps per DHS ---------
    snps_per_dhs, invalid_phens = find_snps_per_dhs(masterlist, bim,
                                                    window=window)  # [DHS x SNPs] boolean matrix, [DHS] boolean vector
    phenotype_data = phenotype_data[~invalid_phens, :]
    snps_per_dhs = snps_per_dhs[~invalid_phens, :]  # [DHS x SNPs] boolean matrix
    masterlist = masterlist.iloc[~invalid_phens, :].reset_index(drop=True)
    print('SNP-DHS pairs -', snps_per_dhs.sum())
    print('DHS with > 2 SNPs -', (snps_per_dhs.sum(axis=1) > 2).sum())

    # --------- Read indiv to sample correspondence ----------
    metadata = pd.read_table(metadata_path)

    # Check if all required columns present
    req_cols = ['ag_id', 'indiv_id']
    if is_cell_specific:
        req_cols.append('CT')
    try:
        assert set(req_cols).issubset(metadata.columns)
    except Exception as e:
        print(f'{req_cols} not in {metadata.columns}')
        raise e
    ordered_meta = metadata[req_cols].merge(
        fam['indiv_id'].reset_index()
    ).set_index('ag_id').loc[samples_order, :]
    difference = len(metadata.index) - len(ordered_meta.index)
    if difference != 0:
        print(f'{difference} samples has been filtered out!')

    indiv2samples_idx = ordered_meta['index'].to_numpy()

    # --------- Read covariates data ---------
    covariates = pd.read_table(f'{plink_prefix}.eigenvec')
    assert covariates['IID'].tolist() == fam['indiv_id'].tolist()

    sample_pcs = covariates.loc[indiv2samples_idx].iloc[:, 2:].to_numpy()

    # transform genotype matrix from [SNP x indiv] to [SNP x sample] format
    bed = bed[:, indiv2samples_idx]
    if is_cell_specific:
        cell_types = ordered_meta['CT'].to_numpy()  # cell_types enumerated by sample_index
        ohe_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe_cell_types = ohe_enc.fit_transform(cell_types.reshape(-1, 1))

        # Filter out cell-types with less than 2 distinct genotypes
        valid_samples = find_valid_samples(bed, ohe_cell_types.T, 3)  # [SNPs x samples]
        before_n = (bed != -1).sum()
        bed[~valid_samples] = -1
        testable_snps = find_testable_snps(bed, min_snps=3, gens=3, ma_frac=allele_frac)
        bed = bed[testable_snps, :]  # [SNPs x indivs]
        snps_per_dhs = snps_per_dhs[:, testable_snps]  # [DHS x SNPs] boolean matrix
        valid_samples = valid_samples[testable_snps, :]
        print(f"SNPxDHS pairs. Before: {before_n}, after: {valid_samples.sum()}")
        bim = bim.iloc[testable_snps, :].reset_index(drop=True)
        print('DHS with > 2 SNPs -', (snps_per_dhs.sum(axis=1) > 2).sum())
        covariates_np = np.concatenate([sample_pcs, ohe_cell_types], axis=1) # [sample x covariate]
    else:
        valid_samples = (bed != -1)  # [SNPs x samples]
        ohe_cell_types = None
        covariates_np = sample_pcs

    # calc residualizer for each variant
    residualizers = np.array([Residualizer(covariates_np[snp_samples_idx, :])
                              for snp_samples_idx in tqdm(valid_samples)])  # [SNPs x covariates]
    print(f"Preprocessing finished in {time.perf_counter() - t}s")
    # ------------ Run regressions -----------
    qtl_mapper = QTLmapper(
        phenotype_matrix=phenotype_data,
        snps_per_phenotype=snps_per_dhs,
        genotype_matrix=bed,
        samples_per_snps=valid_samples,
        residualizers=residualizers,
        snps_data=bim[['variant_id', 'pos']],
        dhs_data=masterlist[["#chr", "start", "end", "chunk_id", "summit"]],
        cell_type_data=ohe_cell_types,
        is_cell_specific=is_cell_specific,
        include_interaction=include_interaction
    )
    res, coefs = qtl_mapper.map_qtl()
    print(f"Processing finished in {time.perf_counter() - t}s")
    return res, coefs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run QTL regression')
    parser.add_argument('chunk_id', help='Path to normalized phenotype matrix, numpy format')
    parser.add_argument('metadata', help='Path to metadata file with ag_id and indiv_id columns.' +
                                         'Should contain cell_type column if run in cell_spec mode')
    parser.add_argument('phenotype_matrix', help='Path to normalized phenotype matrix, numpy format')
    parser.add_argument('mask', help='Path to nan masked DHSs')
    parser.add_argument('index_file', help='Path to file with rows identificators of phenotype matrix')
    parser.add_argument('samples_order',
                        help='Path to file with columns identificators (sample_ids) of phenotype matrix')
    parser.add_argument('plink_prefix', help='Plink prefix to read file with plink_pandas package')
    parser.add_argument('outpath', help='Path to fasta file with SNPs coded as IUPAC symbols')
    parser.add_argument('--cell_spec', help='Specify to do cell-specifc caQTL analysis',
                        default=False, action="store_true")
    parser.add_argument('--with_interaction', help='Specify to include cell x genotype interaction in caQTL analysis',
                        default=False, action="store_true")

    args = parser.parse_args()

    result, coefs = main(
        chunk_id=args.chunk_id,
        masterlist_path=args.index_file,
        non_nan_mask_path=args.mask,
        phenotype_matrix_path=args.phenotype_matrix,
        samples_order_path=args.samples_order,
        plink_prefix=args.plink_prefix,
        outpath=args.outpath,
        is_cell_specific=args.cell_spec,
        metadata_path=args.metadata,
        include_interaction=args.with_interaction
    )
    result.to_csv(f'{args.outpath}.result.tsv', sep='\t', index=False)
    coefs.to_csv(f'{args.outpath}.coefs.tsv', sep='\t', index=False)
