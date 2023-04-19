from pandas_plink import read_plink
import argparse
import numpy as np
import pandas as pd
from scipy import stats as st
import h5py
import time
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

header = [
    "#chr", "start", "end", "chunk_id", "summit",
    'variant_id', 'distance_to_summit',
    'n_samples', 'n_hom_ref', 'n_het', 'n_hom_alt',
    'ss_model', 'ss_residuals', 'df_model', 'df_residuals'
]


class NoDataLeftError(Exception):
    pass


class Residualizer:
    def __init__(self, C):
        # center and orthogonalize
        self.n = np.linalg.matrix_rank(C)
        self.dof = C.shape[0] - self.n
        # debug
        self.C = C
        if self.dof == 0:
            self.Q = None
        else:
            M, _ = remove_redundant_columns(C)  # to make qr more stable
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
    def __init__(self, phenotype_matrix, snps_per_dhs,
                 genotype_matrix, samples_per_snps, residualizers,
                 snps_data, dhs_data, mode, cell_type_data=None):

        self.phenotype_matrix = phenotype_matrix
        self.snps_per_phenotype = snps_per_dhs
        self.genotype_matrix = genotype_matrix
        self.samples_per_snps = samples_per_snps

        self.residualizers = residualizers

        self.snps_data = snps_data
        self.dhs_data = dhs_data

        self.mode = mode
        self.cell_type_data = cell_type_data

        self.singular_matrix_count = 0

    @staticmethod
    def fit_regression(X, Y, df_model, df_residuals):
        XtX = np.matmul(np.transpose(X), X)
        XtY = np.matmul(np.transpose(X), Y)
        XtXinv = np.linalg.inv(XtX)
        coeffs = np.matmul(XtXinv, XtY)

        Y_predicted = np.matmul(X, coeffs)

        # sum of squares
        ss_residuals = np.square(Y - Y_predicted).sum()
        ss_model = np.square(Y_predicted - Y.mean(0, keepdims=True)).sum()

        # mean sum of squares
        ms_residuals = ss_residuals / df_residuals
        if np.any(XtXinv[np.eye(X.shape[1], dtype=bool)][..., None] < 0):
            raise np.linalg.LinAlgError()
        # coeffs standard error
        coeffs_se = np.sqrt(XtXinv[np.eye(X.shape[1], dtype=bool)][..., None] * ms_residuals)

        return [ss_model, ss_residuals, df_model, df_residuals], [coeffs[:, 0], coeffs_se[:, 0]]

    def process_snp(self, snp_phenotypes, snp_genotypes, residualizer):
        design = residualizer.transform(snp_genotypes.T).T
        phenotype_residuals = residualizer.transform(snp_phenotypes.T).T
        n_hom_ref, n_het, n_hom_alt = np.unique(snp_genotypes, return_counts=True)[1]
        df_model = design.shape[1]
        df_residuals = design.shape[0] - df_model - residualizer.n
        snp_stats, coeffs = self.fit_regression(design, phenotype_residuals,
                                                df_model, df_residuals)
        return [design.shape[0],  # samples tested
                n_hom_ref, n_het, n_hom_alt,
                *snp_stats  # coeffs, coeffs_se, ss_model, ss_residuals, df_model, df_residals
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
            if self.mode == 'cell_type':
                # calculate interaction
                interaction = snp_genotypes * self.cell_type_data[valid_samples, :]  # [samples x cell_types]
                snp_genotypes, valid_design_cols_mask = remove_redundant_columns(interaction)
                valid_design_cols_indices = np.where(valid_design_cols_mask)[0]
            else:
                valid_design_cols_indices = np.zeros(1, dtype=int)

            if snp_genotypes.shape[0] - snp_genotypes.shape[1] - residualizer.n < 1:
                continue

            snp_phenotypes = phenotype_matrix[valid_samples][:, None]  # [samples x 1]
            try:
                snp_stats, coeffs = self.process_snp(snp_phenotypes=snp_phenotypes,
                                                     snp_genotypes=snp_genotypes,
                                                     residualizer=residualizer)
            except np.linalg.LinAlgError:
                self.singular_matrix_count += 1
                continue

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
        df['minor_allele_count'] = df[['n_hom_ref', 'n_hom_alt']].min(axis=1) * 2 + df['n_het']
        df['f_stat'] = ((df['ss_model'] / df['df_model']) / (df['ss_residuals'] / df['df_residuals'])).astype(float)
        df['log10_f_pval'] = -st.f.logsf(
            df['f_stat'].to_numpy(),
            dfd=df['df_residuals'].to_numpy(),
            dfn=df['df_model'].to_numpy()) / np.log(10)
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
                                 columns=['variant_id', 'chunk_id', 'design_var_index', 'coeff', 'coeff_se'])
        return self.post_processing(stats_res), coefs_res


def remove_redundant_columns(matrix):
    cols_mask = np.any(matrix != 0, axis=0)
    return matrix[:, cols_mask], cols_mask


class QTLPreprocessing:
    window = 500_000
    allele_frac = 0.05

    def __init__(self, dhs_matrix_path, dhs_masterlist_path, samples_order,
                 plink_prefix, samples_metadata, additional_covariates=None,
                 genomic_region=None, valid_dhs=None, mode='gt_only'):
        self.dhs_matrix = dhs_matrix_path
        self.mode = mode
        self.min_samples_per_genotype = 3
        self.min_unique_genotypes = 3
        self.n_cell_types = 2
        self.plink_prefix = plink_prefix
        self.samples_metadata = samples_metadata
        self.dhs_masterlist = dhs_masterlist_path
        self.samples_order = samples_order
        self.valid_dhs = valid_dhs
        # path to DataFrame with columns ag_id PC1 PC2 ...
        self.additional_covariates = additional_covariates

        if genomic_region:
            self.chrom, self.start, self.end = self.unpack_region(genomic_region)
        else:
            self.chrom = self.start = self.end = None

        self.bim = self.fam = self.bed = self.snps_per_dhs = self.cell_types_list = None
        self.metadata = self.ordered_meta = self.indiv2samples_idx = self.ohe_cell_types = None
        self.covariates = self.valid_samples = self.residualizers = None
        self.cell_lines_names = None

    def transform(self):
        self.read_dhs_matrix_meta()
        # Change summit to start/end if needed
        dhs_chunk_index = (self.dhs_masterlist['#chr'] == self.chrom) & \
                          (self.dhs_masterlist['summit'] >= self.start) & \
                          (self.dhs_masterlist['summit'] < self.end)

        self.filter_dhs_matrix(dhs_chunk_index)

        self.load_snp_data()
        snps_index = self.bim.eval(f'chrom == "{self.chrom}" & pos >= {self.start - self.window + 1}'
                                   f' & pos < {self.end + self.window}').to_numpy().astype(bool)
        self.filter_snp_data(snps_index)

        self.load_samples_metadata()

        self.filter_invalid_test_pairs()
        if self.mode != 'gt_only':
            self.include_cell_type_info()  # [SNPs x samples]
        
        self.load_covariates()

        if self.valid_samples.sum() == 0:
            raise NoDataLeftError()

        return QTLmapper(
            phenotype_matrix=self.dhs_matrix,
            snps_per_dhs=self.snps_per_dhs,
            genotype_matrix=self.bed,
            samples_per_snps=self.valid_samples,
            residualizers=self.residualizers,
            snps_data=self.bim[['variant_id', 'pos']],
            dhs_data=self.dhs_masterlist[["#chr", "start", "end", "chunk_id", "summit"]],
            cell_type_data=self.ohe_cell_types,
            mode=self.mode
        )

    def include_cell_type_info(self):
        # cell_types enumerated by sample_index
        self.cell_types_list = self.metadata['CT'].to_numpy()
        ohe_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.ohe_cell_types = ohe_enc.fit_transform(self.cell_types_list.reshape(-1, 1))
        # Filter out cell-types with less than 3 distinct genotypes
        self.valid_samples = self.find_valid_samples_by_cell_type(self.ohe_cell_types.T)  # [SNPs x samples]
        before_n = (self.bed != -1).sum()
        self.bed[~self.valid_samples] = -1

        testable_snps = self.find_testable_snps()
        self.bed = self.bed[testable_snps, :]  # [SNPs x indivs]
        self.snps_per_dhs = self.snps_per_dhs[:, testable_snps]  # [DHS x SNPs] boolean matrix
        self.valid_samples = self.valid_samples[testable_snps, :]
        print(f"SNPxDHS pairs. Before: {before_n}, after: {self.snps_per_dhs.sum()}")
        self.bim = self.bim.iloc[testable_snps, :].reset_index(drop=True)
        self.cell_lines_names = ohe_enc.categories_[0]

    def extract_variant_dhs_signal(self, variant_id, dhs_chunk_id):
        self.read_dhs_matrix_meta()
        dhs_mask = self.dhs_masterlist['chunk_id'] == dhs_chunk_id
        self.filter_dhs_matrix(dhs_mask)
        P = self.dhs_matrix[dhs_mask, :]

        self.load_snp_data()
        snps_mask = self.bim['variant_id'] == variant_id
        self.filter_snp_data(snps_mask)
        self.load_samples_metadata()
        G = self.bed[snps_mask, :]
        res_dict = {'P': P, 'G': G}
        if self.mode != 'gt_only':
            res_dict['CT'] = self.cell_types_list
        return pd.DataFrame(res_dict)

    def read_dhs_matrix_meta(self):
        # TODO fix for no header case
        self.dhs_masterlist = pd.read_table(
            self.dhs_masterlist,
            names=["#chr", "start", "end", "chunk_id", "score", "n_samples", "n_peaks",
                   "dhs_width", "summit", "start_core", "end_core", "avg_score"],
            header=None
        )
        if self.valid_dhs is not None:
            self.valid_dhs = np.loadtxt(self.valid_dhs, dtype=bool)
            self.dhs_masterlist = self.dhs_masterlist[self.valid_dhs]
        else:
            self.valid_dhs = np.ones(len(self.dhs_masterlist.index), dtype=bool)
        self.samples_order = np.loadtxt(self.samples_order, delimiter='\t', dtype=str)

    @staticmethod
    def unpack_region(s):
        chrom, coords = s.split(":")
        start, end = coords.split("-")
        return chrom, int(start), int(end)

    def filter_dhs_matrix(self, dhs_filter):
        self.dhs_masterlist = self.dhs_masterlist[dhs_filter].reset_index(drop=True)
        with h5py.File(self.dhs_matrix, 'r') as f:
            self.dhs_matrix = f['normalized_counts'][dhs_filter, :]  # [DHS x samples]
        assert (~np.isfinite(self.dhs_matrix)).sum() == 0

    def find_testable_snps(self):
        # TODO: prettify
        valid_snps_mask, (homref, het, homalt) = self.filter_by_genotypes_counts_in_matrix(
            self.bed, return_counts=True)
        if self.allele_frac is None:
            return valid_snps_mask
        ma_passing = np.minimum(homref, homalt) * 2 + het >= self.allele_frac * self.bed.shape[1]
        return ma_passing * valid_snps_mask

    def load_snp_data(self):
        self.bim, self.fam, self.bed = read_plink(self.plink_prefix)
        self.bed = 2 - self.bed
        self.bed[np.isnan(self.bed)] = -1
        self.bed = self.bed.astype(np.int8, copy=False)

    def filter_snp_data(self, snps_index):
        # pos is 1-based, start is 0-based
        self.bed = self.bed[snps_index, :].compute()
        # filter SNPs with homref, homalt and hets present
        testable_snps = self.find_testable_snps()
        self.bed = self.bed[testable_snps, :]  # [SNPs x indivs]
        self.bim = self.bim.iloc[snps_index].iloc[testable_snps].reset_index(drop=True)

        if self.bim.empty:
            raise NoDataLeftError()
        # use eval instead?
        self.bim['variant_id'] = self.bim.apply(
            lambda row: f"{row['chrom']}_{row['pos']}_{row['snp']}_{row['a1']}_{row['a0']}",
            axis=1
        )
        self.fam.rename(columns={'iid': 'indiv_id'}, inplace=True)

    def load_samples_metadata(self):
        metadata = pd.read_table(self.samples_metadata)
        req_cols = ['ag_id', 'indiv_id']
        if self.mode != 'gt_only':
            req_cols.append('CT')
        try:
            assert set(req_cols).issubset(metadata.columns)
        except Exception as e:
            print(f'{req_cols} not in {metadata.columns}')
            raise e
        self.metadata = metadata[req_cols].merge(
            self.fam['indiv_id'].reset_index()
        ).set_index('ag_id').loc[self.samples_order, :]
        difference = len(metadata.index) - len(self.metadata.index)
        if difference != 0:
            print(f'{difference} samples has been filtered out!')

        self.indiv2samples_idx = self.metadata['index'].to_numpy()
        self.bed = self.bed[:, self.indiv2samples_idx]

    def filter_invalid_test_pairs(self):
        # returns [DHS x SNPs] boolean matrix, [DHS] boolean vector
        invalid_phens = self.find_snps_per_dhs()
        self.dhs_matrix = self.dhs_matrix[~invalid_phens, :]
        self.snps_per_dhs = self.snps_per_dhs[~invalid_phens, :]  # [DHS x SNPs] boolean matrix
        self.dhs_masterlist = self.dhs_masterlist.iloc[~invalid_phens, :].reset_index(drop=True)
        self.valid_samples = (self.bed != -1)

    def find_snps_per_dhs(self):
        phenotype_len = len(self.dhs_masterlist.index)
        self.snps_per_dhs = np.zeros((phenotype_len, len(self.bim.index)), dtype=bool)
        invalid_phens_indices = []
        unique_chrs = self.bim['chrom'].unique()
        per_chr_groups = None
        if len(unique_chrs) > 1:
            per_chr_groups = self.bim.reset_index().groupby('chrom')
        for phen_idx, row in self.dhs_masterlist.iterrows():
            chrom = row['#chr']
            if per_chr_groups is None:
                chr_df = self.bim.reset_index()
            else:
                chr_df = per_chr_groups.get_group(chrom)
            snp_positions = chr_df['pos'].to_numpy()

            # Change start/end to summit if needed
            lower_bound = np.searchsorted(snp_positions, row['summit'] + 1 - self.window)
            upper_bound = np.searchsorted(snp_positions, row['summit'] + self.window, side='right')
            if lower_bound != upper_bound:
                snps_indices = chr_df['index'].to_numpy()[lower_bound:upper_bound - 1]  # returns one just before
                self.snps_per_dhs[phen_idx, snps_indices] = True
            else:
                invalid_phens_indices.append(phen_idx)

        if len(invalid_phens_indices) != 0:
            print(f'** dropping {len(invalid_phens_indices)} phenotypes without variants in cis-window')
        invalid_phens_mask = np.zeros(phenotype_len, dtype=bool)
        invalid_phens_mask[invalid_phens_indices] = True
        return invalid_phens_mask

    def filter_by_genotypes_counts_in_matrix(self, matrix, return_counts=False):
        homref = (matrix == 0).sum(axis=1)
        het = (matrix == 1).sum(axis=1)
        homalt = (matrix == 2).sum(axis=1)
        res = ((homref >= self.min_samples_per_genotype).astype(np.int8)
               + (het >= self.min_samples_per_genotype).astype(np.int8)
               + (homalt >= self.min_samples_per_genotype).astype(np.int8)
               ) >= self.min_unique_genotypes

        counts = [homref, het, homalt] if return_counts else None
        return res, counts

    def find_valid_samples_by_cell_type(self, cell_types_matrix):
        # cell_types - [cell_type x sample]
        # genotypes # [SNP x sample]
        res = np.zeros(self.bed.shape, dtype=bool)
        for snp_idx, snp_samples in enumerate(self.bed):
            snp_genotype_by_cell_type = cell_types_matrix * (snp_samples[None, :] + 1) - 1  # [cell_type x sample]
            valid_cell_types_mask, _ = self.filter_by_genotypes_counts_in_matrix(
                snp_genotype_by_cell_type,
                return_counts=False
            )
            if valid_cell_types_mask.sum() < self.n_cell_types:
                continue
            res[snp_idx, :] = np.any(cell_types_matrix[valid_cell_types_mask, :] != 0, axis=0)
        return res * (self.bed != -1).astype(bool)  # [SNP x sample]

    def load_covariates(self):
        gt_covariates = pd.read_table(f'{self.plink_prefix}.eigenvec')
        assert gt_covariates['IID'].tolist() == self.fam['indiv_id'].tolist()
        sample_pcs = gt_covariates.loc[self.indiv2samples_idx].iloc[:, 2:].to_numpy()
        if self.additional_covariates is not None:
            additional_covs = pd.read_table(
                self.additional_covariates).set_index('ag_id').loc[self.samples_order]
            self.covariates = np.concatenate(
                [sample_pcs, additional_covs.to_numpy()], axis=1)  # [sample x covariate]
        else:
            self.covariates = sample_pcs

        self.residualizers = np.array([Residualizer(self.covariates[snp_samples_idx, :])
                                       for snp_samples_idx in self.valid_samples])


def main(chunk_id, masterlist_path, non_nan_mask_path, phenotype_matrix_path,
         samples_order_path, plink_prefix, metadata_path, additional_covariates, mode):
    t = time.perf_counter()
    print('Processing started')
    # ---- Read data for specific region only -----
    processing = QTLPreprocessing(
        genomic_region=chunk_id,
        dhs_matrix_path=phenotype_matrix_path,
        dhs_masterlist_path=masterlist_path,
        samples_order=samples_order_path,
        plink_prefix=plink_prefix,
        samples_metadata=metadata_path,
        valid_dhs=non_nan_mask_path,
        additional_covariates=additional_covariates,
        mode=mode
    )
    try:
        qtl_mapper = processing.transform()
    except NoDataLeftError:
        return None
    print(f"Preprocessing finished in {time.perf_counter() - t}s")
    # ------------ Run regressions -----------
    res, coefs = qtl_mapper.map_qtl()
    if qtl_mapper.singular_matrix_count > 0:
        print(f'{qtl_mapper.singular_matrix_count} SNPs excluded! Singular matrix.')
    print(f"Processing finished in {time.perf_counter() - t}s")
    return res, coefs, processing.cell_lines_names


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
    parser.add_argument('--additional_covariates', help='Path to tsv file with additional covariates.'
                                                        'Should have the following columns: ag_id, PC1, PC2, ...',
                                                    default=None)
    parser.add_argument('--mode', help='Specify to choose type of caQTL analysis. gt_only, cell_type or interaction',
                        default='gt_only', const='gt_only', nargs='?',
                        choices=['cell_type', 'gt_only'])

    args = parser.parse_args()

    result = main(
        chunk_id=args.chunk_id,
        masterlist_path=args.index_file,
        non_nan_mask_path=args.mask,
        phenotype_matrix_path=args.phenotype_matrix,
        samples_order_path=args.samples_order,
        plink_prefix=args.plink_prefix,
        mode=args.mode,
        metadata_path=args.metadata,
        additional_covariates=args.additional_covariates
    )
    if result is None:
        print(f'No SNPs passing filters found for {args.chunk_id}, exiting')
        Path(f"{args.outpath}.result.tsv.gz").touch()
        Path(f"{args.outpath}.coefs.tsv.gz").touch()
        Path(f'{args.outpath}.cells_order.txt').touch()
        exit(0)
    regression_result, coefficients, cell_types = result

    regression_result.to_csv(f'{args.outpath}.result.tsv.gz', sep='\t', index=False)
    coefficients.to_csv(f'{args.outpath}.coefs.tsv.gz', sep='\t', index=False)
    if args.mode == 'gt_only':
        Path(f'{args.outpath}.cells_order.txt').touch()
    else:
        np.savetxt(f'{args.outpath}.cells_order.txt', cell_types, delimiter='\t', fmt="%s")
