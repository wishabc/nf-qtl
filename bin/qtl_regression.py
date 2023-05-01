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

# Temporary hotfix
bad_cell_types = [
    'K562', 'Lung cancer cell lines', 'Breast cancer cell lines',
    'Hematopoietic cancer cell lines', 'Colon cancer cell lines',
    'Lung cancer cell lines (NCI-H524)', 'Soft tissue cancer cell lines',
    'Caco-2', 'HepG2', 'MG-63', 'SJSA1', 'SK-N-DZ', 'BE2C',
    'G-401',
    'Daoy',
    'Calu3', 'PC-9', 'RPMI-7951',
    'RWPE2', 'SJCRH30',
    'RKO', 'A172', 'ELR', 'LNCaP clone FGC', 'Jurkat, Clone E6-1',
    'HAP-1', 'HT1080',
    'M059J', 'SIG-M5',
    'HL-60', 'SK-N-SH',
    'HS-5', 'TF-1.CN5a.1', 'HT-29', 'L1-S8', 'MCF-7',
    'NCI-H226', 'PC-3', 'HS-27A', 'WERI-Rb-1'
]


def unpack_region(s):
    chrom, coords = s.split(":")
    start, end = coords.split("-")
    return chrom, int(start), int(end)


def remove_redundant_columns(matrix):
    cols_mask = np.any(matrix != 0, axis=0)
    return matrix[:, cols_mask], cols_mask


class QTLPreprocessing:
    window = 500_000
    allele_frac = 0.05

    def __init__(self, dhs_matrix_path, dhs_masterlist_path, samples_order,
                 plink_prefix, samples_metadata, additional_covariates=None,
                 valid_dhs=None, mode='gt_only', cond_num_tr=100):
        self.dhs_matrix_path = dhs_matrix_path
        self.mode = mode
        self.min_individuals_per_genotype = 2
        self.min_unique_genotypes = 3
        self.n_cell_types = 2
        self.plink_prefix = plink_prefix
        self.cond_num_tr = cond_num_tr
        self.valid_dhs = valid_dhs
        # path to DataFrame with columns ag_id PC1 PC2 ...
        self.additional_covariates = additional_covariates

        self.initial_dhs_masterlist = self.dhs_masterlist = self.indiv_names = self.samples_order = None
        self.initial_bim = self.bim = self.fam = self.bed = self.bed_by_sample = self.bed_dask = None
        self.dhs_matrix = None
        self.snps_per_dhs = self.cell_types_list = self.ct_names = self.good_indivs_mask = None
        self.metadata = self.ordered_meta = self.id2indiv = self.sample2id = self.ohe_cell_types = None
        self.covariates = self.valid_samples = self.residualizers = None

        self.read_dhs_matrix_meta(dhs_masterlist_path, samples_order)
        self.load_snp_data()
        self.load_samples_metadata(samples_metadata)

    def transform(self, genomic_region):
        chrom, start, end = unpack_region(genomic_region)

        # Change summit to start/end if needed
        dhs_chunk_mask = (self.initial_dhs_masterlist['#chr'] == chrom) & \
                         (self.initial_dhs_masterlist['summit'] >= start) & \
                         (self.initial_dhs_masterlist['summit'] < end)
        snps_mask = self.initial_bim.eval(f'chrom == "{chrom}" & pos >= {start - self.window + 1}'
                                          f' & pos < {end + self.window}').to_numpy().astype(bool)

        self.preprocess(dhs_chunk_mask, snps_mask)
        if self.valid_samples.sum() == 0:
            raise NoDataLeftError()

        return QTLmapper(
            phenotype_matrix=self.dhs_matrix,
            snps_per_dhs=self.snps_per_dhs,
            genotype_matrix=self.bed_by_sample,
            samples_per_snps=self.valid_samples,
            residualizers=self.residualizers,
            snps_data=self.bim[['variant_id', 'pos']],
            dhs_data=self.dhs_masterlist[["#chr", "start", "end", "chunk_id", "summit"]],
            ct_data=self.ohe_cell_types,
            ct_names=self.ct_names,
            cond_num_tr=self.cond_num_tr,
            mode=self.mode
        )

    def preprocess(self, dhs_chunk_mask, snps_mask):
        self.load_dhs_matrix(dhs_chunk_mask)
        self.load_snp_matrix(snps_mask)
        if self.mode != 'gt_only':
            self.include_cell_type_info()  # [SNPs x samples]

        self.filter_invalid_test_pairs()
        self.load_covariates()
        self.bed = None

    def extract_variant_dhs_signal(self, variant_id, dhs_chunk_id):
        dhs_mask = self.initial_dhs_masterlist['chunk_id'] == dhs_chunk_id

        chrom, pos, _, _, a0 = variant_id.split('_')
        snps_mask = self.initial_bim.eval(
            f'chrom == "{chrom}" & pos == {pos} & a0 == "{a0}"'
        )
        self.preprocess(dhs_mask, snps_mask)
        g = self.bed_by_sample.squeeze()
        p = self.dhs_matrix.squeeze()
        valid_samples = self.valid_samples.squeeze()
        residualizer = self.residualizers[0]

        # FIXME for multiple variants case
        g_res = np.full(g.shape[0], np.nan)
        g_res[valid_samples] = residualizer.transform(
            g[valid_samples][None, :]
        ).squeeze()
        p_res = np.full(p.shape[0], np.nan)
        p_res[valid_samples] = residualizer.transform(
            p[valid_samples][None, :]
        ).squeeze()
        res_dict = {
            'P': p,
            'G': g,
            'S': valid_samples,
            'P_res': p_res,
            'G_res': g_res,
            'indiv_index': self.indiv_names
        }
        if self.mode != 'gt_only':
            res_dict['CT'] = self.cell_types_list

        return pd.DataFrame(res_dict)

    def include_cell_type_info(self):
        self.cell_types_list = self.metadata['CT'].to_numpy()
        ohe_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.ohe_cell_types = ohe_enc.fit_transform(self.cell_types_list.reshape(-1, 1)).astype(bool)
        self.ct_names = ohe_enc.categories_[0]

    def read_dhs_matrix_meta(self, dhs_masterlist_path, samples_order):
        # TODO fix for no header case
        self.initial_dhs_masterlist = pd.read_table(
            dhs_masterlist_path,
            names=["#chr", "start", "end", "chunk_id", "score", "n_samples", "n_peaks",
                   "dhs_width", "summit", "start_core", "end_core", "avg_score"],
            header=None
        )
        if self.valid_dhs is not None:
            self.valid_dhs = np.loadtxt(self.valid_dhs, dtype=bool)
            self.initial_dhs_masterlist = self.initial_dhs_masterlist[self.valid_dhs]
        else:
            self.valid_dhs = np.ones(len(self.initial_dhs_masterlist.index), dtype=bool)
        self.samples_order = np.loadtxt(samples_order, delimiter='\t', dtype=str)

    def load_dhs_matrix(self, dhs_filter):
        self.dhs_masterlist = self.initial_dhs_masterlist[dhs_filter].reset_index(drop=True)
        if self.dhs_masterlist.empty:
            raise NoDataLeftError()
        with h5py.File(self.dhs_matrix_path, 'r') as f:
            self.dhs_matrix = f['normalized_counts'][dhs_filter, :]  # [DHS x samples]
        assert (~np.isfinite(self.dhs_matrix)).sum() == 0
        self.dhs_matrix = self.reformat_samples(self.dhs_matrix, downsample=True)

    def load_snp_data(self):
        self.initial_bim, self.fam, self.bed_dask = read_plink(self.plink_prefix)
        self.bed_dask = 2 - self.bed_dask
        self.bed_dask[np.isnan(self.bed_dask)] = -1
        self.bed_dask = self.bed_dask.astype(np.int8, copy=False)
        self.fam.rename(columns={'iid': 'indiv_id'}, inplace=True)

    def load_snp_matrix(self, snps_index):
        self.bim = self.initial_bim.loc[snps_index].reset_index(drop=True)
        if self.bim.empty:
            raise NoDataLeftError()
        self.bed = self.bed_dask[snps_index, :].compute()  # [SNPs x indivs]

        # use eval instead?
        self.bim['variant_id'] = self.bim.apply(
            lambda row: f"{row['chrom']}_{row['pos']}_{row['snp']}_{row['a1']}_{row['a0']}",
            axis=1
        )
        self.bed_by_sample = self.reformat_samples(self.bed, downsample=False)

    def load_samples_metadata(self, samples_metadata):
        metadata = pd.read_table(samples_metadata)

        self.metadata = metadata.merge(
            self.fam['indiv_id'].reset_index()
        ).set_index('ag_id').loc[self.samples_order, :]

        cell_type_by_indiv = self.metadata.sort_values(
            ['CT', 'indiv_id'], ascending=[True, True]
        ).drop_duplicates(['CT', 'indiv_id'])[['CT', 'indiv_id', 'index']].reset_index(drop=True)
        cell_type_by_indiv['cti'] = np.arange(len(cell_type_by_indiv.index))

        self.id2indiv = cell_type_by_indiv['index'].to_numpy()
        self.sample2id = self.metadata.merge(cell_type_by_indiv,
                                             on=['CT', 'indiv_id', 'index'])['cti'].to_numpy()
        # --------- Temporary fix --------
        bad_samples_mask = self.metadata['CT'].isin(bad_cell_types).to_numpy()
        bad_indivs = self.metadata[bad_samples_mask]['index'].unique()
        self.bed_dask[:, bad_indivs] = -1
        # --------------------------------
        difference = len(metadata.index) - len(self.metadata.index)
        if difference != 0:
            print(f'{difference} samples has been filtered out!')

        self.indiv_names = self.metadata['indiv_id'].to_numpy()

    def filter_invalid_test_pairs(self):
        invalid_phens = self.find_snps_per_dhs()
        self.dhs_matrix = self.dhs_matrix[~invalid_phens, :]
        self.snps_per_dhs = self.snps_per_dhs[~invalid_phens, :]  # [DHS x SNPs] boolean matrix
        self.dhs_masterlist = self.dhs_masterlist.iloc[~invalid_phens, :].reset_index(drop=True)
        before_n = (self.bed_by_sample != -1).sum()
        self.valid_samples = (self.bed_by_sample != -1)
        if self.mode != 'gt_only':
            # Filter out cell-types with less than 3 distinct genotypes
            self.valid_samples *= self.find_valid_samples_by_cell_type()  # [SNPs x samples]

        self.bed_by_sample[~self.valid_samples] = -1
        testable_snps, (homref, het, homalt) = self.filter_by_genotypes_counts_in_matrix(
            self.bed_by_sample, return_counts=True)
        if self.allele_frac is not None:
            ma_passing = np.minimum(homref, homalt) * 2 + het >= self.allele_frac * self.bed.shape[1]
            testable_snps = ma_passing * testable_snps

        self.bed_by_sample = self.bed_by_sample[testable_snps, :]  # [SNPs x indivs]
        self.snps_per_dhs = self.snps_per_dhs[:, testable_snps]  # [DHS x SNPs] boolean matrix
        self.valid_samples = self.valid_samples[testable_snps, :]
        print(f"SNPxDHS pairs. Before: {before_n}, after: {self.snps_per_dhs.sum()}")
        self.bim = self.bim.iloc[testable_snps, :].reset_index(drop=True)

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
                snps_indices = chr_df['index'].to_numpy()[lower_bound:upper_bound]
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
        res = ((homref >= self.min_individuals_per_genotype).astype(np.int8)
               + (het >= self.min_individuals_per_genotype).astype(np.int8)
               + (homalt >= self.min_individuals_per_genotype).astype(np.int8)
               ) >= self.min_unique_genotypes

        counts = [homref, het, homalt] if return_counts else None
        return res, counts

    # NEED TO THINK
    def reformat_samples(self, matrix, downsample=True):
        if downsample:
            ids_count = self.id2indiv.shape[0]
            res = np.zeros((matrix.shape[0], ids_count), dtype=matrix.dtype)
            counts = np.zeros(ids_count)
            for sample_id, agg_id in enumerate(self.sample2id):
                res[:, agg_id] += matrix[:, sample_id]
                counts[agg_id] += 1
            print(self.sample2id)
            res /= counts[None, :]
        else:
            res = matrix[:, self.id2indiv]
        return res.astype(matrix.dtype)  # [N x id]

    def find_valid_samples_by_cell_type(self):
        cell_types_matrix = self.reformat_samples(self.ohe_cell_types.T, downsample=True)  # - [cell_type x id]

        res = np.zeros(self.bed.shape, dtype=bool)  # [SNP x id]
        for snp_idx, snp_sample_gt in enumerate(self.bed):
            # [cell_type x indiv]
            snp_genotype_by_cell_type = cell_types_matrix.astype(bool) * (snp_sample_gt[None, :] + 1) - 1
            valid_cell_types_mask, _ = self.filter_by_genotypes_counts_in_matrix(
                snp_genotype_by_cell_type,
                return_counts=False
            )
            if valid_cell_types_mask.sum() < self.n_cell_types:
                continue
            res[snp_idx, :] = np.any(cell_types_matrix[valid_cell_types_mask, :] != 0, axis=0)
        return res.astype(bool)  # [SNP x sample]

    def load_covariates(self):
        if self.additional_covariates is not None:
            additional_covs = pd.read_table(
                self.additional_covariates
            ).set_index('ag_id').loc[self.samples_order]
            self.covariates = self.reformat_samples(additional_covs.to_numpy(), downsample=True)
            # self.covariates = np.concatenate(
            # [sample_pcs, additional_covs.to_numpy()], axis=1)  # [sample x covariate]
        else:
            gt_covariates = pd.read_table(f'{self.plink_prefix}.eigenvec')
            assert gt_covariates['IID'].tolist() == self.fam['indiv_id'].tolist()
            sample_pcs = gt_covariates[self.good_indivs_mask].loc[self.id2indiv].iloc[:, 2:].to_numpy()
            self.covariates = sample_pcs

        self.residualizers = np.array([Residualizer(self.covariates[snp_samples_idx, :], self.cond_num_tr)
                                       for snp_samples_idx in self.valid_samples])


class NoDataLeftError(Exception):
    pass


class Residualizer:
    def __init__(self, C, cond_num=100):
        # center and orthogonalize
        self.n = np.linalg.matrix_rank(C)
        self.dof = C.shape[0] - self.n
        # debug
        self.C = C
        if self.dof == 0 or np.linalg.cond(C) > cond_num:
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
                 snps_data, dhs_data, mode,
                 cond_num_tr=100, ct_data=None, ct_names=None):

        self.phenotype_matrix = phenotype_matrix
        self.snps_per_phenotype = snps_per_dhs
        self.genotype_matrix = genotype_matrix
        self.samples_per_snps = samples_per_snps

        self.residualizers = residualizers

        self.snps_data = snps_data
        self.dhs_data = dhs_data

        self.mode = mode
        self.ct_data = ct_data
        self.ct_names = ct_names

        self.cond_num_tr = cond_num_tr

        self.singular_matrix_count = 0
        self.poorly_conditioned = 0

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
        if self.mode != 'ct_only':

            try:
                n_hom_ref, n_het, n_hom_alt = np.unique(snp_genotypes, return_counts=True)[1]
            except Exception:
                print(np.unique(snp_genotypes, return_counts=True))
                raise
        else:
            n_hom_ref = n_het = n_hom_alt = np.nan
        df_model = design.shape[1]
        df_residuals = design.shape[0] - df_model - residualizer.n
        if np.linalg.cond(design) >= self.cond_num_tr:
            self.poorly_conditioned += 1
            raise np.linalg.LinAlgError()
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
                self.poorly_conditioned += 1
                continue
            if self.mode in ('interaction', 'ct_only'):
                # calculate interaction
                ohe_cell_types, valid_design_cols_mask = remove_redundant_columns(self.ct_data[valid_samples, :])
                used_names = self.ct_names[valid_design_cols_mask]
                if self.mode == 'interaction':
                    snp_genotypes = snp_genotypes * ohe_cell_types  # [samples x cell_types]
                else:
                    snp_genotypes = ohe_cell_types
            else:
                used_names = np.full(1, np.nan, dtype=str)

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
                               used_names.shape[0], axis=0)

            stack = np.stack([used_names, *coeffs]).T

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
        if len(stats_res) == 0:
            raise NoDataLeftError()

        stats_res = pd.DataFrame(stats_res, columns=[
            "#chr", "start", "end", "chunk_id", "summit",
            'variant_id', 'distance_to_summit',
            'n_samples', 'n_hom_ref', 'n_het', 'n_hom_alt',
            'ss_model', 'ss_residuals', 'df_model', 'df_residuals'
        ])
        coefs_res = pd.DataFrame(np.concatenate(coefs_res),
                                 columns=['variant_id', 'chunk_id', 'design_var_index', 'coeff', 'coeff_se'])
        return self.post_processing(stats_res), coefs_res


def main(chunk_id, masterlist_path, non_nan_mask_path, phenotype_matrix_path,
         samples_order_path, plink_prefix, metadata_path, additional_covariates, mode):
    t = time.perf_counter()
    print('Processing started')
    # ---- Read data for specific region only -----
    processing = QTLPreprocessing(
        dhs_matrix_path=phenotype_matrix_path,
        dhs_masterlist_path=masterlist_path,
        samples_order=samples_order_path,
        plink_prefix=plink_prefix,
        samples_metadata=metadata_path,
        valid_dhs=non_nan_mask_path,
        additional_covariates=additional_covariates,
        cond_num_tr=100,
        mode=mode
    )
    try:
        qtl_mapper = processing.transform(genomic_region=chunk_id)
        print(f"Preprocessing finished in {time.perf_counter() - t}s")
        res, coefs = qtl_mapper.map_qtl()
    except NoDataLeftError:
        return None
    # ------------ Run regressions -----------
    if qtl_mapper.singular_matrix_count > 0:
        print(f'{qtl_mapper.singular_matrix_count - qtl_mapper.poorly_conditioned} SNPs excluded! Singular matrix.')
    if qtl_mapper.poorly_conditioned > 0:
        print(f'{qtl_mapper.poorly_conditioned} SNPs excluded! Poorly conditioned residuals.')
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
    parser.add_argument('--additional_covariates', help='Path to tsv file with additional covariates.'
                                                        'Should have the following columns: ag_id, PC1, PC2, ...',
                        default=None)
    parser.add_argument('--mode', help='Specify to choose type of caQTL analysis. gt_only, cell_type or interaction',
                        default='gt_only', const='gt_only', nargs='?',
                        choices=['cell_type', 'interaction', 'ct_only', 'gt_only'])

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
        print(f'No SNPs can be tested for {args.chunk_id}, exiting')
        Path(f"{args.outpath}.result.tsv.gz").touch()
        Path(f"{args.outpath}.coefs.tsv.gz").touch()
        exit(0)
    regression_result, coefficients = result

    regression_result.to_csv(f'{args.outpath}.result.tsv.gz', sep='\t', index=False)
    coefficients.to_csv(f'{args.outpath}.coefs.tsv.gz', sep='\t', index=False)
