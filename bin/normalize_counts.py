import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import h5py
exclude_chrs = ['chrX', 'chrY', 'chrM']


def main(raw_tag_counts, regions_annotations):
    # Normalize by total counts & # of mappable base in element
    normalized_tag_counts = raw_tag_counts / raw_tag_counts.sum(axis=0)* 1e6
    normalized_tag_counts = normalized_tag_counts / regions_annotations["n_mappable"].values[:, None]

    # GC content normalization
    n_quantiles = 50
    gc_bins = pd.qcut(regions_annotations["percent_gc"], q=n_quantiles, duplicates='drop', labels=False)


    ## To optimize, should work decent for small n_quantiles
    for gc_bin in np.unique(gc_bins):
        indexes = gc_bins == gc_bin
        normalized_subset = normalized_tag_counts[indexes, :]
        normalized_tag_counts[indexes, :] = normalized_subset - np.nanmedian(normalized_subset, axis=0)

    # Mean and variance scaling
    row_means = np.nanmean(normalized_tag_counts, axis=1)[:, None]
    row_sigmas = np.nanstd(normalized_tag_counts, axis=1)[:, None]
    normalized_tag_counts = (normalized_tag_counts - row_means) / row_sigmas
    # Quantile normalization

    qt = QuantileTransformer(n_quantiles=1000, random_state=0, output_distribution='normal')

    normalized_tag_counts = qt.fit_transform(normalized_tag_counts)

    # Drop rows that are all NAs
    #normalized_tag_counts.dropna(axis='rows', inplace=True)
    nan_mask = np.isfinite(normalized_tag_counts).any(axis=1)
    return normalized_tag_counts[nan_mask, :], nan_mask
    ### do I really need it?
    
    #df = regions_annotations.reset_index()[["#chr", "mid", "end", "region_id"]].join(normalized_tag_counts, on="region_id", how="right")
    #df.sort_values(by = ["chr", "mid"], inplace=True)

    # Rename columns for compatibility with TensorQTL
    #df.rename(columns={"chr": "#chr", "mid": "start", "region_id": "phenotype_id"}, inplace=True)
    #df["end"] = df["start"] + 1


    #df.to_csv(sys.stdout, header=True, index=False, sep="\t", float_format="%0.4f")


if __name__ == '__main__':
    regions_annotations = pd.read_table(sys.argv[1], header=None,
            names=["#chr", "start", "end", "n_bases", "n_gc", "percent_gc", "n_mappable", "region_id", "mid"])
    regions_annotations.set_index("region_id", inplace=True)

    raw_tag_counts = np.load(sys.argv[2])

    # Check whether files match
    assert raw_tag_counts.shape[0] == len(regions_annotations.index), "Counts and annotation files do not match!"
    index_chrs = set(regions_annotations['#chr'].unique())
    assert all([x not in index_chrs for x in exclude_chrs])
    # Get relevant samples (e.g., to match VCF file)
    # raw_tag_counts = raw_tag_counts[:, new_genot_order]
    normalized_tag_counts, nan_mask = main(raw_tag_counts, regions_annotations)
    prefix = sys.argv[3]

    with h5py.File(f'{prefix}.hdf5', 'w') as f:
        f['normalized_counts'] = normalized_tag_counts
    np.savetxt(f'{prefix}.mask.txt', nan_mask, fmt="%5i")
