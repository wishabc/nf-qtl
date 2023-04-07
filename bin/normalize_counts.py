import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

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
    return normalized_tag_counts
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

    # with open(sys.argv[3]) as f:
    #     pheno_indivs = f.readline().strip().split()
    # with open(sys.argv[4]) as f:
    #     genot_indivs = [line.rstrip() for line in f]

    # assert len(pheno_indivs) == len(genot_indivs)

    # new_genot_order = [pheno_indivs.index(x) for x in genot_indivs]
    # Check whether files match
    assert raw_tag_counts.shape[0] == len(regions_annotations.index), "Counts and annotation files do not match!"
    index_chrs = set(regions_annotations['#chr'].unique())
    assert all([x not in index_chrs for x in exclude_chrs])
     # Get relevant samples (e.g., to match VCF file)
    # raw_tag_counts = raw_tag_counts[:, new_genot_order]
    normalized_tag_counts = main(raw_tag_counts, regions_annotations)
    np.save(sys.argv[3], normalized_tag_counts)
