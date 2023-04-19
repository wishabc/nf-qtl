import sys
from tqdm import tqdm
import shutil
import gzip
import os
from qtl_regression import unpack_region
import numpy as np


def sort_key(filepath):
    chunk_id = os.path.basename(filepath).split('.')[0]
    chrom, start, _ = unpack_region(chunk_id)
    return chrom, start


def sort_filenames(filepaths):
    return list(sorted(filepaths, key=sort_key))


def merge_files(sorted_filepaths, outpath):
    header_copied = False
    with open(outpath, 'wb') as outfile:
        for file in tqdm(sorted_filepaths):
            if os.stat(file).st_size == 0:
                continue
            with gzip.open(file, 'rb') as infile:
                a = infile.readline()
                if not a:
                    raise AssertionError
                if not header_copied:
                    infile.seek(0)
                    header_copied = True
                shutil.copyfileobj(infile, outfile)


def main(result_filenames, coefs_filenames, output_prefix):
    merge_files(sort_filenames(result_filenames), f"{output_prefix}.results.bed")
    merge_files(sort_filenames(coefs_filenames), f"{output_prefix}.coeffs.bed")
    


if __name__ == '__main__':
    res_files = np.loadtxt(sys.argv[1], dtype=str)
    coefs_filelist = sys.argv[2]		
    cells_order = sys.argv[3]
    prefix = sys.argv[4]
    main(res_files, output_prefix=prefix)
