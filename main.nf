#!/usr/bin/env nextflow

//params.conda = '/home/jvierstra/.local/miniconda3/envs/py3.8_tensorqtl'
params.conda = "/home/sabramov/miniconda3/envs/babachi"


process extract_gc_content {
	conda params.conda
	publishDir params.outdir

	output:
		path gc_content_path

	script:
	gc_content_path = 'regions_gc_annotated.bed.gz'
	"""
	faidx -i nucleotide -b ${params.index_file} ${params.genome_fasta} \
		| awk -v OFS="\t" 'NR>1 { total =\$4+\$5+\$6+\$7+\$8; cg=\$6+\$7; print \$1, \$2-1, \$3,total, cg, cg/total;  }' \
		| bedmap --delim "\t" --echo --bases-uniq - ${params.mappable_file} \
		| paste - <(cut -f4,9 ${params.index_file}) \
	| bgzip -c > ${gc_content_path}
	"""
}

process gc_normalize_count_matrix {
	conda params.conda
	publishDir params.outdir
	scratch true

	input:
		path gc_content_file

	output:
		path name
	
	script:
	name = "matrix_counts.norm.npy"
	"""
	bcftools query -l ${params.genotype_file} > samples.txt

	python3 $moduleDir/bin/normalize_counts.py \
		${gc_content_file} \
		${params.count_matrix} \
		${params.indivs_order} \
		samples.txt \
		${name}
	"""
}

// Select bi-allelic SNVs and make plink files; PCA on genotypes for covariates
process make_plink {
	memory '16G'
	publishDir "${params.outdir}/plink"
	conda params.conda

	output:
		path "plink.*"

	script:
	"""
	plink2 --make-bed \
    	--output-chr chrM \
    	--vcf ${params.genotype_file} \
        --keep-allele-order \
		--allow-extra-chr \
    	--snps-only \
    	--out plink

    plink2 \
    	--bfile plink \
    	--pca \
    	--out plink
	"""
}

// Chunk genome up only look at regions with in the phenotype matrix
process create_genome_chunks {
	executor 'local'
	memory '4G'
	conda params.conda

	input:
		path count_matrix

	output:
		stdout

	script:
	"""
	zcat ${count_matrix} | cut -f1-3 | sort-bed - > regions.bed

	cat ${params.genome_chrom_sizes} \
  	| awk -v step=${params.chunksize} -v OFS="\t" \
		'{ \
			for(i=step; i<=\$2; i+=step) { \
				print \$1, i-step+1, i; \
			} \
			print \$1, i-step+1, \$2; \
		}' \
	| sort-bed - > chunks.bed

	bedops -e 1 chunks.bed regions.bed \
	 	| awk -v OFS="\t" '{ print \$1":"\$2"-"\$3; }'
	"""
}

workflow caqtlCalling {
	count_matrix = extract_gc_content() | gc_normalize_count_matrix
	plink_files = make_plink()
	create_genome_chunks(count_matrix)
}

workflow {
	caqtlCalling()
}

// process qtl_by_region {
// 	tag "${region}"    
// 	label "gpu"

//     conda '/home/jvierstra/.local/miniconda3/envs/py3.8_tensorqtl'

//     publishDir params.outdir + '/nominal', mode: 'symlink', pattern: '*.parquet'

// 	input: 
// 	set file(count_matrix), file(count_matrix_index) from NORMED_COUNTS_FILES
// 	each region from GENOME_CHUNKS.flatMap{ it.split() }
// 	file '*' from PLINK_FILES.collect()

// 	output:
// 	file "*.txt.gz" into QTL_EMPIRICAL
// 	file "*.parquet" into QTL_PAIRS_NOMINAL

// 	script:
// 	"""
// 	qtl.py plink ${count_matrix} plink.eigenvec ${region}
// 	"""
// }

// process merge_permutations {
// 	executor 'slurm'
	
//     conda '/home/jvierstra/.local/miniconda3/envs/py3.8_tensorqtl'

// 	module "R/4.0.5"

// 	publishDir params.outdir + '/qtl', mode: 'symlink'

// 	input:
// 	file '*' from QTL_EMPIRICAL.collect()

// 	output:
// 	file 'all.phenotypes.txt.gz' into QTL_EMPIRICAL_SIGNIF

// 	script:
// 	"""
// 	find \$PWD -name "chr*.txt.gz" > filelist.txt

// 	merge_permutation_results.py filelist.txt all
// 	"""
// }

// QTL_PAIRS_NOMINAL
// 	.map{ it -> 
// 		def names = (it.name.split(":")) 
// 		tuple(names[0], it)
// 	}
// 	.groupTuple(by: 0)
// 	.set{ QTL_PAIRS_NOMINAL_BY_CHR }

// process filter_nominal_pairs {
// 	tag "${chr}"

// 	executor 'slurm'

//     conda '/home/jvierstra/.local/miniconda3/envs/py3.8_tensorqtl'


// 	publishDir params.outdir + '/qtl', mode: 'copy'

// 	input:
// 	set val(chr), file('*') from QTL_PAIRS_NOMINAL_BY_CHR 
// 	file phenotypes_file from QTL_EMPIRICAL_SIGNIF

// 	output:
// 	file "${chr}.signifpairs.txt.gz" into QTL_PAIRS_SIGNIF_BY_CHR

// 	script:
// 	"""
// 	ls *.parquet > filelist.txt

// 	merge_nominal_results.py --fdr 0.05 ${phenotypes_file} filelist.txt ${chr}.signifpairs.txt.gz
// 	"""
// }
