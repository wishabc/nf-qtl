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

	input:
		path gc_content_file

	output:
		tuple path("${prefix}.hdf5"), path("${prefix}.mask.txt")
	
	script:
	prefix = "matrix_counts.norm"
	"""
	python3 $moduleDir/bin/normalize_counts.py \
		${gc_content_file} \
		${params.count_matrix} \
		${prefix}
	"""
}

// Select bi-allelic SNVs and make plink files; PCA on genotypes for covariates
process make_plink {
	publishDir "${params.outdir}/plink"
	memory 400.GB
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
		--allow-extra-chr \
    	--out plink
	"""
}

// Chunk genome up only look at regions with in the phenotype matrix
process create_genome_chunks {
	executor 'local'
	conda params.conda

	output:
		stdout

	script:
	"""
	cat ${params.index_file} | cut -f1-3 | sort-bed - > regions.bed

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

process qtl_regression {
	conda params.conda
	tag "${genome_chunk}"

	input:
		each genome_chunk
		tuple path(normalized_matrix), path(mask)
		path plink_files 	// Files are named as plink.<suffix>

	output:
		tuple val(genome_chunk), path(name)

	script:
	plink_prefix = "${plink_files[0].simpleName}" // Assumes that prefix of all the files is the same and doesn't contain .
	cell_spec = params.cell_spec ? "--cell_spec" : ""
	interaction = params.interaction ? "--with_interaction" : ""
	name = "${genome_chunk}.qtl_results.${cell_spec.replaceAll('-', '')}${interaction.replaceAll('-', '')}"
	"""
	python3 $moduleDir/bin/qtl_regression.py \
		${genome_chunk} \
		${params.samples_file} \
		${normalized_matrix} \
		${mask} \
		${params.index_file} \
		${params.indivs_order} \
		${plink_prefix} \
		${name} \
		${cell_spec} \
		${interaction}
	"""
}

workflow caqtlCalling {
	data = extract_gc_content() | gc_normalize_count_matrix
	plink_files = make_plink()
	genome_chunks = create_genome_chunks() | flatMap(n -> n.split())
	qtl_regression(genome_chunks, data, plink_files) | collectFile(
		name: "caqtl_results.tsv",
		storeDir: params.outdir,
		skip: 1,
		keepHeader: true
	)
}
workflow test {
	genome_chunks = create_genome_chunks() | flatMap(n -> n.split())
	count_matrix = Channel.of(tuple(
		file("/net/seq/data2/projects/sabramov/ENCODE4/caqtl-analysis/output/matrix_counts.hdf5"),
		file("/net/seq/data2/projects/sabramov/ENCODE4/caqtl-analysis/output/matrix_counts.mask.txt")
		))
	plink_files = Channel.of("/net/seq/data2/projects/sabramov/ENCODE4/caqtl-analysis/output/plink/plink*")
		.map(it -> file(it)).collect(sort: true, flat: true)

	qtl_regression(genome_chunks, count_matrix, plink_files) | map(it -> it[1]) | collectFile(
		name: "caqtl_results.tsv",
		storeDir: "${params.outdir}",
		skip: 1,
		keepHeader: true
	)
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
