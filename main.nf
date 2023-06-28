#!/usr/bin/env nextflow

//params.conda = '/home/jvierstra/.local/miniconda3/envs/py3.8_tensorqtl'
params.conda = "/home/sabramov/miniconda3/envs/babachi"


process extract_gc_content {
	conda params.conda
	publishDir params.outdir
	scratch true

	output:
		path gc_content_path

	script:
	gc_content_path = 'regions_gc_annotated.bed.gz'
	"""
	# write header
	echo '#chr	start	end	n_bases	n_gc	percent_gc	region_id	mid	n_mappable' > result.bed
	
	awk 'NR>1' ${params.index_file} > noheader.bed
	
	faidx -i nucleotide -b noheader.bed ${params.genome_fasta} \
		| awk -v OFS="\t" 'NR>1 { total =\$4+\$5+\$6+\$7+\$8; cg=\$6+\$7; print \$1, \$2-1, \$3,total, cg, cg/total;  }' \
		| paste - <( cut -f4,9 noheader.bed ) \
		| bedmap --delim "\t" --echo \
			--bases-uniq - ${params.mappable_file} >> result.bed
		
	cat result.bed | bgzip -c > ${gc_content_path}
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
	tag "${mode}:${genome_chunk}"
	publishDir "${params.outdir}/chunks"

	input:
		each genome_chunk
		tuple path(normalized_matrix), path(mask)
		path plink_files // Files are named as plink.<suffix>
		each mode

	output:
		tuple val(mode), path("${prefix}.result.tsv.gz"), path("${prefix}.coefs.tsv.gz")

	script:
	plink_prefix = "${plink_files[0].simpleName}" // Assumes that prefix of all the files is the same and doesn't contain .
	prefix = "${genome_chunk}.qtl_results.${mode}"
	use_resiudalizer = params.use_resiudalizer ? '--use_resiudalizer' : ''
	covars = params.covars ? "--covariates ${params.covars}" : ""
	"""
	python3 $moduleDir/bin/qtl_regression.py \
		${genome_chunk} \
		${params.samples_file} \
		${normalized_matrix} \
		${mask} \
		${params.index_file} \
		${params.indivs_order} \
		${plink_prefix} \
		${prefix} \
		--mode ${mode} \
		${use_resiudalizer} \
		${covars} \
		--include_ct
	"""
}

process merge_files {
	publishDir params.outdir
	conda params.conda
	tag "${mode}"

	input:
		tuple val(mode), path(results), path(coeffs)
	
	output:
		path "${out_prefix}*"

	script:
	out_prefix = "caqtl_${mode}"
	"""
	ls *.result.tsv.gz > results.filelist.txt
	ls *.coefs.tsv.gz > coeffs.filelist.txt
	python3 $moduleDir/bin/merge_results.py \
		results.filelist.txt \
		coeffs.filelist.txt \
		${out_prefix}

	bgzip ${out_prefix}.results.bed
	tabix ${out_prefix}.results.bed.gz

	bgzip ${out_prefix}.coeffs.bed
	"""
}

workflow extractGC {
	extract_gc_content()
}

workflow caqtlCalling {
	data = extractGC() | gc_normalize_count_matrix
	plink_files = make_plink()
	genome_chunks = create_genome_chunks() | flatMap(n -> n.split())
	qtl_regression(genome_chunks, data, plink_files) //| collectFile(
	// 	name: "caqtl_results.tsv",
	// 	storeDir: params.outdir,
	// 	skip: 1,
	// 	keepHeader: true
	// )
}

workflow mergeFiles {
	res_ct = Channel.fromPath(
		"/net/seq/data2/projects/sabramov/ENCODE4/caqtl-analysis/data.v4/output/chunks/*.cell_type.result.tsv.gz"
	).map(it -> tuple(it.simpleName, it))
	cof_ct = Channel.fromPath(
		"/net/seq/data2/projects/sabramov/ENCODE4/caqtl-analysis/data.v4/output/chunks/*.cell_type.coefs.tsv.gz"
	).map(it -> tuple(it.simpleName, it))

	a = res_ct.join(cof_ct).map(it -> tuple('cell_type', it[1], it[2]))

	res_in = Channel.fromPath(
		"/net/seq/data2/projects/sabramov/ENCODE4/caqtl-analysis/data.v4/output/chunks/*.interaction.result.tsv.gz"
	).map(it -> tuple(it.simpleName, it))
	cof_in = Channel.fromPath(
		"/net/seq/data2/projects/sabramov/ENCODE4/caqtl-analysis/data.v4/output/chunks/*.interaction.coefs.tsv.gz"
	).map(it -> tuple(it.simpleName, it))

	b = res_in.join(cof_in).map(it -> tuple('interaction', it[1], it[2]))
	merge_files(a.concat(b).groupTuple())
}

workflow {
	caqtlCalling()
}
