#!/usr/bin/env nextflow
nextflow.enable.dsl = 2
params.vcf_file='/net/seq/data/projects/regulotyping/genotypes/genotype_panel/imputed_genotypes/chroms1-22.phaseI+II.annotated.ancestral.vcf.gz'
// params.count_matrix_file='/net/seq/data/projects/regulotyping/dnase/by_celltype_donor/h.CD3+/index/tag_counts/matrix_tagcounts.txt.gz'


params.regions_file='/net/seq/data/projects/regulotyping/dnase/by_celltype_donor/h.CD3+/index/masterlist_DHSs_h.CD3+_nonovl_core_chunkIDs.bed'

params.genome_fasta_file='/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa'
params.genome_chrom_sizes_file = '/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.chrom_sizes'
params.genome_mappable_file = '/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.K76.mappable_only.bed'
params.chunksize=25000000 //be careful not set this too small or the qtl.py script gives a cryptic error
//params.conda = '/home/jvierstra/.local/miniconda3/envs/py3.8_tensorqtl'
params.conda = '/home/sabramov/miniconda3/envs/tensorqtl'

params.outdir='output'



// process normalize_count_matrix {
// 	executor 'slurm'

// 	conda params.conda

// 	publishDir params.outdir, mode: 'copy'

// 	input:
// 	file 'matrix_counts.txt.gz' from file(params.count_matrix_file)
// 	file 'genotypes.vcf.gz' from file(params.vcf_file)
// 	file 'regions.bed' from file(params.regions_file)
	
// 	file 'genome.fa' from file(params.genome_fasta_file)
// 	file 'genome.fa.fai' from file("${params.genome_fasta_file}.fai")
// 	file 'mappable.bed' from file(params.genome_mappable_file)

// 	output:
// 	file 'regions_annotated.bed.gz*'
// 	set file('matrix_counts.norm.bed.gz'), file('matrix_counts.norm.bed.gz.tbi') into NORMED_COUNTS_FILES, NORMED_COUNTS_REGIONS

// 	script:
// 	"""
// 	faidx -i nucleotide -b regions.bed genome.fa \
// 		| awk -v OFS="\\t" 'NR>1 { total =\$4+\$5+\$6+\$7+\$8; cg=\$6+\$7; print \$1, \$2-1, \$3,total, cg, cg/total;  }' \
// 		| bedmap --delim "\t" --echo --bases-uniq - mappable.bed \
// 		| paste - <(cut -f4,9 regions.bed) \
// 	| bgzip -c > regions_annotated.bed.gz

// 	bcftools query -l genotypes.vcf.gz > samples.txt

// 	normalize_counts.py \
//         regions_annotated.bed.gz \
//         matrix_counts.txt.gz \
//         samples.txt \
// 	    | bgzip -c > matrix_counts.norm.bed.gz
// 	tabix -p bed matrix_counts.norm.bed.gz
// 	"""
// }

// // Select bi-allelic SNVs and make plink files; PCA on genotypes for covariates
// process make_plink {
// 	executor 'local'
// 	memory '16G'
// 	publishDir params.outdir + '/plink' , mode: 'symlink'

// 	input:
// 	file 'genotypes.vcf.gz' from file(params.vcf_file)

// 	output:
// 	file "plink.*" into PLINK_FILES

// 	script:
// 	"""
// 	plink2 --make-bed \
//     	--output-chr chrM \
//     	--vcf genotypes.vcf.gz \
//         --keep-allele-order \
//     	--snps-only \
//     	--out plink

//     plink2 \
//     	--bfile plink \
//     	--pca \
//     	--out plink
// 	"""
// }

// Chunk genome up only look at regions with in the phenotype matrix
process create_genome_chunks {
	executor 'local'
	memory '4G'

	input:
	    tuple path(count_matrix), path(count_matrix_index) 

	output:
	    stdout // into GENOME_CHUNKS

	script:
	"""
	zcat ${count_matrix} | cut -f1-3 | sort-bed - > regions.bed

	cat "${params.genome_chrom_sizes_file}" \
  	| awk -v step=${params.chunksize} -v OFS="\\t" \
		'{ \
			for(i=step; i<=\$2; i+=step) { \
				print \$1, i-step+1, i; \
			} \
			print \$1, i-step+1, \$2; \
		}' \
	| sort-bed - > chunks.bed

	bedops -e 1 chunks.bed regions.bed \
        | awk -v OFS="\\t" '{ print \$1":"\$2"-"\$3; }'
	"""
} 

process qtl_by_region {
	tag "${region}"    
	label "gpu"

    conda params.conda

    publishDir params.outdir + '/nominal', mode: 'symlink', pattern: '*.parquet'

	input: 
	    tuple val(region), path(count_matrix), path(count_matrix_index)
	    path plink_files // from PLINK_FILES.collect()
        

	output:
	    path "*.txt.gz", emit: qtl_empirical
	    path "*.parquet", emit: qtl_nominal

	script:
    prefix = "${plink_files[0].simpleName}"
	"""
	qtl.py plink \
        ${count_matrix} \
        ${prefix}.eigenvec \
        ${region}
	"""
}

process merge_permutations {
	executor 'slurm'
	
    conda params.conda

	module "R/4.0.5"

	publishDir params.outdir + '/qtl', mode: 'symlink'

	input:
	    path qtl_pairs

	output:
	    path 'all.phenotypes.txt.gz' // into QTL_EMPIRICAL_SIGNIF

	script:
	"""
	find \$PWD -name "chr*.txt.gz" > filelist.txt

	$moduleDir/bin/merge_permutation_results.py filelist.txt all
	"""
}

process filter_nominal_pairs {
	tag "${chrom}"

	executor 'slurm'

    conda params.conda


	publishDir params.outdir + '/qtl', mode: 'copy'

	input:
	    tuple val(chrom), file('*') // from QTL_PAIRS_NOMINAL_BY_CHR 
	    path phenotypes_file // from QTL_EMPIRICAL_SIGNIF

	output:
	    path name // into QTL_PAIRS_SIGNIF_BY_CHR

	script:
    name = "${chrom}.signifpairs.txt.gz"
	"""
	ls *.parquet > filelist.txt

	merge_nominal_results.py --fdr 0.05 ${phenotypes_file} filelist.txt ${name}
	"""
}


workflow {
    params.plink_prefix = "/net/seq/data2/projects/sabramov/regulotyping-phaseI-II/imputed_genotypes/chroms1-22.phaseI+II"
    params.count_matrix_file = '/net/seq/data2/projects/sabramov/regulotyping-phaseI/rnaseq-eqtls/phaseI.expression.bed.gz'
    plink_files = Channel.fromPath("${params.plink_prefix}*")

    count_matrix = Channel.of(params.count_matrix_file)
        | map(it -> tuple(file(it), file("${it}.tbi")) )

    count_matrix_w_chunks = count_matrix 
        | create_genome_chunks
        | combine(count_matrix)

    qtl_data = qtl_by_region(count_matrix_w_chunks, plink_files)

    qtl = qtl_data.qtl_nominal
        | map(it -> tuple(it.name.split(":")[0], it))
        | groupTuple(by: 0)
    
    phenotypes = qtl_data.qtl_empirical.collect()
        | merge_permutations
    
    filter_nominal_pairs(qtl, phenotypes)

}