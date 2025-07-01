#!/bin/bash

# LcCNV - GATK CNV calling without Panel of Normals (Median Standardization)
# WARNING: Less sensitive than PoN-based approach, suitable for large predominant events

FASTQ_DIR="."
REF_GENOME="/home/administrator/Lc/LcDatabase_hg19/bwa_index/hs37d5/hs37d5.noalt"
REF_GENOME_FA="/home/administrator/Lc/LcDatabase_hg19/bwa_index/hs37d5/hs37d5.noalt.fa"
TARGETS="/home/administrator/Lc/LcDatabase_hg19/bed_files/WES_HG19/S33266340_Covered.adj.bed"

THREADS=28

process_cnv_sample() {
    local sample=$1
    local fastq1=$2
    local fastq2=$3

#-------------------------- Filtering & Alignment ---------------------------#

gatk PreprocessIntervals \
	-R $REF_GENOME_FA \
	-L $TARGETS \
	--bin-length 1250 \
	--interval-merging-rule OVERLAPPING_ONLY \
	-O ${sample}.targets.preprocessed.interval_list

gatk AnnotateIntervals \
	-R $REF_GENOME_FA \
	-L ${sample}.targets.preprocessed.interval_list \
	--interval-merging-rule OVERLAPPING_ONLY \
	--mappability-track mappability_sorted.bed \
	--segmental-duplication-track segdups.noalt.bed \
	-O ${sample}.annotated.interval_list

#-------------------------- CNV Read Count Collection ---------------------------#

gatk CollectReadCounts \
	-I ${sample}_aligned_marked_bqsr.bam \
	-L ${sample}.targets.preprocessed.interval_list \
	--interval-merging-rule OVERLAPPING_ONLY \
	-O ${sample}.counts.hdf5

#-------------------------- CNV Denoising (WITHOUT PoN - Median Standardization) ---------------------------#

gatk DenoiseReadCounts \
	-I ${sample}.counts.hdf5 \
	--annotated-intervals ${sample}.annotated.interval_list \
	--standardized-copy-ratios ${sample}.standardized.tsv \
	--denoised-copy-ratios ${sample}.denoised.tsv

#-------------------------- CNV Segmentation ---------------------------#

gatk ModelSegments \
	--denoised-copy-ratios ${sample}.denoised.tsv \
	--output ${sample}_segments \
	--output-prefix ${sample} \

gatk CallCopyRatioSegments \
	--input ${sample}_segments/${sample}.cr.seg \
	--output ${sample}.called.seg

#-------------------------- CNV Process and Classification ---------------------------#

# Convert seg to bed
awk 'NR>27 && ($6 > 0.5 || $6 < -0.5) {print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6}' ${sample}.called.seg > ${sample}.cnv.filtered.bed

# Convert + to DUP and - to DEL, keep header, filter out zero-length CNVs
awk 'BEGIN{OFS="\t"} {
    # Handle header line
    if (NR == 1) {
        print $1, $2, $3, $4  # Print header as-is
    } else {
        # Process data lines
        if ($6 == "+") {
            cnv_type = "DUP"
        } else if ($6 == "-") {
            cnv_type = "DEL"
        } else {
            cnv_type = $6  # In case there are other values
        }
        # Only print if end position > start position (filter out zero-length CNVs)
        if ($3 > $2) {
            print $1, $2, $3, cnv_type
        }
    }
}' ${sample}.cnv.filtered.bed > ${sample}.CNV.bed

# Run Classify CNV
ClassifyCNV.py --infile ${sample}.CNV.bed --GenomeBuild hg19

mv *.seg ClassifyCNV_results/
mv ${sample}.annotated.interval_list ClassifyCNV_results/
mv ${sample}.cnv.filtered.bed ClassifyCNV_results/
mv ${sample}.counts.hdf5 ClassifyCNV_results/
mv ${sample}.denoised.tsv ClassifyCNV_results/
mv ${sample}.standardized.tsv ClassifyCNV_results/
mv ${sample}.targets.preprocessed.interval_list ClassifyCNV_results/
mv ${sample}.CNV.bed ClassifyCNV_results/

mv ClassifyCNV_results/Result_*/Scoresheet.txt .

#-------------------------- CNV Prioritization ---------------------------#

(head -1 Scoresheet.txt && tail -n +2 Scoresheet.txt | awk -F'\t' '{
if($6=="Pathogenic") priority=1
else if($6=="Likely pathogenic") priority=2
else if($6=="Uncertain significance") priority=3
else if($6=="Likely benign") priority=4
else if($6=="Benign") priority=5
else priority=6
print priority "\t" $0}' | sort -t$'\t' -k1,1n -k8,8nr | cut -f2-) > ${sample}_CNV_prioritized.tsv

#-------------------------- CNV Report ---------------------------#

# Create Html report for CNV
python LcWesCNV_html.py ${sample}_CNV_prioritized.tsv ${sample}_CNV.html

}

# Main script
for fastq1 in $FASTQ_DIR/*_1.fq.gz; do
    fastq2="${fastq1/_1.fq.gz/_2.fq.gz}"
    if [ -f "$fastq2" ]; then
        sample=$(basename "$fastq1" | sed 's/_1.fq.gz//')
        process_cnv_sample "$sample" "$fastq1" "$fastq2"
    fi
done
