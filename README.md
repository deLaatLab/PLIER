# PLIER

# TODO: add description

# define PLIER settings
e.g. reference genome path

# alignments

Map the source FASTQ files to the genome (hg19 in this case): 
```
bwa mem -SP -t6 -k12 -A2 -B3 reference_index_hg19_bwa ./fastqs/test_seqrun/FXXX_R1.fastq.gz ./fastqs/test_seqrun/FXXX_R2.fastq.gz | samtools sort -n | samtools view -h -q1 -b - > ./bams/test_seqrun/FXXX.bam"
```
