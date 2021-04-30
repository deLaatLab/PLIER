# Introduction
PLIER is a simple yet powerful framework to identify regions in the genome with more than expected coverage. This coverage can be defined as a typical mapping depth of reads mapping to the reference genome, or number of independent ligation-products captured in a proximity-ligation dataset.
[![DOI](https://zenodo.org/badge/300543907.svg)](https://zenodo.org/badge/latestdoi/300543907)


# How to use PLIER

## Pipeline requirements:
The PLIER pipeline requires the following tools:
- A Unix like shell (e.g. Bash v3.2+)
- Samtools v1.9+
- Bwa v0.7.17+
- Python v3.5+ and the following Python packages:
    - numpy v1.13.3+
    - pandas v0.23.4+
    - h5py v2.7.1+
    - matplotlib v2.1.2+ (only for producing summary statistics)
    - pysam v0.15.1+
    - scikit-learn v0.23.2+
    - python-Levenshtein v0.12.0+

The user can run the following command to make sure all required packages are installed:
```
pip3 install --upgrade numpy pandas h5py matplotlib pysam scikit-learn python-Levenshtein
```

## Path definitions
PLIER's paths can be defined by opening `./configs.json` in a text editor and describing the proper paths in your system.  
These paths include (for example) path to the FASTA file of your reference genome (`reference_path`) or path to its
corresponiding BWA index (`bwa_index`).

## Defining probe sets (or viewpoints) coordinates
Probes coordinates are defined (by default) in `./probs/prob_info.tsv`. Its a tab-delimited file that
describes the probe coordinates as well as a given experiment name. For example:

**Table.1.** An example of probes details.

|chr|pos_begin|pos_end|target_locus|description|
|:---|:---|:---|:---|:---|
|chr18|60736416|60996600|BCL2|BCL2_rearrangement|
|chr3|187429165|187760000|BCL6|BCL6_rearrangement|
|chr8|128448316|129453680|MYC|MYC300-700kb|

## Defining viewpoints (or experiment)
Each row in the prob information file (i.e. `./probs/prob_info.tsv`, **Table.1**) produces an experiment 
(also known as "viewpoint", e.g. the target gene such as MYC, BCL2, etc.). Under the hood, PLIER
assigns reads to their related viewpoint based on 
their fragment’s overlap with the viewpoint’s coordinates. A mapped read is discarded if it did not overlap with 
any viewpoint. As a result of this procedure, each combination of sample and viewpoint produces an independent 
FFPE-TLC experiment. Other related meta-data for each experiment are stored in `./vp_info.tsv`. 

**Note:** No modification is needed to be done in `./vp_info.tsv` file to run the PLIER on the "test" example. For an
actual application of PLIER, the user should update this file according to the details of his/her experiments. 

## Running PLIER (an example "test" case)
1. Start by defining configurations (in `configs.json`) and probe details (in `prob_info.tsv`, see above).
2. For FFPE-TLC experiments:
   1. Aligning sequenced reads: We recommend to use BWA-MEM using the following command template (`hg19` in this case):
      ```
      bwa mem -SP -t6 -k12 -A2 -B3 reference_index_hg19_bwa ./fastqs/test_seqrun/FXXX_R1.fastq.gz ./fastqs/test_seqrun/FXXX_R2.fastq.gz | samtools sort -n | samtools view -h -q1 -b - > ./bams/test_seqrun/FXXX.bam"
      ```
      **Notes:** 
         - No alignment is needed to go thorough the "test" example. The FASTQ files are already aligned and stored in
         `./bams/` folder.
         - The alignments are sorted by name (i.e. samtools sorting with `-n` argument). This facilitates
         collection of fragments produced by a single read (i.e. split-mapping) when traversing the BAM file.
   
   2. Demultiplexing reads into experiments by: 
      ```
      python3 ./01_TLC_01-demux_by_coords.py --seqrun_id='TLC-test' --patient_id='FXXX' --genome='hg19'
      ```
      For each defined viewpoint (i.e. each row in `./probs/prob_info.tsv` file), this script produces an independent 
      alignment file that contains reads that enclosed at least a fragment mapping to that viewpoint.
      
   3. Storing the demultiplexed reads into an HDF5 container by:
      ```
      python3 ./02_make_dataset.py
      ```
      PLIER maps the fragments into an in-silico digested reference genome. This procedure requires positions of restriction enzyme's
      recognition sites of a given reference genome that are stored in a file. These files are stored in `./renzs/` folder. 
      Such files are tab-delimited compressed files and are automatically created by PLIER (if not found) but they can be 
      generated manualy as well. Automatic generation is done by scanning the given FASTA file (defined in `configs.json`) 
      of a given reference genome and storing the enzyme recognized coordinates in the relevant tab-delimited and compressed file. 
      For example `./renzs/hg19_NlaIII.npz` refers to the `hg19` genome that is in-silico digested using `NlaIII` restriction 
      enzyme (i.e. `CATG` sequence).
   
   4. (optional) Calculating varied statistics for each experiments (for quality control purposes).
      ```
      python3 03_calculate_datasetsStatistics.py
      ```
      This command computes the wide range of statistics for every experiment in the `vp_info.tsv` file. But subselection can be made using
      `--ridx_beg` and `--ridx_end` arguments.
   
   5. Computing enrichment scores using PLIER by: 
      ```
      python3 04_compute_enrichment_scores.py --run_id=0
      ```
      This command computes the enrichment scores for the first (`index=0`, default) experiment defined in `vp_info.tsv` file. The
      user can change the index to compute enrichment scores for another experiment (i.e. another row in `vp_info.tsv` file). 
      Therefore, each application of PLIER is executed independently. This faciliate implementation of PLIER in a high-performance computing environment.
      For each experiment, the enrichment scores are stored in the following path: 
      ```
      ./outputs/04_sv-caller_01-zscores-per-bin/AllBins_v1.0/ 
      ```

## FFPE-4C experiments:
We also provided scripts that can be used to process FFPE-4C experiments. This can be done by the following steps: 
   1. Demultiplexing the original sequencing FASTQ file into individual FASTQs per experiment: `./01_4C_01_demuxer.py'
   2. Trimming the primer sequences to facilitate alignment: `./01_4C_02_trimmer.py'
   3. Mapping the trimmed FASTQ files to the reference genome: `./01_4C_03_mapper.py'
   4. Storing the alignments in an HDF5 container: './02_make_dataset.py'
   5. Calculating the enrichment scores: `./04_compute_enrichment_scores.py`


## Contact & Questions
For any inquiry please contact Amin Allahyar at a{DOT}allahyar{AT}hubrecht{DOT}eu.


