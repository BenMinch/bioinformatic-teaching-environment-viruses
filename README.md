# bioinformatic-teaching-environment-viruses
An environment for learning virus bioinformatics

## Activity 1: Running BEREN
BEREN is a tool for the recovery of Giant viruses and Preplasmiviricota viruses.
### First we need to install BEREN
`git clone https://gitlab.com/benminch1/BEREN`
Then enter the directory
`cd BEREN`
### Setting up the resources
`chmod +rwx scripts/prodigal-gv`
We also need to download the databases
`python Database_setup.py`
### Running BEREN with the test dataset
`python BEREN.py -i Virus_metagenome_test.fasta -o BEREN_output -m NCLDV_marker,NCLDV_SCGs,PPV`
### Examining the outputs

## Activity 2: Running CheckGV to access genome quality
`python CheckGV.py --input-dir NCLDV_SCG_Genomes -o CheckGV_summary.tsv`

### Inspect the output

## Activity 3: Seqkit to look at virus genomes

## Activity 4: Protein Prediction using Prodigal
Choose a giant virus genome and run this command
`prodigal -p meta -i NCLDV_SCG_Genomes/yourgenome -a yourgenome.faa`

## Activity 5: Annotate your proteins
`python AnnoMazing.py -i yourgenome.faa -db AMGs.hmm -annot AMGs.csv`

### Inspect the results and choose a virus to dive deeper.
