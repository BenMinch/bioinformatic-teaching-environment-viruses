##Total Markersearch##
import pandas as pd
import numpy as np
import sys,os,subprocess,re,argparse

parser = argparse.ArgumentParser(description='NewVirusFinder A tool to find new viruses in metagenomic data')
parser.add_argument('-i', '--input', help='Input folder of genomes', required=True)
parser.add_argument('-o', '--output', help='Output file', required=True)
parser.add_argument('-t', '--threads', help='Number of threads', required=True)
parser.add_argument('-f', '--flag', help='Flag where genome name splits to protein', required=True)
args = parser.parse_args()

input_file = args.input
output_file = args.output
threads = args.threads
flag = args.flag

###Running AnnoMazing to annotate proteins###
print("Finding Markers")
command= 'python scripts/AnnoMazing.py -i '+input_file+' -o '+output_file+' -db hmm/polintovirus.hmm -annot resources/PolintoVirus_markers.csv'
subprocess.call(command, shell=True,stdout=open('Tm_log.txt','w'),stderr=open('Tm_log.err','w'))

annomazing_output = output_file+'_final.csv'

###Parsing AnnoMazing output###
results = pd.read_csv(annomazing_output)
##filter out evalue less than 1e-5
results = results[results['evalue']<1e-10]
#create a column called genome splitting at the flag
results['genome'] = results['query_id'].str.split(flag).str[0]
#group by genome and Source and count the number of hits
results = results.groupby(['genome','Source']).size().reset_index(name='counts')
#pivot the table
results = results.pivot(index='genome', columns='Source', values='counts').reset_index()
#replace NaN with 0
results = results.fillna(0)

#Get rowsum for all columns that contain NCLDV in the header

results['Total_PLV'] = results.filter(like='PLV').sum(axis=1)


results.to_csv(output_file+'_plv_markers.csv',index=False)
