import os,argparse,re,subprocess
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='''A wrapper for binning NCLDV genomes from metagenomic assemblies''')
parser.add_argument('-i', '--input', help='Input metagenomic assembly in fasta format', required=True)
parser.add_argument('-o', '--output', help='Output folder', required=True)
parser.add_argument('-cov', '--coverage', help='Coverage file for metagenomic binning', required=True)

args = parser.parse_args()
input_file = args.input
output = args.output
coverage = args.coverage

#Binning Contigs
os.mkdir(output+'/01_Metabat_bins')
command= 'metabat2 -i '+input_file+' -a '+ coverage+ ' -o '+output+'/01_Metabat_bins/bin -m 5000 -s 50000'
subprocess.call(command, shell=True)
#If no bins are found, exit program
if len(os.listdir(output+'/01_Metabat_bins'))==0:
    print('No bins found')
    exit()
#Predict proteins for bins
os.mkdir(output+'/02_Prodigal_bins')
prodigal_input= output+'/01_Metabat_bins'
command= 'python scripts/prodigal_launcher.py '+prodigal_input+' '+output+'/02_Prodigal_bins'
subprocess.call(command, shell=True)
os.mkdir(output+'/02_Prodigal_bins/Proteins')
move= 'mv '+output+'/02_Prodigal_bins/*.faa '+output+'/02_Prodigal_bins/Proteins'
subprocess.call(move, shell=True)

#Markersearch
os.mkdir(output+'/03_NCLDV_Markersearch')
search= 'python scripts/ncldv_markersearch.py -i '+output+'/02_Prodigal_bins/Proteins -a -m  GVOGm0890,GVOGm0760,GVOGm0461,GVOGm0172,GVOGm0054,GVOGm0023,GVOGm0013,GVOGm0022,GVOGm0003 -t 6 -n '+output+'/03_NCLDV_Markersearch/markersearch'
subprocess.call(search, shell=True,stdout=open("NCLDV_Markersearch.out", "w"), stderr=open("NCLDV_Markersearch.err", "w"))
ncldv_out= pd.read_csv(output+'/03_NCLDV_Markersearch/markersearch.table.tsv', sep='\t')
ncldv_out['total']=ncldv_out.iloc[:,1:].astype(bool).sum(axis=1)
ncldv_out=ncldv_out[ncldv_out['total']>=1]
ncldv_out2=ncldv_out.iloc[:,0]
ncldv_out.to_csv(output+'/03_NCLDV_Markersearch/ncldv_summary.tsv', sep='\t', index=False)
ncldv_out2.to_csv(output+'/03_NCLDV_Markersearch/ncldv_bins.txt', sep='\t', index=False)

os.mkdir(output+'/03_NCLDV_Markersearch/ncldv_good_bins')
extract= 'cat '+output+'/03_NCLDV_Markersearch/ncldv_bins.txt | xargs -I {} cp '+output+'/01_Metabat_bins/{} '+output+'/03_NCLDV_Markersearch/ncldv_good_bins'
subprocess.call(extract, shell=True,stdout=open("NCLDV_Markersearch.out", "w"), stderr=open("NCLDV_Markersearch.err", "w"))
#If no NCLDV genomes are found, exit program
if len(ncldv_out)==0:
    print('No NCLDV genomes found')
    exit()
#Run ViralRecall
os.mkdir(output+'/04_Viral_recall')
recall= 'python scripts/viralrecall.py -i '+output+'/03_NCLDV_Markersearch/ncldv_good_bins -p '+output+'/04_Viral_recall/viral_recall_out -t 6 -c -b'
subprocess.call(recall, shell=True)
os.mkdir(output+'/05_Viral_screening')
os.mkdir(output+'/05_Viral_screening/summary_files')
extract= 'python scripts/extract_anything.py '+output+'/04_Viral_recall/viral_recall_out '+output+'/05_Viral_screening/summary_files .summary.tsv'
subprocess.call(extract, shell=True,stdout=open("Viral_screening.out", "w"), stderr=open("Viral_screening.err", "w"))
post_screen= 'python scripts/viralrecall_post_screen.py '+ output+'/03_NCLDV_Markersearch/ncldv_summary.tsv'+ ' '+output+'/05_Viral_screening/summary_files'
subprocess.call(post_screen, shell=True,stdout=open("Viral_screening.out", "w"), stderr=open("Viral_screening.err", "w"))
move= 'mv viralrecall_batch_summary.tsv '+output+'/05_Viral_screening'
subprocess.call(move, shell=True)
#moving good genomes into directory
os.mkdir(output+'/05_Viral_screening/ncldv_good_genomes')
viral_screen=pd.read_csv(output+'/05_Viral_screening/viralrecall_batch_summary.tsv', sep='\t')
viral_partial = viral_screen
viral_screen=viral_screen[viral_screen['Virus']=='yes']
viral_screen=viral_screen[viral_screen['Other_Markers']>=3]
#Get genomes not in viral_screen
viral_partial=viral_partial[viral_partial['genome'].isin(viral_screen['genome'])==False]
viral_partial_genomes=viral_partial.iloc[:,0]
viral_genomes=viral_screen.iloc[:,0]
#If no viral genomes are found, exit program
if len(viral_genomes)==0:
    print('No complete viral genomes found')
else:
    viral_genomes.to_csv(output+'/05_Viral_screening/ncldv_good_genomes.txt', sep='\t', index=False)
viral_partial_genomes.to_csv(output+'/05_Viral_screening/ncldv_partial_genomes.txt', sep='\t', index=False)
extract= 'cat '+output+'/05_Viral_screening/ncldv_good_genomes.txt | xargs -I {} cp '+output+'/03_NCLDV_Markersearch/ncldv_good_bins/{} '+output+'/05_Viral_screening/ncldv_good_genomes'
subprocess.call(extract, shell=True,stdout=open("Viral_screening.out", "w"), stderr=open("Viral_screening.err", "w"))
os.mkdir(output+'/05_Viral_screening/ncldv_partial_genomes')
extract2= 'cat '+output+'/05_Viral_screening/ncldv_partial_genomes.txt | xargs -I {} cp '+output+'/03_NCLDV_Markersearch/ncldv_good_bins/{} '+output+'/05_Viral_screening/ncldv_partial_genomes'
subprocess.call(extract2, shell=True,stdout=open("Viral_screening.out", "w"), stderr=open("Viral_screening.err", "w"))

##Decontamination
os.mkdir(output+'/06_Final_Results')
if len(viral_genomes)==0:
    print('No complete viral genomes found')
else:
    os.mkdir(output+'/05_Viral_screening/clean_genomes')
    files=os.listdir(output+'/05_Viral_screening/ncldv_good_genomes')
    new_files=[]
    for file in files:
        new_files.append(file.replace('.fa','.summary.tsv'))
    new_files=pd.DataFrame(new_files)
    new_files.to_csv(output+'/05_Viral_screening/new_files.txt', sep='\t', index=False)
    os.mkdir(output+'/05_Viral_screening/good_summary_files')
    extract= 'cat '+output+'/05_Viral_screening/new_files.txt | xargs -I {} cp '+output+'/05_Viral_screening/summary_files/{} '+output+'/05_Viral_screening/good_summary_files'
    subprocess.call(extract, shell=True,stdout=open("Viral_screening.out", "w"), stderr=open("Viral_screening.err", "w"))
    for file in os.listdir(output+'/05_Viral_screening/good_summary_files'):
        #read in the summary file if it ends with tsv
        if file.endswith('.tsv'):
            summary=pd.read_csv(output+'/05_Viral_screening/good_summary_files/'+file, sep='\t')
            summary=summary[summary['score']<=0]
            summary_removes= summary['replicon']
            summary_removes.to_csv(output+'/05_Viral_screening/good_summary_files/'+file+'.removes.txt', sep='\t', index=False)
    for file in os.listdir(output+'/05_Viral_screening/ncldv_good_genomes'):
        #read in the summary file if it ends with tsv
        if file.endswith('.fa'):
            #get the name of the file without the .fa
            name=file.replace('.fa','')
            #make a variable for the removes file
            removes=output+'/05_Viral_screening/good_summary_files/'+name+'.summary.tsv.removes.txt'
            #run the remove script
            remove= 'python scripts/Decontaminate_fastas.py '+output+'/05_Viral_screening/ncldv_good_genomes/'+file+' '+removes+' > '+output+'/05_Viral_screening/clean_genomes/'+file
            subprocess.call(remove, shell=True,stdout=open("Viral_screening.out", "w"), stderr=open("Viral_screening.err", "w"))
    move_dRep= 'mv '+output+'/05_Viral_screening/clean_genomes '+output+'/06_Final_Results'
    subprocess.call(move_dRep, shell=True)


    print('Done')


##Cleaning partial genomes
if len(viral_partial_genomes)==0:
    print('No partial viral genomes found')
else:
    os.mkdir(output+'/05_Viral_screening/clean_partial_genomes')
    files=os.listdir(output+'/05_Viral_screening/ncldv_partial_genomes')
    new_files=[]
    for file in files:
        new_files.append(file.replace('.fa','.summary.tsv'))
    new_files=pd.DataFrame(new_files)
    new_files.to_csv(output+'/05_Viral_screening/new_partial_files.txt', sep='\t', index=False)
    os.mkdir(output+'/05_Viral_screening/partial_summary_files')
    extract= 'cat '+output+'/05_Viral_screening/new_partial_files.txt | xargs -I {} cp '+output+'/05_Viral_screening/summary_files/{} '+output+'/05_Viral_screening/partial_summary_files'
    subprocess.call(extract, shell=True,stdout=open("Viral_screening.out", "w"), stderr=open("Viral_screening.err", "w"))
    for file in os.listdir(output+'/05_Viral_screening/partial_summary_files'):
        #read in the summary file if it ends with tsv
        if file.endswith('.tsv'):
            summary=pd.read_csv(output+'/05_Viral_screening/partial_summary_files/'+file, sep='\t')
            summary=summary[summary['score']<=0]
            summary_removes= summary['replicon']
            summary_removes.to_csv(output+'/05_Viral_screening/partial_summary_files/'+file+'.removes.txt', sep='\t', index=False)
    for file in os.listdir(output+'/05_Viral_screening/ncldv_partial_genomes'):
        #read in the summary file if it ends with tsv
        if file.endswith('.fa'):
            #get the name of the file without the .fa
            name=file.replace('.fa','')
            #make a variable for the removes file
            removes=output+'/05_Viral_screening/partial_summary_files/'+name+'.summary.tsv.removes.txt'
            #run the remove script
            remove= 'python scripts/Decontaminate_fastas.py '+output+'/05_Viral_screening/ncldv_partial_genomes/'+file+' '+removes+' > '+output+'/05_Viral_screening/clean_partial_genomes/'+file
            subprocess.call(remove, shell=True,stdout=open("Viral_screening.out", "w"), stderr=open("Viral_screening.err", "w"))
    print('Done')

    move_dRep= 'mv '+output+'/05_Viral_screening/clean_partial_genomes '+output+'/06_Final_Results'
    subprocess.call(move_dRep, shell=True)
    #Remove files from clean_partial_genomes if they are empty files
for file in os.listdir(output+'/06_Final_Results/clean_partial_genomes'):
    if os.stat(output+'/06_Final_Results/clean_partial_genomes/'+file).st_size==0:
        remove= 'rm '+output+'/06_Final_Results/clean_partial_genomes/'+file
        subprocess.call(remove, shell=True)
remove1= 'rm '+ output+'/03_NCLDV_Markersearch/ncldv_bins.txt'
remove2= 'rm '+ output+'/03_NCLDV_Markersearch/ncldv_summary.tsv'
subprocess.call(remove1, shell=True)
subprocess.call(remove2, shell=True)
remove3= 'rm '+ output+'/05_Viral_screening/ncldv_good_genomes.txt'
remove4= 'rm '+ output+'/05_Viral_screening/new_files.txt'
subprocess.call(remove3, shell=True)
subprocess.call(remove4, shell=True)
remove5= 'rm -r '+ output+'/05_Viral_screening/good_summary_files'
subprocess.call(remove5, shell=True)
move= 'mv '+output+'/05_Viral_screening/viralrecall_batch_summary.tsv '+output+'/06_Final_Results/NCLDV_Genome_Summary.tsv'
subprocess.call(move, shell=True)