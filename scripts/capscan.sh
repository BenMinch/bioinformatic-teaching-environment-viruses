#!/bin/bash
# capscan: Scans all gene predictions (.faa files) for PLV or NCLDV-like MCP genes, classifies the PLV and deposits then in a file.
# Usage: capscan.sh <genes.faa>
# prerequisites: seqkit, HH-suite

VERSION="0.1.0"
echo "capscan version $VERSION"

# change location of HMM profile databases
MCPSCAN="hmm/PLV_Viro_Yara_NCLDV_MCPs_V3.1.hmm" # 50 profiles to scan for MCP genes (quick)
MCPCLUSTER="hmm/VC_MCPs.hmm" # 293 profiles to determine closest viral cluster (Slow and runs on results from above) (Includes 59 Tectivirus profiles - just for fun)

#check for $1 input
if [ -z "$1" ]; then
    echo "Usage: CapscanPLV_V1.sh <genes.faa>"
    exit 1
fi


# Check for seqkit
if ! command -v seqkit &> /dev/null; then
    echo "Error: seqkit is not installed or not in PATH."
    exit 1
fi

# Check for hmmsearch
if ! command -v hmmsearch &> /dev/null; then
    echo "Error: hmmsearch is not installed or not in PATH."
    exit 1
fi

#strip basename
# base=$(basename "$1" .faa)
base="${1%.*}"

# echo $base  # Uncomment for debugging: shows base filename

# set number of CPUs to use (default: 4)
CPU=${CPU:-4}

# scan for capsids
echo 'Scanning for MCP genes'
hmmsearch --tblout $base'_capscan.tblout' -o $base'_capscan_out' -E 1e-9 --cpu $CPU $MCPSCAN $1

echo 'counting MCP hits'
count=$(cat $base'_capscan.tblout' | awk '{print $1}' | grep -v '#' | sort -nk 1 | uniq | wc -l) 
echo $count ' MCP genes found'
echo 'Pulling MCP genes with seqkit'
awk '{print $1}' $base'_capscan.tblout' | sort -nk 1 | uniq | seqkit grep -f - $1 >$base'_MCP.faa'

#parse besthit for MCP gene
echo 'Picking besthits'
awk '{print $1, $6, $0}' $base'_capscan.tblout' | sort -k1,1 -k2,2nr |  awk '!seen[$1]++ {$1=""; $2=""; print $0}' | sed 's/^  //' | awk '{print $1,$2,$3,$4,$5,$6}' | tr ' ' '\t' >$base'_capscan_besthit.tblout'
echo 'File created: '$base'_capscan_besthit.tblout'

echo ' Separating PLVs and GVs'
grep -v '#' $base'_capscan_besthit.tblout' |grep GVOG | awk '{print $1}' >$base'_GVMCPS.list'
grep -v '#' $base'_capscan_besthit.tblout' |grep -v GVOG | awk '{print $1}' >$base'_PLV.list'

# filter for hits over 75 bitscore
cat $base'_capscan_besthit.tblout'  | grep -v '#' |grep -v GVOG | awk '$6>=75 {print $1}' >$base'_PLV75.list'

# create list of weak hits
if [ -s "$base"_PLV75.list ] && [ -s "$base"_PLV.list ]; then
	grep -v -f "$base"_PLV75.list "$base"_PLV.list >"$base"_PLV_weak_hits.list
else
	# If either file is empty, create an empty weak hits list
	>"$base"_PLV_weak_hits.list
fi


seqkit grep -f $base'_PLV_weak_hits.list' $base'_MCP.faa' >$base'_PLV_weak_hits.faa'
seqkit grep -f $base'_GVMCPS.list' $base'_MCP.faa' >$base'_GVMCP.faa'
seqkit grep -f $base'_PLV75.list' $base'_MCP.faa' >$base'_PLVMCP_strong_hits.faa'


count1=$(grep -c ">" "${base}_PLVMCP_strong_hits.faa")
count2=$(grep -c ">" "${base}_GVMCP.faa")

echo "Results:"
echo "$count1 PLV MCPs detected"
echo "$count2 NCLDV MCPs detected"


#format each header with basename (optional)
#cat ${base}_PLVMCP_strong_hits.faa | sed "s/>/>${base}_/" >$base'_PLVMCP75H.faa'

echo 'Finding MCP cluster'
#Find closest MCP cluster

if [ "$count1" -gt 0 ]; then
		#Perform cluster search
	hmmsearch --tblout $base'_clusscan.tblout' -o $base'_clusscan_out' -E 1e-9 --cpu $CPU $MCPCLUSTER $base'_PLVMCP_strong_hits.faa'
	#parse besthits from  MCP cluster scan
	awk '{print $1, $6, $0}' $base'_clusscan.tblout'  | sort -k1,1 -k2,2nr |  awk '!seen[$1]++ {$1=""; $2=""; print $0}' | sed 's/ //' | awk '{print $1,$2,$3,$4,$5,$6}' | tr ' ' '\t' >$base'_clusscan_besthit.tblout'
	else
		# continue
		echo "PLV Count is 0, Skipping cluster search"
fi

echo $base'_capscan_besthit.tblout is the capsid supergroup hit file  (evalue, bitscore)'
echo $base'_clusscan_besthit.tblout shows the closest Virus Cluster  (evalue, bitscore)'
echo $base'_PLVMCP_strong_hits.faa are PLV MCPs'

#cleanup
rm $base'_PLV_weak_hits.list'
mv $base'_PLV75.list' $base'_PLV.list'
rm $base'_MCP.faa'	
