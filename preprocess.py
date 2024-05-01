import pandas as pd
from Bio import SeqIO
import numpy as np 


def gtf_to_dataframe(gtf_file: str):
    '''Returns a pandas dataframe of the original GTF file - Jonas'''
    gtf_data = []
    with open(gtf_file) as file:
        for record in parseGTF(file):
            gtf_data.append(record)

    df = pd.DataFrame(gtf_data)
    return df

def parseGTF(gtfFile: str):
    '''Helper for gtf_to_dataframe which reads GTF file and organises it into fields. - Jonas'''
    for line in gtfFile:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        attributes = {}
        for attribute in fields[8].split(";"):
            if attribute.strip():
                try:
                    key, value = attribute.strip().split(" ", 1)  
                    attributes[key] = value.strip('"')
                except ValueError:
                    key = attribute.strip()
                    attributes[key] = None  
        yield {
            "seqname": fields[0],
            "source": fields[1],
            "feature": fields[2],
            "start": int(fields[3]),
            "end": int(fields[4]),
            "score": fields[5],
            "strand": fields[6],
            "frame": fields[7],
            **attributes
        }

def get_SS_data(df: pd.DataFrame): 
    '''Takes in a pandas dataFrame of the GTF file and returns a sorted list of the start and stop indices
    (in the FASTA file) for exons - Jonas'''
    exon_df = df[df['feature'] == 'exon']
    start_df = exon_df['start']
    end_df = exon_df['end']
    ss_df = pd.concat([start_df, end_df], axis=1)
    ss_df = ss_df.sort_values(by=['start'], ascending=True)
    return ss_df 


#NOTE: May want to adjust how genome is stored (string, list, etc)

def readFASTA_by_chromosome(fastaFile: str):
  '''Return a 2D array of base pairs indexable by chromosome and then position within chromosome - Jonas'''
  genome = []
  with open(fastaFile) as file:
    #Iterate through each 'record' (representing a chromosome) and turn into a string
    for chromosome in SeqIO.parse(file, 'fasta'):
      seq = list(str(chromosome.seq).upper()) #potential error
      genome.append(seq)

  return genome
"""
def readFASTA_by_genome(fastaFile: str): 
  '''Return base pairs indexable by position, not taking into account chromosomes - Jonas'''

  genome = []
  with open(fastaFile) as file:
    for chromosome in SeqIO.parse(file, 'fasta'):
      seq = str(chromosome.seq)
      genome.append(seq)

  return genome
"""
def encodeWindows(gtfFile: str, sequence: str, window_sz: int, sig_str: int, sig_end: int):
  '''slides a window over a sequence of genes and checks whether there is a splice site or not. If so, returns a sequence of window size with label 1, else label 0 - Jonas'''
  ss_df = get_SS_data(gtf_to_dataframe(gtfFile))
  ss_array = ss_df.T.values

  embeddings = []
  labels = []

  for i in range(0,len(sequence),100):
    chunk = sequence[i:i+window_sz]

    #There IS a signal IF the signal start is an element of the ss_df[0] OR if sig_end is an element of the ss_df[1]
    if (((i + sig_end) in ss_array[0]) or ((i + sig_str) in ss_array[1])):
      labels.append([1,0])
    else: 
      labels.append([0,1])

    embeddings.append(chunk)

  #embeddings is an array of window size and the corresponding label (either SS or not)
  return embeddings, labels 

