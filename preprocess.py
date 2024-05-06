import tensorflow as tf
import pandas as pd
from Bio import SeqIO
import numpy as np
from itertools import product

############################################################################################################
# processing and embedding data                                                                            #
############################################################################################################

def TextToList(fileName):
    ''' Read fasta file and return its sequences as a list '''
    dna_list = []
    with open(fileName) as file:
        for line in file:
            li = line.strip()
            if not li.startswith(">"):
                dna_list.append(line.rstrip("\n"))
    file.close()
    return dna_list

def split_up_down(dna_list, sig_str, sig_end, begin, end):
    ''' Split regions around SS to Up stream and Down stream '''
    ''' TODO: is this exclding data?? '''
    down=[]
    up=[]
    for s in dna_list:
        up.append(s[begin:sig_str])
        down.append(s[sig_end:end])

    return up, down

def EncodeOneHot(dna_list):
    ''' Encode a list of DNA sequences to a list of 4xLx1 "images" '''
    seq = dna_list[0]
    n = len(seq)
    bases = 'ACGT'
    base_to_ind = {b: i for i, b in enumerate(bases)}
    data = np.zeros((len(dna_list), 4, n))
    for i, seq in enumerate(dna_list):
        for j, base in enumerate(seq):
            data[i][base_to_ind[base]][j] += 1

    return data

def EncodeSeqToMono_4D(dna_list):
    ''' Encode a list of DNA sequences to a list of 4xLx1 "images" '''
    seq = dna_list[0]
    n = len(seq)
    bases = 'ACGT'
    base_to_ind = {b: i for i, b in enumerate(bases)}
    data = []
    for seq in dna_list:
        image = np.zeros((4, n))
        for i, base in enumerate(seq):
            image[base_to_ind[base]][i] +=1

        data.append(image)

    return tf.expand_dims(tf.constant(data), axis=3)

def EncodeSeqToTri_64D(dna_list):
    ''' Encode a list of DNA sequences to a list of 64xLx1 "images" '''
    seq = dna_list[0]
    n = len(seq)
    tris = list(map(''.join, product('ACGT', repeat=3)))
    tris_to_ind = {tri: i for i, tri in enumerate(tris)}
    data = []
    for seq in dna_list:
        image = np.zeros((64, n))
        for i in range(len(seq) - 2):
            tri = ''.join(seq[i:i+3])
            image[tris_to_ind[tri]][i] += 1

        data.append(image)

    return tf.expand_dims(tf.constant(data), axis=3)

def RemoveNonAGCT(dna_list):
    ''' Remove sequences that contain characters other than ACGT '''
    chars = set('ACGT')
    dna_listACGT = [s for s in dna_list if all(c in chars for c in s)]
    if len(dna_listACGT) < len(dna_list):
            print('Some sequence excluded, contained characters other than ACGT')

    return dna_listACGT

############################################################################################################
# reading genome from file                                                                                 #
############################################################################################################

def parseGTF(gtfFile: str):
    ''' Helper for gtf_to_dataframe which reads GTF file and organises it into fields '''
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

def gtf_to_dataframe(gtf_file: str):
    ''' Returns a pandas dataframe of the original GTF file '''
    gtf_data = []
    with open(gtf_file) as file:
        for record in parseGTF(file):
            gtf_data.append(record)

    df = pd.DataFrame(gtf_data)
    return df

def get_SS_data(gtf_file: str):
    ''' Takes in a pandas dataFrame of the GTF file and returns a sorted list
    of the start and stop indices (in the FASTA file) for exons '''
    df = gtf_to_dataframe(gtf_file)
    exon_df = df[df['feature'] == 'exon']
    start_df = exon_df['start']
    end_df = exon_df['end']
    ss_df = pd.concat([start_df, end_df], axis=1)
    ss_df = ss_df.sort_values(by=['start'], ascending=True)
    return ss_df

def readFASTA_by_chromosome(fastaFile: str):
    ''' Return a 2D array of base pairs indexable by chromosome and then position within chromosome by
        iterating through each and turning the sequence into a list of uppercase characters'''
    genome = []
    with open(fastaFile) as file:
        for chromosome in SeqIO.parse(file, 'fasta'):
            seq = list(str(chromosome.seq).upper())
            genome = seq
            break # only get one chromosome

    return genome

def encodePositives(ss_df: pd.DataFrame, sequence: str, window_sz: int, sig_str: int, sig_end: int):
    ''' extracts all positive exmaples from the sequence and returns them as a list of embeddings '''
    pos_embeddings = []
    length = len(sequence)

    ss_df = ss_df[ss_df <= length]

    for pos in ss_df:
        window_start = max(pos - sig_str, 0)
        window_end = min(pos + sig_end, length)
        chunk = sequence[window_start:window_end]
        if len(chunk) == window_sz:
          pos_embeddings.append(chunk)

    return pos_embeddings, [[1,0]] * len(pos_embeddings)

def encodeWindows(ss_df: pd.DataFrame, sequence: str, window_sz: int, sig_str: int, sig_end: int) -> tuple[list, tf.Tensor]:
    ''' slides a window over a sequence of genes and checks whether there is a splice site or not.
        If so, returns a sequence of window size with label 1, else label 0 '''

    embeddings, labels = encodePositives(ss_df, sequence, window_sz, sig_str, sig_end)

    sig = sig_str if ss_df.name == 'start' else sig_end
    # get ~ as many negative as positive
    step = len(sequence) // len(embeddings)
    for i in range(0, len(sequence), step):
    #There IS a signal IF the signal start is an element of the ss_df[0] OR if sig_end is an element of the ss_df[1]
        if not (i + sig) in ss_df:
            chunk = sequence[i:i+window_sz]
            embeddings.append(chunk)
            labels.append([0,1])

    #embeddings is an array of window size and the corresponding label (either SS or not)
    return embeddings, labels
