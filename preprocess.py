import tensorflow as tf
import pandas as pd
from Bio import SeqIO
import numpy as np
from itertools import product

############################################################################################################
# processing and embedding data                                                                            #
############################################################################################################

def RemoveNonAGCT(dna_list, labels):
    ''' Remove sequences that contain characters other than ACGT '''
    chars = set('ACGT')
    dna_listACGT = []
    labelsACGT = []
    for s, label in zip(dna_list, labels):
        if all(c in chars for c in s):
            dna_listACGT.append(s)
            labelsACGT.append(label)
        else:
            print('Some sequence excluded, contained characters other than ACGT')

    return dna_listACGT, labelsACGT

def split_up_down(dna_list, sig_str, sig_end, begin, end):
    ''' Split regions around SS to Up stream and Down stream '''
    ''' TODO: is this exclding data?? '''
    down=[]
    up=[]
    for s in dna_list:
        up.append(s[begin:sig_str])
        down.append(s[sig_end:end])

    return up, down

def EncodeSeqToMono_4D(dna_list):
    ''' Encode a list of DNA sequences to a list of 4xLx1 "images" '''
    seq = dna_list[0]
    n = len(seq)
    bases = 'ACGT'
    base_to_ind = {b: i for i, b in enumerate(bases)}
    data = np.zeros((len(dna_list), 4, n), dtype=np.uint8)
    for i, seq in enumerate(dna_list):
        indices = [base_to_ind[base] for base in seq]
        data[i, indices, np.arange(n)] = 1
    return tf.expand_dims(tf.convert_to_tensor(data, dtype=tf.float32), axis=3)

def EncodeSeqToTri_64D(dna_list):
    ''' Encode a list of DNA sequences to a list of 64xLx1 "images" '''
    seq = dna_list[0]
    n = len(seq)
    tris = list(map(''.join, product('ACGT', repeat=3)))
    tris_to_ind = {tri: i for i, tri in enumerate(tris)}
    data = np.zeros((len(dna_list), 64, n - 2), dtype=np.uint8)
    for i, seq in enumerate(dna_list):
        indices = [tris_to_ind[''.join(seq[j:j+3])] for j in range(n - 2)]
        data[i, indices, np.arange(n - 2)] = 1
    return tf.expand_dims(tf.convert_to_tensor(data, dtype=tf.float32), axis=3)

############################################################################################################
# reading genome from file                                                                                 #
############################################################################################################

def parseGTF(gtfFile: str):
    ''' Helper for gtf_to_dataframe which reads GTF file and organises it into fields '''
    for line in gtfFile:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        yield {
            "seqname": fields[0],
            "source": fields[1],
            "feature": fields[2],
            "start": int(fields[3]),
            "end": int(fields[4]),
            "score": fields[5],
            "strand": fields[6],
            "frame": fields[7]
        }

def gtf_to_dataframe(gtf_file: str):
    ''' Returns a pandas dataframe of the original GTF file '''
    gtf_data = []
    with open(gtf_file) as file:
        for record in parseGTF(file):
            gtf_data.append(record)

    df = pd.DataFrame(gtf_data)
    return df

def get_SS_data(gtf_file: str, boundary: str):
    ''' Takes in a pandas dataFrame of the GTF file and returns a sorted list
    of the start and stop indices (in the FASTA file) for exons '''
    df = gtf_to_dataframe(gtf_file)
    exon_df = df[df['feature'] == 'exon']
    start_df = exon_df['start']
    end_df = exon_df['end']
    ss_df = pd.concat([start_df, end_df], axis=1)
    ss_df = ss_df.sort_values(by=['start'], ascending=True)
    return ss_df[boundary]

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

def encodeWindows(ss_df: pd.DataFrame, sequence: list[str], window_sz: int, sig_str: int, sig_end: int) -> tuple[list, tf.Tensor]:
    ''' slides a window over a sequence of genes and checks whether there is a splice site or not.
        If so, returns a sequence of window size with label 1, else label 0 '''

    # filter out any indices that are out of bounds
    len_seq = len(sequence)
    ss_df = ss_df[(ss_df > sig_str) & (ss_df < len_seq - sig_end)]

    pos_embeddings = [sequence[pos-sig_str:pos+sig_end] for pos in ss_df]
    pos_labels = [[1, 0]] * len(pos_embeddings)

    # Create a mask for if window has splice site near center
    mask = np.zeros(len_seq, dtype=bool)
    for pos in ss_df:
        mask[pos-sig_str//2:pos+sig_end//2] = True
    # Find valid indices where the mask is False and randomly select from them
    neg_inds = np.where(~mask)[0]
    neg_inds = neg_inds[(neg_inds > sig_str) & (neg_inds < len_seq - sig_end)]
    # get as many negative as positive
    n_emb = len(pos_embeddings)
    neg_starts = np.random.choice(neg_inds, size=n_emb, replace=False)
    neg_embeddings = [sequence[start-sig_str:start+sig_end] for start in neg_starts]
    neg_labels = [[0, 1]] * len(neg_embeddings)

    pos_embeddings.extend(neg_embeddings)
    pos_labels.extend(neg_labels)

    return pos_embeddings, pos_labels
