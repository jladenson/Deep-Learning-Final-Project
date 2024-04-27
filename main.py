import tensorflow as tf
from keras import losses, metrics
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from models import Conv, MLP
rng = np.random.default_rng()

############################################################################################################
# preprocessing                                                                                            #
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

def EncodeSeqToMono_4D(dna_list):
    ''' Encode a list of DNA sequences to a list of 4xL "images" '''
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
    ''' Encode a list of DNA sequences to a list of 64xL "images" '''
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
# training models                                                                                          #
############################################################################################################

def train_trinucleotide_model(data_64d, labels):
    conv = Conv(embedding_dim=64, filter_len=4)
    conv.compile(optimizer='nadam',
                 loss=losses.BinaryCrossentropy(from_logits=True),
                 metrics=[metrics.BinaryCrossentropy(from_logits=True)])
    conv.fit(data_64d, labels)
    preds = conv.predict(data_64d)
    return preds

def train_single_nucleotide_model(data_4d, labels):
    conv = Conv(embedding_dim=4, filter_len=4)
    conv.compile(optimizer='nadam',
                 loss=losses.BinaryCrossentropy(from_logits=True),
                 metrics=[metrics.BinaryCrossentropy(from_logits=True)])
    conv.fit(data_4d, labels)
    preds = conv.predict(data_4d)
    return preds

def train_donor_models(data, data_up, data_down, labels):
    ''' upstream: trinucleuotide
    downstream: single nucleuotide '''
    data_surrounding_4d = EncodeSeqToMono_4D(data)
    data_up_64d = EncodeSeqToTri_64D(data_up)
    data_down_4d = EncodeSeqToMono_4D(data_down)

    # surrounding
    surrounding_predictions = train_single_nucleotide_model(data_surrounding_4d, labels)

    # upstream
    up_predictions = train_trinucleotide_model(data_up_64d, labels)

    # downstream
    down_predictions = train_single_nucleotide_model(data_down_4d, labels)

    # input (surrounding + upstream + downstream) to mlp
    combined_data = tf.concat((surrounding_predictions,
                               up_predictions,
                               down_predictions),
                               axis=1)
    mlp = MLP()
    mlp.compile(optimizer='adam',
                loss=losses.BinaryCrossentropy(from_logits=True),
                metrics=[metrics.BinaryCrossentropy(from_logits=True)])
    mlp.fit(combined_data, labels)
    # preds = mlp.predict(combined_data)
    loss, acc = mlp.evaluate(combined_data, labels)
    return acc

def train_acceptor_models(data, data_up, data_down, labels):
    ''' upstream: single nucleuotide
        downstream: trinucleuotide '''
    data_surrounding_4d = EncodeSeqToMono_4D(data)
    data_up_4d = EncodeSeqToMono_4D(data_up)
    data_down_64d = EncodeSeqToTri_64D(data_down)

    # surrounding
    surrounding_predictions = train_single_nucleotide_model(data_surrounding_4d, labels)

    # upstream
    up_predictions = train_single_nucleotide_model(data_up_4d, labels)

    # downstream
    down_predictions = train_trinucleotide_model(data_down_64d, labels)

    # input (surrounding + upstream + downstream) to mlp
    combined_data = tf.concat((surrounding_predictions,
                               up_predictions,
                               down_predictions),
                               axis=1)
    mlp = MLP()
    mlp.compile(optimizer='adam',
                loss=losses.BinaryCrossentropy(from_logits=True),
                metrics=[metrics.BinaryCrossentropy(from_logits=True)])
    mlp.fit(combined_data, labels)
    # preds = mlp.predict(combined_data)
    loss, acc = mlp.evaluate(combined_data, labels)
    return acc

############################################################################################################
# main                                                                                                     #
############################################################################################################

def main():
    ''' This script trains the Deep Splice models
        given a DNA sequnce with length 602 and Splice
        site in 300-301 positions :...300N...SS... 300N... '''

    # generate random data for now
    data = rng.choice(['A', 'C', 'G', 'T'], (20, 602))

    # data = TextToList('data/___.fa')
    data = RemoveNonAGCT(data)

    begin = 0
    end = 602
    sig_str = 300
    sig_end = 302
    data_up, data_down = split_up_down(data, sig_str, sig_end, begin, end)

    # generate random labels for now
    labels = tf.constant([rng.choice([[1., 0.], [0., 1.]]) for _ in data])

    # Train the donor models
    don_acc = train_donor_models(data, data_up, data_down, labels)
    print(don_acc)

    # Train the acceptor models
    # acc_acc = train_acceptor_models(data, data_up, data_down, labels)

    # TODO: probably want to save models to file and collect metrics

    return

if __name__ == "__main__":
    main()
