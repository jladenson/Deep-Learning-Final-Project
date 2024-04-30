import tensorflow as tf
from keras.models import load_model
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from models import Conv, MLP
import argparse
import pandas as pd
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


def gtf_to_dataframe(gtf_file: str):
    '''Returns a pandas dataframe of the original GTF file'''
    gtf_data = []
    with open(gtf_file) as file:
        for record in parseGTF(file):
            gtf_data.append(record)

    df = pd.DataFrame(gtf_data)
    return df

def parseGTF(gtfFile: str):
    '''Helper for gtf_to_dataframe which reads GTF file and organises it into fields.'''
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
    (in the FASTA file) for exons'''
    exon_df = df[df['feature'] == 'exon']
    start_df = exon_df['start']
    end_df = exon_df['end']
    ss_df = pd.concat([start_df, end_df], axis=1)
    ss_df = ss_df.sort_values(by=['start'], ascending=True)
    return ss_df 

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
# loading models                                                                                           #
############################################################################################################

def load_donor_models(data, data_up, data_down, labels):
    path = 'models/donor/donor_'
    models = ['srdng', 'up', 'down', 'final']
    if (not all(tf.io.gfile.exists(path + model + '.keras') for model in models)):
        return -1
    srdng_model = load_model(path + 'srdng.keras')
    up_model = load_model(path + 'up.keras')
    down_model = load_model(path + 'down.keras')
    final_model = load_model(path + 'final.keras')

    data_srdng_4d = EncodeSeqToMono_4D(data)
    data_up_64d = EncodeSeqToTri_64D(data_up)
    data_down_4d = EncodeSeqToMono_4D(data_down)

    srdng_preds = srdng_model.predict(data_srdng_4d)
    up_preds = up_model.predict(data_up_64d)
    down_preds = down_model.predict(data_down_4d)

    combined_data = tf.concat((srdng_preds,
                               up_preds,
                               down_preds),
                               axis=1)

    loss, acc = final_model.evaluate(combined_data, labels)
    return acc

def load_acceptor_models(data, data_up, data_down, labels):
    path = 'models/acceptor/acceptor_'
    models = ['srdng', 'up', 'down', 'final']
    if (not all(tf.io.gfile.exists(path + model + '.keras') for model in models)):
        return -1

    srdng_model = load_model(path + 'srdng.keras')
    up_model = load_model(path + 'up.keras')
    down_model = load_model(path + 'down.keras')
    final_model = load_model(path + 'final.keras')

    data_srdng_4d = EncodeSeqToMono_4D(data)
    data_up_4d = EncodeSeqToMono_4D(data_up)
    data_down_64d = EncodeSeqToTri_64D(data_down)

    srdng_preds = srdng_model.predict(data_srdng_4d)
    up_preds = up_model.predict(data_up_4d)
    down_preds = down_model.predict(data_down_64d)

    combined_data = tf.concat((srdng_preds,
                            up_preds,
                            down_preds),
                            axis=1)

    loss, acc = final_model.evaluate(combined_data, labels)
    return acc

############################################################################################################
# training models                                                                                          #
############################################################################################################

def train_trinucleotide_model(data_64d, labels):
    conv = Conv(embedding_dim=64, filter_lens=[3, 5, 10, 21, 31, 41, 50, 61])
    conv.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
    conv.fit(data_64d,
             labels,
             epochs=10,
             batch_size=64)
    return conv

def train_single_nucleotide_model(data_4d, labels):
    conv = Conv(embedding_dim=4, filter_lens=list(range(1, 10)))
    conv.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
    conv.fit(data_4d,
             labels,
             epochs=10,
             batch_size=64)
    return conv

def train_donor_models(data, data_up, data_down, labels):
    ''' upstream: trinucleuotide
    downstream: single nucleuotide '''
    data_srdng_4d = EncodeSeqToMono_4D(data)
    data_up_64d = EncodeSeqToTri_64D(data_up)
    data_down_4d = EncodeSeqToMono_4D(data_down)

    # surrounding
    srdng_model = train_single_nucleotide_model(data_srdng_4d, labels)
    srdng_model.save('models/donor/donor_srdng.keras')
    srdng_preds = srdng_model.predict(data_srdng_4d)

    # upstream
    up_model = train_trinucleotide_model(data_up_64d, labels)
    up_model.save('models/donor/donor_up.keras')
    up_preds = up_model.predict(data_up_64d)

    # downstream
    down_model = train_single_nucleotide_model(data_down_4d, labels)
    down_model.save('models/donor/donor_down.keras')
    down_preds = down_model.predict(data_down_4d)

    # input (surrounding + upstream + downstream) to mlp
    combined_data = tf.concat((srdng_preds,
                               up_preds,
                               down_preds),
                               axis=1)
    final_model = MLP()
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    final_model.fit(combined_data,
                    labels,
                    epochs=10,
                    batch_size=64)
    final_model.save('models/donor/donor_final.keras')
    loss, acc = final_model.evaluate(combined_data, labels)
    return acc

def train_acceptor_models(data, data_up, data_down, labels):
    ''' upstream: single nucleuotide
        downstream: trinucleuotide '''
    data_srdng_4d = EncodeSeqToMono_4D(data)
    data_up_4d = EncodeSeqToMono_4D(data_up)
    data_down_64d = EncodeSeqToTri_64D(data_down)

    # surrounding
    srdng_model = train_single_nucleotide_model(data_srdng_4d, labels)
    srdng_model.save('models/acceptor/acceptor_srdng.keras')
    srdng_preds = srdng_model.predict(data_srdng_4d)

    # upstream
    up_model = train_single_nucleotide_model(data_up_4d, labels)
    up_model.save('models/acceptor/acceptor_up.keras')
    up_preds = up_model.predict(data_up_4d)

    # downstream
    down_model = train_trinucleotide_model(data_down_64d, labels)
    down_model.save('models/acceptor/acceptor_down.keras')
    down_preds = down_model.predict(data_down_64d)

    # input (surrounding + upstream + downstream) to mlp
    combined_data = tf.concat((srdng_preds,
                               up_preds,
                               down_preds),
                               axis=1)
    final_model = MLP()
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    final_model.fit(combined_data,
                    labels,
                    epochs=10,
                    batch_size=64)
    final_model.save('models/acceptor/acceptor_final.keras')
    loss, acc = final_model.evaluate(combined_data, labels)
    return acc

############################################################################################################
# main                                                                                                     #
############################################################################################################

def main(train_donor=False, train_acceptor=False):
    ''' This script trains the Deep Splice models
        given a DNA sequnce with length 602 and Splice
        site in 300-301 positions :...300N...SS... 300N... '''

    if train_donor and train_acceptor:
        raise ValueError('Cannot train both donor and acceptor models simultaneously')

    # generate random data for now
    data = rng.choice(['A', 'C', 'G', 'T'], (100, 602))
    # data = TextToList('data/___.fa')

    data = RemoveNonAGCT(data)

    begin = 0
    end = 602
    sig_str = 300
    sig_end = 302
    data_up, data_down = split_up_down(data, sig_str, sig_end, begin, end)

    # generate random labels for now
    labels = tf.constant([rng.choice([[0, 1], [1, 0]]) for _ in data])

    # Train the donor models
    if train_donor:
        don_acc = train_donor_models(data, data_up, data_down, labels)
    else:
        don_acc = load_donor_models(data, data_up, data_down, labels)
    print(f'donor accuracy: {don_acc}')

    # Train the acceptor models
    if train_acceptor:
        acc_acc = train_acceptor_models(data, data_up, data_down, labels)
    else:
        acc_acc = load_acceptor_models(data, data_up, data_down, labels)
    print(f'acceptor accuracy: {acc_acc}')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_donor", action="store_true", help="Train donor models")
    parser.add_argument("--train_acceptor", action="store_true", help="Train acceptor models")
    args = parser.parse_args()
    main(args.train_donor, args.train_acceptor)
