import tensorflow as tf
from keras.metrics import Precision, Recall
from models import Conv, MLP
from preprocess import *
import argparse

############################################################################################################
# training models                                                                                          #
############################################################################################################

def train_trinucleotide_model(data_64d, labels):
    conv = Conv(embedding_dim=64, filter_lens=[3, 5, 10, 21, 31, 41, 50, 61])
    conv.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
    conv.fit(data_64d,
             labels,
             epochs=5,
             batch_size=128,
             validation_split=0.1)
    return conv

def train_single_nucleotide_model(data_4d, labels):
    conv = Conv(embedding_dim=4, filter_lens=list(range(1, 10)))
    conv.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
    conv.fit(data_4d,
             labels,
             epochs=5,
             batch_size=128,
             validation_split=0.1)
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
    final_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', Precision(), Recall()])
    final_model.fit(combined_data,
                    labels,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.1)
    final_model.save('models/donor/donor_final.keras')
    final_model.evaluate(combined_data, labels)

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
    final_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', Precision(), Recall()])
    final_model.fit(combined_data,
                    labels,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.1)
    final_model.save('models/acceptor/acceptor_final.keras')
    final_model.evaluate(combined_data, labels)

############################################################################################################
# main                                                                                                     #
############################################################################################################

def main(train_donor=False, train_acceptor=False):
    ''' This script trains the Deep Splice models
        given a DNA sequnce with length 602 and Splice
        site in 300-301 positions :...300N...SS... 300N... '''

    if train_donor and train_acceptor:
        raise ValueError('Cannot train both donor and acceptor models simultaneously')

    '''

    data, labels = get_np_array()

    begin = 0
    end = 602
    sig_str = 300
    sig_end = 302

    data_dir = 'data/'
    # fasta_file = data_dir + 'C. Elegans/c_elegans_genome.fa'
    # gtf_file = data_dir + 'C. Elegans/c_elegans_genome.gtf'
    # fasta_file = data_dir + 'Camel/camel_genome-004.fa'
    # gtf_file = data_dir + 'Camel/camel_genome.gtf'
    fasta_file = data_dir + 'Human/human_genome-002.fa'
    gtf_file = data_dir + 'Human/Human_genome.gtf'

    chromosome = readFASTA_by_chromosome(fasta_file)[:500000]
    ss_df = get_SS_data(gtf_file, 'start' if train_acceptor else 'end')

    data, labels = encodeWindows(ss_df, chromosome, end - begin, sig_str, sig_end)
    data, labels = RemoveNonAGCT(data, labels)
    data_up, data_down = split_up_down(data, sig_str, sig_end, begin, end)
    labels = tf.keras.utils.to_categorical(labels)

    '''

    begin = 0
    end = 401
    sig_str = 200
    sig_end = 201

    path = '../DeepSplicer/ensembl_seqs/homo_sapiens/'
    data, labels = get_np_array(path, 'acceptor' if train_acceptor else 'donor')
    print(data.shape, labels.shape)
    data, labels = RemoveNonAGCT(data, labels)
    data_up, data_down = split_up_down(data, sig_str, sig_end, begin, end)
    labels = tf.keras.utils.to_categorical(labels)

    if train_acceptor:
        train_acceptor_models(data, data_up, data_down, labels)
    elif train_donor:
        train_donor_models(data, data_up, data_down, labels)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_donor", action="store_true", help="Train donor models")
    parser.add_argument("--train_acceptor", action="store_true", help="Train acceptor models")
    args = parser.parse_args()
    main(args.train_donor, args.train_acceptor)
