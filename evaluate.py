import tensorflow as tf
from keras.models import load_model
from preprocess import *
import argparse

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
# main                                                                                                     #
############################################################################################################

def main(eval_donor=False, eval_acceptor=False):
    ''' This script evaluates the Deep Splice models
        given a DNA sequnce with length 602 and Splice
        site in 300-301 positions :...300N...SS... 300N... '''

    if eval_donor and eval_acceptor:
        raise ValueError('Cannot evaluate both donor and acceptor models simultaneously')

    gtf_file = 'data/genomic.gtf'
    fasta_file = 'data/GCA_000002985.3_WBcel235_genomic.fna'

    chromosomes = readFASTA_by_chromosome(fasta_file)
    chromosome = chromosomes[0][0:10000]
    ss_df = get_SS_data(gtf_file)
    acc_df, don_df = ss_df['start'], ss_df['end']

    begin = 0
    end = 602
    sig_str = 300
    sig_end = 302

    # evaluate the donor models
    if eval_donor:
        data, labels = encodeWindows(don_df, chromosome, end, sig_str, sig_end)
        data_up, data_down = split_up_down(data, sig_str, sig_end, begin, end)
        don_acc = load_donor_models(data, data_up, data_down, labels)
        print(f'donor accuracy: {don_acc}')

    # evaluate the acceptor models
    if eval_acceptor:
        data, labels = encodeWindows(acc_df, chromosome, end, sig_str, sig_end)
        data_up, data_down = split_up_down(data, sig_str, sig_end, begin, end)
        acc_acc = load_acceptor_models(data, data_up, data_down, labels)
        print(f'acceptor accuracy: {acc_acc}')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_donor", action="store_true", help="Evaluate donor models")
    parser.add_argument("--eval_acceptor", action="store_true", help="Evaluate acceptor models")
    args = parser.parse_args()
    main(args.eval_donor, args.eval_acceptor)