# Deep-Learning-Final-Project
Reimplementing deep learning architecture [Splice2Deep](https://www.sciencedirect.com/science/article/pii/S2590158320300097#s0030) for identifying splice sites in ENSEMBL data. Final implementation uses data from [Deep Splicer](https://www.mdpi.com/2073-4425/13/5/907).

## Data

[Raw ENSEMBL data](https://drive.google.com/drive/folders/1g_PgqlBnby8TVmDwQRrBCQEeMgtTnjL1?usp=sharing)

[Deep Splicer repo](https://github.com/ElisaFernandezCastillo/DeepSplicer)

## Environment

To create an environment, run

    conda env create -f env.yml

## Running

Run

    python train.py --train_donor
or

    python train.py --train_acceptor

to train the respective set of models. Each model will be saved to `/models/[SS type]`.

Running

    python evaluate.py --eval_donor
and

    python evalute.py --eval_acceptor

will load the given models if they have been saved to `/models` then evaluate them on all the organisms.