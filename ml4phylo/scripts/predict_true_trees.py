import argparse
import os

import skbio
import numpy as np
from ete3 import Tree
from tqdm import tqdm
from data import _parse_alignment, _parse_typing
from sklearn.metrics import DistanceMetric
from data import DataType


def is_fasta(path: str) -> bool:
    return path.lower().endswith("fa") or path.lower().endswith("fasta")

def is_txt(path: str) -> bool:
    return path.lower().endswith("txt")


def predict_true_trees(in_dir: str, out_dir: str, data_type: DataType):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for aln in (pbar := tqdm([file for file in os.listdir(in_dir) if is_fasta(file) or is_txt(file)])):
        identifier = aln.split(".")[0]
        pbar.set_description(f"Processing {identifier}")
        
        path = os.path.join(in_dir, aln)
        matrix, alignment = true_trees_typing(path) if data_type == DataType.TYPING else true_trees_sequences(path)
        
        dist_matrix = skbio.DistanceMatrix(matrix, ids=list(alignment.keys()))
        newick_tree = skbio.tree.nj(dist_matrix, result_constructor=str)
        tree = Tree(newick_tree)
        
        tree.write(outfile=os.path.join(out_dir, f"{identifier}.pf.nwk"))


def hamming_distance(seq1, seq2):
    """Calculate the Hamming distance between two sequences."""
    
    assert len(seq1) == len(seq2), "Sequences must be of the same length"
    return sum(char1 != char2 for char1, char2 in zip(seq1, seq2))


def true_trees_sequences(path):
    alignment = _parse_alignment(path)

    sequences = [value for value in alignment.values()]

    n = len(sequences)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            dist = hamming_distance(sequences[i], sequences[j])
            matrix[i][j] = dist
            matrix[j][i] = dist
    
    return matrix, alignment


def true_trees_typing(path):
    alignment = _parse_typing(path)
    dist = DistanceMetric.get_metric("hamming")

    X = [value for value in alignment.values()]

    return dist.pairwise(X, X), alignment

DATA_TYPES = DataType.toList()

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict phylogenetic trees from MSAs "
            "using the ML4Phylo neural network"
        )
    )
    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        help="path to input directory containing corresponding\
            data files: [.fasta for alignments or .txt for typing data]",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        required=True,
        help="path to the output directory were the\
    .nwk tree files will be saved",
    )
    parser.add_argument(
        "-dt",
        "--data_type",
        required=False,
        type=str,
        default=DataType.AMINO_ACIDS.name,
        choices=DATA_TYPES,
        help=f"type of input data. Choices: {DATA_TYPES}",
    )
    args = parser.parse_args()

    data_type = args.data_type.upper()

    if data_type not in [type.name for type in DataType]:
        raise ValueError(f"Invalid data type: {args.data_type}")

    predict_true_trees(args.indir, args.outdir, DataType[data_type])

    print("\nDone!")


if __name__ == "__main__":
    main()
