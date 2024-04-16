import argparse
import os   

from data import load_alignment


def is_fasta(path: str) -> bool:
    return path.lower().endswith("fa") or path.lower().endswith("fasta")


def alignment_encoding(aln_dir: str):
    for aln in [file for file in os.listdir(aln_dir) if is_fasta(file)]:

        # This function is applied to every fasta file present the input directory
        tensor, ids = load_alignment(os.path.join(aln_dir, aln), True)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict phylogenetic trees from MSAs "
            "using the ML4Phylo neural network"
        )
    )
    parser.add_argument(
        "alidir",
        type=str,
        help="path to input directory containing the\
    .fasta alignments",
    )
    args = parser.parse_args()

    alignment_encoding(args.alidir)

    print("\nDone!")


if __name__ == "__main__":
    main()
