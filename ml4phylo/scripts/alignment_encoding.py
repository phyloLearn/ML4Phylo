import argparse
import os

from tqdm import tqdm

from data import load_alignment


def is_fasta(path: str) -> bool:
    return path.lower().endswith("fa") or path.lower().endswith("fasta")


def alignment_encoding(aln_dir: str):
    for aln in (pbar := tqdm([file for file in os.listdir(aln_dir) if is_fasta(file)])):
        base = aln.split(".")[0]
        pbar.set_description(f"Processing {base}")

        tensor, ids = load_alignment(os.path.join(aln_dir, aln))

        print("TENSOR:")
        print(tensor)
        print("IDS:")
        print(ids)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict phylogenetic trees from MSAs "
            "using the Phyloformer neural network"
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
