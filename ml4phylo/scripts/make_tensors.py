import argparse
import os

import torch
from tqdm import tqdm

from data import load_tree, load_data, DataType

def make_tensors(tree_dir: str, data_dir: str, out_dir: str, data_type: DataType):
    trees = [file for file in os.listdir(tree_dir) if file.endswith(".nwk")]
    for tree_file in (pbar := tqdm(trees)):
        identifier = tree_file.rstrip(".nwk")
        pbar.set_description(f"Processing {identifier}")
        tree_tensor, _ = load_tree(os.path.join(tree_dir, tree_file))
        data_tensor, _ = load_data(os.path.join(data_dir, f"{identifier}{".txt" if data_type == DataType.TYPING else ".fasta"}"), data_type)

        torch.save(
            {"X": data_tensor, "y": tree_tensor},
            os.path.join(out_dir, f"{identifier}.tensor_pair"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a tensor training set from trees and MSAs"
    )
    parser.add_argument(
        "-t",
        "--treedir",
        required=True,
        type=str,
        help="path to input directory containing the .nwk tree files",
    )
    parser.add_argument(
        "-d",
        "--datadir",
        required=True,
        type=str,
        help="path to input directory containing corresponding data files: [.fasta for alignments or .txt for typing data]",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=".",
        type=str,
        help="path to output directory (default: current directory)",
    )
    parser.add_argument(
        "-dt",
        "--data_type",
        required=False,
        default="AMINO_ACIDS",
        type=str,
        help="data type to encode: [AMINO_ACIDS, NUCLEOTIDES, TYPING]",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    
    if args.data_type not in DataType:
        raise ValueError(f"Invalid data type: {args.data_type}")

    make_tensors(args.treedir, args.datadir, args.output, DataType[args.data_type])


if __name__ == "__main__":
    main()
