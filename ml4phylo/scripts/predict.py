import argparse
import os

import torch
from tqdm import tqdm

from data import load_data, write_dm, DataType
from model import AttentionNet, load_model


def is_fasta(path: str) -> bool:
    return path.lower().endswith("fa") or path.lower().endswith("fasta")

def is_txt(path: str) -> bool:
    return path.lower().endswith("txt")


def make_predictions(model: AttentionNet, aln_dir: str, out_dir: str, save_dm: bool, data_type: DataType):
    for aln in (pbar := tqdm([file for file in os.listdir(aln_dir) if is_fasta(file) or is_txt(file)])):
        identifier = aln.split(".")[0]
        pbar.set_description(f"Processing {identifier}")

        tensor, ids = load_data(os.path.join(aln_dir, aln), data_type)

        # check if model input settings match alignment
        _, data_len, n_data = tensor.shape
        if model.data_len != data_len or model.n_data != n_data:
            model._init_seq2pair(n_data=n_data, data_len=data_len)

        dm = model.infer_dm(tensor, ids)
        if save_dm:
            write_dm(dm, os.path.join(out_dir, f"{identifier}.pf.dm"))
        tree = model.infer_tree(tensor, dm=dm)
        tree.write(outfile=os.path.join(out_dir, f"{identifier}.pf.nwk"))

DATA_TYPES = DataType.toList()

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict phylogenetic trees from MSAs "
            "using the ML4Phylo neural network"
        )
    )
    parser.add_argument(
        "-dd",
        "--datadir",
        type=str,
        help="path to input directory containing corresponding\
            data files: [.fasta for alignments or .txt for typing data]",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="path to the output directory were the\
    .tree tree files will be saved (default: datadir)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default="seqgen",
        help=(
            "path to the NN model state dictionary, path/to/model.pt"
        ),
    )
    parser.add_argument(
        "-g",
        "--gpu",
        required=False,
        action="store_true",
        help="use the GPU for inference (default: false)",
    )
    parser.add_argument(
        "-d",
        "--dm",
        required=False,
        action="store_true",
        help="save predicted distance matrix (default: false)",
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

    out_dir = args.output if args.output is not None else args.datadir
    if out_dir != "." and not os.path.exists(out_dir):
        os.mkdir(out_dir)

    device = "cpu"
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    elif args.gpu and torch.backends.mps.is_available():
        device = "mps"

    model = None
    if args.model is not None:
        if not os.path.isfile(args.model):
            raise ValueError(f"The specified model file: {args.model} does not exist")
        model = load_model(args.model, device=device)
    else:
        raise ValueError("You must specify the model to use")

    model.to(device)

    print("ML4Phylo predict:\n")
    print(f"Predicting trees for alignments in {args.datadir}")
    print(f"Using the {args.model} model on {device}")
    print(f"Saving predicted trees in {out_dir}")
    if args.dm:
        print(f"Saving Distance matrices in {out_dir}")
    print()

    make_predictions(model, args.datadir, out_dir, args.dm, DataType[data_type])

    print("\nDone!")


if __name__ == "__main__":
    main()
