import os
import argparse
from tqdm import tqdm
from data import _parse_alignment

def is_fasta(path: str) -> bool:
    return path.lower().endswith("fasta") or path.lower().endswith("fa")

def alignments_intervals(in_dir, out_dir, blocks, block_size, interval_size):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for alignment in (pbar := tqdm([file for file in os.listdir(in_dir) if is_fasta(file)])):
        identifier = alignment.split(".")[0]
        pbar.set_description(f"Processing {identifier}")
        
        alignment_dict = _parse_alignment(os.path.join(in_dir, alignment))
        output = ""

        for seq_name, sequence in alignment_dict.items():
            seq = break_into_blocks(sequence, blocks, block_size, interval_size)
            output += f">{seq_name}\n{seq}\n"
        
        with open(os.path.join(out_dir, f"{identifier}.fasta"), "w") as fout:
            fout.write(output)

def break_into_blocks(sequence, blocks, block_size, interval_size):
    n_blocks = 0
    seq = ""
    for i in range(0, len(sequence), block_size + interval_size):
        if i + block_size > len(sequence) or n_blocks >= blocks:
            break
        
        current_block = sequence[i:i + block_size]
        n_blocks += 1
        
        seq += current_block

    return seq

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-in",
        "--input",
        required=True,
        type=str,
        help="path to input directory containing the\
    .fasta files",
    )
    parser.add_argument(
        "-out", 
        "--output", 
        required=True, 
        type=str,
        help="path to output directory"
    )
    parser.add_argument(
        "-b",
        "--blocks",
        required=True,
        type=int,
        help="number of blocks of sequences required",
    )
    parser.add_argument(
        "-s",
        "--block_size",
        required=True,
        type=int,
        help="size of the blocks of sequences required",
    )
    parser.add_argument(
        "-i",
        "--interval",
        required=True,
        type=int,
        help="size of the interval between blocks of sequences",
    )
    args = parser.parse_args()

    alignments_intervals(args.input, args.output, args.blocks, args.block_size, args.interval)

if __name__ == "__main__":
    main()