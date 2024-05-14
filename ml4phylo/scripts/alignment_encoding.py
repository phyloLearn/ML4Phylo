import argparse
import os   

from data_typing import load_typing


def is_txt(path: str) -> bool:
    return path.lower().endswith("txt")


def alignment_encoding(aln_dir: str):
    align_encoded = 0
    
    for aln in [file for file in os.listdir(aln_dir) if is_txt(file)]:

        # This function is applied to every txt file present the input directory
        load_typing(os.path.join(aln_dir, aln))
        
        align_encoded += 1
    
    print(f"Encoded {align_encoded} alignments")

def main():
    parser = argparse.ArgumentParser(description=("Encodes typing data files into tensors"))
    parser.add_argument(
        "alidir",
        type=str,
        help="path to input directory containing the alignments",
    )
    args = parser.parse_args()

    alignment_encoding(args.alidir)

    print("\nDone!")


if __name__ == "__main__":
    main()
