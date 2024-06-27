import argparse
import os   

from data import load_data, DataType

def is_fasta(path: str) -> bool:
    return path.lower().endswith("fa") or path.lower().endswith("fasta")

def is_txt(path: str) -> bool:
    return path.lower().endswith("txt")

def typing_encoding(aln_dir: str, data_type: str):
    align_encoded = 0
    
    for aln in [file for file in os.listdir(aln_dir) if is_txt(file) or is_fasta(file)]:

        # This function is applied to every file present the input directory
        load_data(os.path.join(aln_dir, aln), data_type)
        
        align_encoded += 1
        
    
    print(f"Encoded {align_encoded} alignments")

def main():
    parser = argparse.ArgumentParser(description=("Encodes typing data files into tensors"))
    parser.add_argument(
        "typingdir",
        type=str,
        help="path to input directory containing the typing data",
    )
    parser.add_argument(
        "data_type",
        type=str,
        help="type of data to encode (typing or sequences). Allowed Values: [TYPING, AMINO_ACIDS, NUCLEOTIDES]",
    )
    args = parser.parse_args()

    if args.data_type not in DataType:
        raise ValueError(f"Invalid data type: {args.data_type}")

    typing_encoding(args.typingdir, args.data_type)

    print("\nDone!")


if __name__ == "__main__":
    main()
