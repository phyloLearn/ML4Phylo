import os
import argparse
from tqdm import tqdm
from data import _parse_alignment

def is_fasta(path: str) -> bool:
    return path.lower().endswith("fasta") or path.lower().endswith("fa")

def sequence_to_typing(seq, gene_dic, total_blocks,  block_size, interval_block_size):
    n_blocks = 0
    typing_seq = []
    
    for char in range(0, len(seq), block_size + interval_block_size):
        if char + block_size > len(seq) and n_blocks >= total_blocks:
            break
        
        current_block = seq[char:char + block_size]
        n_blocks += 1
        
        current_gene = "gene_" + str(n_blocks)
        
        if current_gene not in gene_dic:
            gene_dic[current_gene] = {current_block: 1}
        elif current_block not in gene_dic[current_gene]:
            gene_dic[current_gene][current_block] = len(gene_dic[current_gene]) + 1
            
            
        typing_seq.append(gene_dic[current_gene][current_block])

    return typing_seq
    
def fasta_to_typing(total_blocks, block_size, interval_block_size, alignment):
    gene_dict = {}
    typing_seqs = {}
    
    for seq_name, seq in alignment.items():
        typing_data = sequence_to_typing(seq, gene_dict, total_blocks, block_size, interval_block_size)
        typing_seqs[seq_name] = typing_data
        
    return typing_seqs


def simulate_typing_data(in_dir, out_dir, blocks, block_size, interval_size):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    counter = 0
    
    for alignment in (pbar := tqdm([file for file in os.listdir(in_dir) if is_fasta(file)])):
        base = alignment.split(".")[0]
        pbar.set_description(f"Processing {base}")
        
        alignment_dict = _parse_alignment(os.path.join(in_dir, alignment))

        typing_data_dict = fasta_to_typing(blocks, block_size, interval_size, alignment_dict)
        
        output = "\n" # First line needs to be empty to represent the gene names
        
        for typing_seq in typing_data_dict.values():
            typing_seq_string = '\t'.join([str(gene_id) for gene_id in typing_seq])
            output += typing_seq_string + '\n'

        with open(os.path.join(out_dir, "typing_data_" + str(counter) + ".txt"), "w") as fout:
            fout.write(output)
    
        counter += 1

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
        "-out", "--output", 
        required=True, 
        type=str,
        help="path to output directory"
    )
    parser.add_argument(
        "-b",
        "--blocks",
        required=True,
        type=int,
        help="number of blocks in the typing data",
    )
    parser.add_argument(
        "-s",
        "--block_size",
        required=True,
        type=int,
        help="size of the blocks in the typing data",
    )
    parser.add_argument(
        "-i",
        "--interval",
        required=True,
        type=int,
        help="size of the interval between blocks in the typing data",
    )
    args = parser.parse_args()

    simulate_typing_data(args.input, args.output, args.blocks, args.block_size, args.interval)

if __name__ == "__main__":
    main()
