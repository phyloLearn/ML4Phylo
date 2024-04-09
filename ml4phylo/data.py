from typing import Dict, List, Tuple

import numpy as np
import torch
from Bio import SeqIO
from utils import println

NUCLEOTIDES = np.array(list("ATGC")) # used for our alignment example
AMINO_ACIDS = np.array(list("ARNDCQEGHILKMFPSTWYVX-"))

def load_alignment(path: str, isNucleotides: bool) -> Tuple[torch.Tensor, List[str]]:
    """Loads an alignment into a tensor digestible by the Ml4Phylo network

    Parameters
    ----------
    path : str
        Path to a fasta file containing the alignment

    Returns
    -------
    Tuple[torch.Tensor, List[str]]
        a tuple containing:
         - a tensor representing the alignment (shape 22 * seq_len * n_seq)
         - a list of ids of the sequences in the alignment

    """
    print("--------------ENCODING--------------")

    # check if the sequences use the aminoacids alphabet or the nucleotides
    alphabet_size = len(NUCLEOTIDES) if isNucleotides else len(AMINO_ACIDS)
    tensor_list = []
    parsed = _parse_alignment(path)

    println("Alignment parsed:", parsed)

    # Iterate over all the sequences present in the dictionary
    for sequence in parsed.values():
        """
            Encodes every sequence obtaining a matrix with 2 dimensions (sequence_length, alphabet_size)
            This matrix stores binary values that represent for each char of the sequence its corresponding
            amino acid or nucleotide.
        """
        one_hot = _sequence_to_one_hot(sequence, isNucleotides)

        println("One hot encoded seq:", one_hot)

        # Creates a tensor from the encoded sequence inverting his dimension to (alphabet_size, sequence_length)
        tensor = torch.from_numpy(one_hot).t()

        println("Tensor obtained from the encoded seq:", tensor)

        # Reshapes the tensor to a 3-dimensional one
        reshaped_tensor = tensor.view(alphabet_size, 1, -1)

        println("Reshaped:", reshaped_tensor)

        tensor_list.append(                                              
            reshaped_tensor
        )

    println("All sequences:", tensor_list)

    """
        Concats all the tensors present in the list.
        As tensors are made up of 3 dimensions (alphabet_size, 1, seq_length), it presents (alphabet_size) matrixes.
        After the concatenation the obtained tensor has matrixes with dimension (seq_length, n_seqs), leading to
        a tensor of dimensions (alphabet_size, n_seqs, seq_length).
    """
    concated_tensors = torch.cat(tensor_list, dim=1)

    println("All tensors concat:", concated_tensors)

    """
        Finally, the transpose of the last two dimensions is performed,
        resulting in a tensor of dimensions (alphabet_size, seq_length, n_seqs).
    """
    final_tensor = concated_tensors.transpose(-1, -2)

    println("Final Tensor:", final_tensor)

    print("--------------ENCODING DONE--------------")
    return final_tensor, list(parsed.keys())


def _parse_alignment(path: str) -> Dict[str, str]:
    """Parses one fasta alignment

    Parameters
    ----------
    path : str
        Path to .fasta alignment file

    Returns
    -------
    Dict[str,str]
        A dictionnary with ids as keys and sequence as values
    """
    return {record.id: str(record.seq) for record in SeqIO.parse(path, format="fasta")}


def _sequence_to_one_hot(seq: str, isNucleotides: bool) -> np.ndarray:
    """Encode a sequence with one-hot encoding

    Parameters
    ----------
    seq : str
        Sequence to encode

    Returns
    -------
    np.ndarray
        Encoded sequence (shape 22 * seq_len)
    """
    alphabet = NUCLEOTIDES if isNucleotides else AMINO_ACIDS
    return np.array([(alphabet == char).astype(int) for char in seq])
