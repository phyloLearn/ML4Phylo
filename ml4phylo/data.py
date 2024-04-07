from typing import Dict, List, Tuple

import numpy as np
import torch
from Bio import SeqIO

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
    # check if the sequences use the aminoacids alphabet or the nucleotides
    alphabet_size = len(NUCLEOTIDES) if isNucleotides else len(AMINO_ACIDS)
    tensor_list = []
    parsed = _parse_alignment(path)

    print("Alignment parsed:")
    print(parsed)

    # Iterate over all the sequences present in the dictionary
    for sequence in parsed.values():
        """
            Encodes every sequence obtaining a matrix with 2 dimensions (alphabet_size, sequence_length)
            This matrix presents binary values that represent for each char of the sequence its corresponding
            amino acid or nucleotide.
        """
        one_hot = _sequence_to_one_hot(sequence, isNucleotides)

        print("One hot encoded seq:")
        print(one_hot)

        # Creates a tensor from the encoded sequence inverting his dimension to (sequence_length, alphabet_size)
        tensor = torch.from_numpy(one_hot).t()

        print("Tensor obtained from the encoded seq:")
        print(tensor)

        # Reshapes the tensor to a 3-dimensional one
        reshaped_tensor = tensor.view(alphabet_size, 1, -1)

        print("Reshaped:")
        print(reshaped_tensor)

        tensor_list.append(                                              
            reshaped_tensor
        )

    print("All sequences:")
    print(tensor_list)    

    """
        Concats all the tensors present in the list.
        As tensors are made up of 3 dimensions (alphabet_size, 1, seq_length), it presents (alphabet_size) layers.
        The concatenation is done between these layers, resulting in a tensor with (n_seqs) values in each layer.
        Each of these values have a dimension of (seq_length).

        At the end of this process we get a tensor with dimensions (alphabet_size, n_seqs, seq_length)
    """
    concated_tensors = torch.cat(tensor_list, dim=1)

    print("All tensors concat:")
    print(concated_tensors)

    """
        Finally, the transpose of the last two dimensions is performed,
        resulting in a tensor of dimensions (alphabet_size, seq_length, n_seqs).
    """
    final_tensor = concated_tensors.transpose(-1, -2)

    print("Final Tensor:")
    print(final_tensor)

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
        Encoded sequence (shape 22\*seq_len)
    """
    alphabet = NUCLEOTIDES if isNucleotides else AMINO_ACIDS
    return np.array([(alphabet == char).astype(int) for char in seq])
