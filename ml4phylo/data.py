from typing import Dict, List, Tuple

import numpy as np
import torch
from Bio import SeqIO

AMINO_ACIDS = np.array(list("ATGC"))
# AMINO_ACIDS = np.array(list("ARNDCQEGHILKMFPSTWYVX-"))

def load_alignment(path: str) -> Tuple[torch.Tensor, List[str]]:
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

    tensor = []
    parsed = _parse_alignment(path)
    for sequence in parsed.values():
        tensor.append(                                              
            torch.from_numpy(_sequence_to_one_hot(sequence)).t().view(4, 1, -1) # Change back to 22
        )

    return torch.cat(tensor, dim=1).transpose(-1, -2), list(parsed.keys())


def _parse_alignment(path: str) -> Dict[str, str]:
    """Parser a fasta alignment

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


def _sequence_to_one_hot(seq: str) -> np.ndarray:
    """Encode an amino acid sequence with one-hot encoding

    Parameters
    ----------
    seq : str
        Sequence of amino acids to encode

    Returns
    -------
    np.ndarray
        Encoded sequence (shape 22\*seq_len)
    """
    return np.array([(AMINO_ACIDS == aa).astype(int) for aa in seq])
