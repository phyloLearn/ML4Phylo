import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import skbio
import torch
from ete3 import Tree
from Bio import SeqIO
from utils import println
from torch.utils.data import Dataset

NUCLEOTIDES = np.array(list("ATGC")) # used for our alignment example
AMINO_ACIDS = np.array(list("ARNDCQEGHILKMFPSTWYVX-"))

class TensorDataset(Dataset):
    """A Dataset class to train ML4Phylo networks"""

    def __init__(self, directory: str, filter: Optional[List[str]] = None):
        """Instanciates a TensorDataset

        Parameters
        ----------
        directory : str
            Path to the directory containing .tensor_pair files generated by the
            `make_tensors` script.
        filter: List[str], optional
            List of tensor pair names to keep (useful if you keep training and
            validation tensors in the same directory), default is None

        Returns
        -------
        TensorDataset
            A instance of TensorDataset for training ML4Phylo
        """
        super(TensorDataset, self).__init__()
        self.directory = directory
        self.pairs = [
            filepath
            for filepath in os.listdir(self.directory)
            if filepath.endswith(".tensor_pair")
        ]
        if filter is not None:
            self.pairs = [id for id in self.pairs if id in filter]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = torch.load(os.path.join(self.directory, (self.pairs[index])))
        return pair["X"], pair["y"]


def load_alignment(path: str, isNucleotides: bool = False) -> Tuple[torch.Tensor, List[str]]:
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

    # Iterate over all the sequences present in the dictionary
    for sequence in parsed.values():
        """
            Encodes every sequence obtaining a matrix with 2 dimensions (sequence_length, alphabet_size)
            This matrix stores binary values that represent for each char of the sequence its corresponding
            amino acid or nucleotide.
        """
        one_hot = _sequence_to_one_hot(sequence, isNucleotides)

        # Creates a tensor from the encoded sequence inverting his dimension to (alphabet_size, sequence_length)
        tensor = torch.from_numpy(one_hot).t()

        # Reshapes the tensor to a 3-dimensional one
        reshaped_tensor = tensor.view(alphabet_size, 1, -1)

        tensor_list.append(                                              
            reshaped_tensor
        )

    """
        Concats all the tensors present in the list.
        As tensors are made up of 3 dimensions (alphabet_size, 1, seq_length), it presents (alphabet_size) matrixes.
        After the concatenation the obtained tensor has matrixes with dimension (n_seqs, seq_length), leading to
        a tensor of dimensions (alphabet_size, n_seqs, seq_length).
    """
    concated_tensors = torch.cat(tensor_list, dim=1)

    """
        Finally, the transpose of the last two dimensions is performed,
        resulting in a tensor of dimensions (alphabet_size, seq_length, n_seqs).
    """
    final_tensor = concated_tensors.transpose(-1, -2)

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


def load_tree(path: str) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
    """Loads a tree as a tensor of pairwise distances, digestible by the ML4Phylo
    network

    Parameters
    ----------
    path : str
        Path to the newick file containing the tree

    Returns
    -------
    Tuple[torch.Tensor, List[Tuple[str, str]]]
        a tuple containing:
         - a tensor representing the distance matrix of the tree (shape 1\*n_pairs)
         - a list of tuples of ids indicating between which leafs the distance was
           computed

    """
    distances = _read_distances_from_tree(path)

    tensor, ids = [], []
    for pair, distance in distances.items():
        tensor.append(distance)
        ids.append(pair)

    return (torch.tensor(tensor), ids)


def _read_distances_from_tree(
    path: str, normalize: bool = False
) -> Dict[Tuple[str, str], float]:
    """Reads a phylogenetic tree and returns the corresponding distance matrix

    Parameters
    ----------
    path : str
        Path to the newick file containing the tree
    normalize : bool, optional
        Wether to normalize distances or not, by default False

    Returns
    -------
    Dict[Tuple[str, str], float]
        A dictionary representing the triangular distance matrix with:
         - as keys: a tuple of the leaf ids between which the distance is computed
         - as values: the distances

    """

    """
        The tree is iterated with the objective to calculate all the distances in the tree and register them in a dictionary.
        The ete3 function [leaf1.get_distance(leaf2)] permits the calculation of the distance between 2 nodes/leafs.
    """
    tree = Tree(path)
    distances = dict()
    for i, leaf1 in enumerate(tree):
        for j, leaf2 in enumerate(tree):
            if i < j:
                distances[(leaf1.name, leaf2.name)] = leaf1.get_distance(leaf2)

    """
        Finally, all the distances of the tree are normalized if necessary.
        This normalization process, finds the maximum distance value in the dict and divides every other
        distance by this maximum value.
        This is done to scale all the distances so that they fall between 0 and 1.
        Its a useful method for comparing different trees, as it puts the distances on a consistent scale.
    """
    if normalize:
        diameter = max(distances.values())
        for key in distances:
            distances[key] /= diameter

    return distances


def write_dm(dm: skbio.DistanceMatrix, path: str):
        """Write a distance matrix to disk in the square Phylip matrix format

        Parameters
        ----------
        dm : skbio.DistanceMatrix
            Distance matrix to save
        path : str
            Path where to save the matrix
        """

        with open(path, "w+") as file:
            file.write(f"{len(dm.ids)}\n")
            for id, dists in zip(dm.ids, dm.data):
                line = " ".join(str(dist) for dist in dists)
                file.write(f"{id}     {line}\n")