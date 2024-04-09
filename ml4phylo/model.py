"""The phyloformer module contains the Phyloformer network as well as functions to 
create and load instances of the network from disk
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from scipy.special import binom
from utils import println


class AttentionNet(nn.Module):
    """ML4Phylo Network"""

    def __init__(
        self,
        n_blocks: int = 1,
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0.0,
        device: str = "cpu",
        n_seqs: int = 3,
        seq_len: int = 7,
        **kwargs
    ):
        """Initializes internal Module state

        Parameters
        ----------
        n_blocks : int, optional
            Number of blocks in transformer, by default 1
        n_heads : int, optional
            Number of heads in multi-head attention, by default 4
        h_dim : int, optional
            Hidden dimension, by default 64
        dropout : float, optional
            Droupout rate, by default 0.0
        device : str, optional
            Device for model ("cuda" or "cpu"), by default "cpu"
        n_seqs : int, optional
            Number of sequences in input alignments, by default 20
        seq_len : int, optional
            Length of sequences in input alignment, by default 200

        Returns
        -------
        AttentionNet
            Functional instance of AttentionNet for inference/fine-tuning

        Raises
        ------
        ValueError
            If h_dim is not divisible by n_heads
        """

        if h_dim % n_heads != 0:
            raise ValueError(
                "The embedding dimension (h_dim) must be divisible"
                "by the number of heads (n_heads)!"
            )

        super(AttentionNet, self).__init__()
        # Initialize variables
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.h_dim = h_dim
        self.dropout = dropout
        self.device = device

        self._init_seq2pair(n_seqs, seq_len)

        # Initialize Module lists
        self.rowAttentions = nn.ModuleList()
        self.columnAttentions = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.fNNs = nn.ModuleList()

        layers_1_1 = [
            nn.Conv2d(in_channels=22, out_channels=h_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        ]
        self.block_1_1 = nn.Sequential(*layers_1_1)
        
    def _init_seq2pair(self, n_seqs: int, seq_len: int):
        """Initialize Seq2Pair matrix"""

        print("------------Seq2Pair------------")

        self.n_seqs = n_seqs
        self.seq_len = seq_len

        # Calculate all possible combinations of 2 sequences
        self.n_pairs = int(binom(n_seqs, 2))

        println("Number os pairs: ", self.n_pairs)

        # Create a tensor with zeros of dimensions (n_pairs, n_seqs)
        seq2pair = torch.zeros(self.n_pairs, self.n_seqs)

        """
            Iterates over the created tensor and places the value 1
            in the positions that indicate which sequences belong to each pair.

            For example: The pair 1 will be constituted by sequence 1 and 2, thus
            the positions [0,0] and [0,1] of the tensor will have the value 1.

            In our example the tensor will look like this:
                      seqs
                    [1, 1, 0]
              pairs [1, 0, 1]
                    [0, 1, 1]
        """
        k = 0
        for i in range(self.n_seqs):
            for j in range(i + 1, self.n_seqs):
                seq2pair[k, i] = 1
                seq2pair[k, j] = 1
                k = k + 1

        println("Seq2Pair tensor:", seq2pair)

        print("------------Seq2Pair Done------------")

        self.seq2pair = seq2pair.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Any]]:
        """Does a forward pass through the ML4Phylo network

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (shape 1\*22\*n_seqs\*seq_len)

        Returns
        -------
        torch.Tensor
            Output tensor (shape 1\*n_pairs)
        List[Any]
            Attention maps

        Raises
        ------
        ValueError
            If the tensors aren't the right shape
        """
        attentionmaps = []
        # 2D convolution that gives us the features in the third dimension
        # (i.e. initial embedding of each amino acid)
        out = self.block_1_1(x)

        println("Model after first layer:", out)

        out = torch.matmul(self.seq2pair, out.transpose(-1, -2))  # pair representation

        println("Output Model:", out)

        # From here on the tensor has shape (batch_size,features,nb_pairs,seq_len), all
        # the transpose/permute allow to apply layernorm and attention over the desired
        # dimensions and are then followed by the inverse transposition/permutation
        # of dimensions

        return out

    def _get_architecture(self) -> Dict[str, Any]:
        """Returns architecture parameters of the model

        Returns
        -------
        Dict[str, Any]
            Dictionnary containing model architecture
        """
        return {
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "h_dim": self.h_dim,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "n_seqs": self.n_seqs,
        }

    def save(self, path: str) -> None:
        """Saves the model parameters to disk

        Parameters
        ----------
        path : str
            Path to save the model to
        """
        torch.save(
            {
                "architecture": self._get_architecture(),
                "state_dict": self.state_dict(),
            },
            path,
        )