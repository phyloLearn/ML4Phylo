<img src="./resources/ML4Phylo_logo.png" alt="ML4Phylo" width="500"/>

# ML4Phylo
Machine Learning techniques applied to Phylogenetic Analysis

# Dependencies
Python 3.9 minimum.

The dependencies to be installed are as follows:
- scipy 
- numpy 
- ete3 
- biopython 
- dendropy 
- scikit-bio 
- tqdm

To install any of these packages you only need to run the command:

```txt
    pip install <package name>
```

## Torch Dependency
You should also install the torch package. 
If you intend to train the model using torch with CUDA, instead of just installing torch you should run one of these commands:

### CUDA 11.8
```txt
   pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 
```

### CUDA 12.1
```txt
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

# Seq-Gen (Random genetic sequence generator)
To install Seq-Gen:
https://github.com/rambaut/Seq-Gen/releases/tag/1.3.4

The Seq-Gen executable for Windows is already available in the repository. If a Linux executable is needed, a new one should be compiled for Linux.

# Instructions to train the neural model 
After having the dependencies and Seq-Gen ready, you can run the ML4Phylo scripts:

In config.json file, you should checked first what device you're planning on using: "cpu" or "cuda". (default is "cuda")
You should open the command line through the console.bat present in the repo to set the necessary environment variable.

## Simulate the trees
```txt
simulate_trees
    --nleaves <number of leaves in each tree> (default 20)
    --ntrees <number of trees> (default 20)
    --topology <tree topology> (default uniform)
    --output <output directory>
    --branchlength <branch lenght distribution> (default uniform)
```
Example: python .\ml4phylo\scripts\simulate_trees.py ....args......

## Simulate the alignments (sequences)
```txt
simulate_alignments
    --input <input directory with the .nwk tree files>
    --output <output directory>
    --length <length of the simulated sequences> (default 200)
    --seqgen <path to Seq-Gen executable>
    --model <model of evolution> (default PAM)
```
Example: python .\ml4phylo\scripts\simulate_alignments.py ....args......

## Transforming genetic sequences into typing data
```txt
simulate_typing_data
    --input <input directory with the .fasta files>
    --output <output directory>
    --blocks <number of blocks for typing data>
    --block_size <size of each block>
    --interval <size of the interval between blocks>
```
Example: python .\ml4phylo\scripts\simulate_typing_data.py ....args......

## Creating tensors for the neural model training

### With sequences
```txt
make_tensors
    --treedir <input directory with the .nwk tree files>
    --datadir <input directory containing corresponding data files: [.fasta for alignments or .txt for typing data]>
    --output <output directory>
    --data_type <type of input data. Possible values: [AMINO_ACIDS, NUCLEOTIDES, TYPING]> (default: AMINO_ACIDS)
```
Example: python .\ml4phylo\scripts\make_tensors.py ....args......

## Train the neural model
```txt
train
    --input <input directory containing the tensor pairs on which the model will be trained>
    --validation <input directory containing the tensor pairs on which the model will be evaluated.>
                    (If left empty 10% of the training set will be used as validation data.)
    --config <configuration json file for the hyperparameters.>
    --output <output directory where the model parameters and the metrics will be saved.>
    --data_type <type of input data. Possible values: [AMINO_ACIDS, NUCLEOTIDES, TYPING].> (default: AMINO_ACIDS)
    --n_data <Number of sequences in input alignments.> (default: 20)
    --data_len <Length of sequences in the alignments or the number of genomes in typing.> (default: 200)
```
Example: python .\ml4phylo\scripts\train.py ....args......

## Alignments with Intervals
Instead of converting the sequences to typing data, it is also possible, for training purposes, to split these sequences into blocks without converting them to genome identifiers.
```txt
alignments_intervals
    --input <path to input directory containing the .fasta files>
    --output <path to output directory>
    --blocks <number of blocks of sequences required>
    --block_size <size of the blocks of sequences required>
    --interval <size of the interval between blocks of sequences>
```
Example: python .\ml4phylo\scripts\alignments_intervals.py ....args......


## Prediction of pair wise distances

### With sequences
```txt
predict
    --datadir <path to input directory containing corresponding data files: [.fasta for alignments or .txt for typing data]>
    --output <path to the output directory were the .tree tree files will be saved>
    --model <NN model state dictionary, path/to/model.pt>
    --data_type <type of input data. Possible values: [AMINO_ACIDS, NUCLEOTIDES, TYPING].> (default: AMINO_ACIDS)
```
Example: python .\ml4phylo\scripts\predict.py ....args......

## Prediction of true trees

### With sequences
```txt
predict_true_trees
    --indir <path to input directory containing corresponding data files: [.fasta for alignments or .txt for typing data]>
    --outdir <output directory were the .nwk tree files will be saved>
    --data_type <type of input data. Possible values: [AMINO_ACIDS, NUCLEOTIDES, TYPING].> (default: AMINO_ACIDS)
```
Example: python .\ml4phylo\scripts\predict_true_trees.py ....args......

## Evaluation of the obtained phylogenetic trees
```txt
evaluate
    --true <directory containing true trees in .nwk format>
    --predictions <directory containing predicted trees in .nwk format>
```
Example: python .\ml4phylo\scripts\evaluate.py ....args......

# Important Note

## Training folders
In \testdata\dataset\training there are some folders you can use to store any values gotten from any operations necessary to train the model:

- \testdata\dataset\training
    - trees &rarr; Store any .nwk files of generated trees;
    - alignments &rarr; Store any .fasta files of generated sequence alignments;
    - typing_data &rarr; Store any .txt files of typing data;
    - tensors
        - sequences &rarr; Store any tensor pairs of your sequence alignments;
        - typing_data &rarr; Store any tensor pairs of your typing data;
    - models 
        - sequences &rarr; Store any models gotten from training the model with sequences.
        - typing_data &rarr; Store any models gotten from training the model with typing data.

## Prediction folders
In \testdata\dataset\predictions there are some folders you can use to store any values gotten from any operations necessary to predict the phylogenetic trees:

- \testdata\dataset\predictions
    - alignments &rarr; Store any .fasta files of sequence alignments;
    - typing_data &rarr; Store any .txt files of typing data;
    - trees:
      - predicted:
        - sequences &rarr; Store any .nwk files of predicted trees from sequence alignments;
        - typing_data &rarr; Store any .nwk files of predicted trees from typing data;
      - true:
        - sequences &rarr; Store any .nwk files of true trees from sequence alignments;
        - typing_data &rarr; Store any .nwk files of true trees from typing data;

Feel free to use these existing folders, but you can always have your own!

**Whatever folder you might be using to store files, they are not emptied if you try to generate new ones! Remember to empty these folders to avoid any possible problems when you want to store any new files!**
