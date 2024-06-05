<img src="./resources/ML4Phylo_logo.png" alt="ML4Phylo" width="500"/>

# ML4Phylo
Machine Learning techniques applied to Phylogenetic Analysis

# Dependencies
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
    --alidir <input directory with the corresponding .fasta alignments>
    --output <output directory>
    --nucleotides <boolean that indicates if it's used nucleotides instead of aminoacids>
```
Example: python .\ml4phylo\scripts\make_tensors.py ....args......

### With typing data
```txt
make_tensors_typing
    --treedir <input directory with the .nwk tree files>
    --typingdir <input directory with the corresponding .txt typing data files>
    --output <output directory>
```
Example: python .\ml4phylo\scripts\make_tensors_typing.py ....args......

## Train the neural model
```txt
train
    --input <input directory containing the tensor pairs on which the model will be trained>
    --validation <input directory containing the tensor pairs on which the model will be evaluated.>
                    (If left empty 10% of the training set will be used as validation data.)
    --config <configuration json file for the hyperparameters.>
    --output <output directory where the model parameters and the metrics will be saved.>
    --input_type <type of input data. Possible values: [nucleotides, aminoacids, typing].> (default: aminoacids)
    --n_seqs <Number of sequences in input alignments.> (default: 20)
    --seq_len <Length of sequences in input alignments.> (default: 200)
```
Example: python .\ml4phylo\scripts\train.py ....args......

<!-- # Instructions for prediction
In the current state of the project, the scripts responsible for the prediction and evaluation of phylogenetic trees do not work for typing data.


## Prediction of pair wise distances
```txt
predict
    --alidir <input directory containing the .fasta alignments>
    --output <path to the output directory were the .tree tree files will be saved (default: alidir)>
    --model <NN model state dictionary. Possible values are: [seqgen, evosimz, <path/to/model.pt>]> (default: seqgen)
```
Example: python .\ml4phylo\scripts\predict.py ....args......

## Evaluation of the obtained phylogenetic trees
```txt
evaluate
    --true <directory containing true trees in .nwk format>
    --predictions <directory containing predicted trees in .nwk format>
```
Example: python .\ml4phylo\scripts\evaluate.py ....args...... -->

# Important Note

## Training folders
In \testdata\dataset\training there are some folders you can use to store any values gotten from any operations necessary to train the model:

- \testdata\dataset\training
    - trees &rarr; store the .nwk files of the generated trees;
    - alignments &rarr; store the .fasta files of the generated sequence alignments;
    - typing_data &rarr; store the .txt files of the typing data files;
    - tensors_sequences &rarr; store the tensor pairs of your sequence alignments;
    - tensors_typing &rarr; store the tensor pairs of your typing data;
    - output &rarr; store the output gotten from training the model.

Feel free to use these existing folders, but you can always have your own!

## Current non-available scripts

### "predict.py" and "predict_typing.py"
The prediction scripts "predict.py" and "predict_typing.py" for both sequence alignments and typing data, respectivelly, are momentarily unavailable as these are being worked on at the moment. 

### "evaluate.py"
The evaluation script "evaluate.py" is also momentarily unavailable as it requires results from the previously mentioned scripts.


