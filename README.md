# expected-statistic-regularization
Implementation and experiments for "[Improving Low-Resource Cross-lingual Parsing with Expected Statistic Regularization]()" to appear in TACL 2023

## Installation

Experiments were all run on a g4dn.xlarge instance on AWS with the amazon deep learning AMI on ubuntu 18 using python 3.6.15.
(You'll need at least 16GB of GPU ram to run these experiments as is.)

The python requirements are in `requirements.txt`

## Library TOC

The code is based off of the [udify]() codebase and allennlp library.

The directory structure is as follows:
- `config`: holds allennlp experiment configurations. the configs define the model basics and use environment variables
 for hyperparams.
- `data`: holds datasets and model assets. Our experiments are with the UD v2.8 treebanks, which are available
[here](http://hdl.handle.net/11234/1-3687). The treebanks from that link must be placed in this folder, as in
`expected-statistic-regularization/data/ud-treebanks-v2.8/`. The base udify model weights (trained on the 13 datasets
from the paper) can be downloaded from this
[gdrive]() link, and should be placed in the `weights` folder as `udify-13-model_weights.th`.
- `ml`: holds source code for experiments. 
    - `ml.cmd` contains routines for creating datasets used in experiments
    - `ml.models` contains the main model classes used in the paper `expected_syntax_udify_model`, and the `ppt` baseline.
    - `ml.ppt` contains the source code from [Kurniwan's PPT method]() that we reuse for the baseline
    - `ml.training` contains code for training the model
    - `ml.udify` contains the source udify code from [Kondryak's UDify method](), which we extend.
    - `ml.utils` contains utilities for working with ud data and analyzing its statistics. the statistic functions are
      also implemented in here, both in regular python as `stat_funcs` and in pytorch as `tensor_stat_funcs`. The
      `test_tensor_stat_funcs` module ensures that these two implementations agree with eachother for correctness. 
- `scripts`: holds base scripts for generating datasets, computing stats, and launching experiments.

## Usage

### Setting up the data

To be able to run the experiments, you need the UD v2.8 [data] and the pretrained model [weights]. Given that these are
in the data directory, the subsampled datasets used in the experiments can be produced with:

```
bash scripts/subsample_datasets.sh
bash scripts/subsample_ud-all_datasets.sh
```

Once the datasets have been subsampled, we need to use them to compute bootstrap statistic targets and margins that
can be used for supervising transfer. These supervision hyperparameters are precomputed and stored into dataframes
that are read later by models when creating the supervision vectors. 

To compute the statistics, run

```
bash compare_stats_gold_udify-13.sh
bash compare_stats_subsamples_udify-13.sh
bash compute_stats_subsamples_ud-all.sh
```

### Running the experiments

Once the data has been populated and the stats computed, published experiments can all be run using the scripts:

Experiment 5.2 - Oracle transfer for different stats
```
bash scripts/run_ud-13_individual-losses.sh
```

Experiment 5.3 - Loss function ablations
```
bash scripts/ud-13_loss-func_ablation.sh
```

Experiment 6.1.1 - UD-13 Semisupervised Transfer Learning Curves
```
bash scripts/supervised/ud-13_learning_curve.sh
bash scripts/ppt/ud-13_semisup_learning_curve.sh
bash scripts/tail_dep_dir/ud-13_semisup_oracle_learning_curve.sh
bash scripts/tail_dep_dir/ud-13_semisup_real_learning_curve.sh
bash scripts/universal_arc/ud-13_semisup_real_learning_curve.sh
```

Experiment 6.1.2 - Scratch Semisupervised Transfer Learning Curves
```
bash scripts/supervised/scratch_learning_curve.sh
bash scripts/ppt/scratch_semisup_learning_curve.sh
bash scripts/tail_dep_dir/scratch_semisup_real_learning_curve.sh
bash scripts/universal_arc/scratch_semisup_real_learning_curve.sh
```

Experiment 6.2 - Extended transfer experiment with 50 labeled sentences for many treebanks
```
bash scripts/run_ud-13_all_S50.sh
bash scripts/run_ud-13_all_ud_semisup.sh
```


## Citing

If you make use of this code or information from the paper please cite:
<!-- ```
@misc{effland2022improving,
      title={Improving Low-Resource Cross-Lingual Parsing with Expected Statistic Regularization}, 
      author={Thomas Effland and Michael Collins},
      year={2022},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
``` -->
