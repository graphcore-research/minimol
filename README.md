# MiniMol
A small yet very powerful molecular featurizer that generalises well to biological tasks thanks to the effective pre-training on biological and quantum mechnical datasets.

The model has been introduced in the paper [𝙼𝚒𝚗𝚒𝙼𝚘𝚕: A Parameter-Efficient Foundation Model for Molecular Learning](https://arxiv.org/abs/2404.14986), published in the ICML workshop on *Accessible and Efficient Foundation Models for Biological Discovery* in 2024.

## Installation

#### Local-dev

Dependencies are listed in `env.yml`. Installation steps:

```
# Create and activate the environment
mamba env create -f env.yml
mamba activate minimol
```

#### pip-installable package

Installation is handled in `setup.py`.

## Performance

The model has been evaluated on 22 benchmarks from the ADMET group of Therapeutics Data Commons (TDC). These are the results when comparing to [MolE]() and TOP5 models from the TDC leaderboard (as of June 2024):

| TDC Dataset              |                                |                                        | TDC Leaderboard | MolE           |          | MiniMol (GINE)|          |
|--------------------------|--------------------------------|----------------------------------------|-----------------|----------------|----------|---------------|----------|
| **Name**                 | **Size**                       | **Metric**                             | **SOTA Result** | **Result**     | **Rank** | **Result**    | **Rank** |
| **Absorption**           |                                |                                        |                 |                |          |               |          |
| Caco2 Wang               | 906                            | MAE                                    | 0.276 ± 0.005   | 0.310 ± 0.010  | 6        | 0.324 ± 0.012 | 7        |
| Bioavailability Ma       | 640                            | AUROC                                  | 0.748 ± 0.033   | 0.654 ± 0.028  | 7        | 0.699 ± 0.008 | 6        |
| Lipophilicity AZ         | 4,200                          | MAE                                    | 0.467 ± 0.006   | 0.469 ± 0.009  | 3        | 0.455 ± 0.001 | 1        |
| Solubility AqSolDB       | 9,982                          | MAE                                    | 0.761 ± 0.025   | 0.792 ± 0.005  | 5        | 0.750 ± 0.012 | 1        |
| HIA Hou                  | 578                            | AUROC                                  | 0.989 ± 0.001   | 0.963 ± 0.019  | 7        | 0.994 ± 0.003 | 1        |
| Pgp Broccatelli          | 1,212                          | AUROC                                  | 0.938 ± 0.002   | 0.915 ± 0.005  | 7        | 0.994 ± 0.002 | 1        |
| **Distribution**         |                                |                                        |                 |                |          |               |          |
| BBB Martins              | 1,975                          | AUROC                                  | 0.916 ± 0.001   | 0.903 ± 0.005  | 7        | 0.923 ± 0.002 | 1        |
| PPBR AZ                  | 1,797                          | MAE                                    | 7.526 ± 0.106   | 8.073 ± 0.335  | 6        | 7.807 ± 0.188 | 4        |
| VDss Lombardo            | 1,130                          | Spearman                               | 0.713 ± 0.007   | 0.654 ± 0.031  | 3        | 0.570 ± 0.015 | 7        |
| **Metabolism**           |                                |                                        |                 |                |          |               |          |
| CYP2C9 Veith             | 12,092                         | AUPRC                                  | 0.859 ± 0.001   | 0.801 ± 0.003  | 5        | 0.819 ± 0.001 | 4        |
| CYP2D6 Veith             | 13,130                         | AUPRC                                  | 0.790 ± 0.001   | 0.682 ± 0.008  | 6        | 0.718 ± 0.003 | 5        |
| CYP3A4 Veith             | 12,328                         | AUPRC                                  | 0.916 ± 0.000   | 0.867 ± 0.003  | 7        | 0.878 ± 0.001 | 5        |
| CYP2C9 Substrate         | 666                            | AUPRC                                  | 0.441 ± 0.033   | 0.446 ± 0.062  | 2        | 0.481 ± 0.013 | 1        |
| CYP2D6 Substrate         | 664                            | AUPRC                                  | 0.736 ± 0.024   | 0.699 ± 0.018  | 7        | 0.726 ± 0.006 | 2        |
| CYP3A4 Substrate         | 667                            | AUROC                                  | 0.662 ± 0.031   | 0.670 ± 0.018  | 1        | 0.644 ± 0.006 | 6        |
| **Excretion**            |                                |                                        |                 |                |          |               |          |
| Half Life Obach          | 667                            | Spearman                               | 0.562 ± 0.008   | 0.549 ± 0.024  | 4        | 0.493 ± 0.002 | 7        |
| Clearance Hepatocyte     | 1,102                          | Spearman                               | 0.498 ± 0.009   | 0.381 ± 0.038  | 7        | 0.448 ± 0.006 | 4        |
| Clearance Microsome      | 1,020                          | Spearman                               | 0.630 ± 0.010   | 0.607 ± 0.027  | 6        | 0.652 ± 0.007 | 1        |
| **Toxicity**             |                                |                                        |                 |                |          |               |          |
| LD50 Zhu                 | 7,385                          | MAE                                    | 0.552 ± 0.009   | 0.823 ± 0.019  | 7        | 0.588 ± 0.010 | 3        |
| hERG                     | 648                            | AUROC                                  | 0.880 ± 0.002   | 0.813 ± 0.009  | 7        | 0.849 ± 0.007 | 6        |
| Ames                     | 7,255                          | AUROC                                  | 0.871 ± 0.002   | 0.883 ± 0.005  | 1        | 0.856 ± 0.001 | 5        |
| DILI                     | 475                            | AUROC                                  | 0.925 ± 0.005   | 0.577 ± 0.021  | 7        | 0.944 ± 0.007 | 1        |
|                          |                                |                                        |                 | **Mean Rank:** | 5.2      |               | 3.4      |