# Data Augmentation in Graph Neural Networks: The Role of Generated Synthetic Graphs

## Citation
If you use this code, please cite our work:

```bibtex
@misc{bas2024dataaugmentationgraphneural,
  title={Data Augmentation in Graph Neural Networks: The Role of Generated Synthetic Graphs}, 
  author={Sumeyye Bas and Kiymet Kaya and Resul Tugay and Sule Gunduz Oguducu},
  year={2024},
  eprint={2407.14765},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.14765}
}
```

## Overview

This repository is organized into two main parts:
1. **Graph Generation**
2. **Graph Classification**

Some models utilize NetworkX graphs, while others are built on PyTorch Geometric graphs. Conversions between these formats may be necessary depending on the use case.

The `graph_analysis.ipynb` notebook can be used to obtain statistical information on any graph dataset.

----------------------------------------------------

## Graph Generation

We employ two different generation models based on graph size:
- For graphs with fewer than 100 nodes, we use **GraphRNN**.
- For larger graphs, we use **GRAN**.

You can find the notebooks we used for GraphRNN and GRAN.

### GRAN

Original repository: [GRAN GitHub Repository](https://github.com/lrjconan/GRAN/tree/master)

We modified several files to customize data loading and splitting for our experiments:
- **Data Splits**: Edit `utils/data_helper.py` to adjust dataset splits.
- **Configuration**: Update `config/collab_sample.yaml` to set parameters for your experiments.

### GraphRNN

Original repository: [GraphRNN GitHub Repository](https://github.com/JiaxuanYou/graph-generation)

We altered the files to read data from `graphs.pt` and to handle data splits as required for our experiments.
Changed files are : 

----------------------------------------------------

## Graph Classification

Graph classification algorithms are implemented to evaluate the generated synthetic graphs. The `datasets` folder contains subfolders for different experiments. Currently, it includes experiments on the COLLAB dataset, with variations b (raw-data), c (w/ Real), exp_1 (w/ Gen.) and e (test data). 

Experiments `exp1` and `exp2` include augmented data. You can modify the `config.txt` file to adjust hyperparameters and set up different experiments. Test data will always be the folder `e`.

### Datasets

The `datasets` folder includes multiple subfolders, each corresponding to different experimental setups. Currently, it contains the COLLAB dataset's experimental files. The different configurations (b, c, e) are represented as explained in the paper.

### Configuration

To adjust the hyperparameters or experimental settings, modify the `config.txt` file according to your needs.
