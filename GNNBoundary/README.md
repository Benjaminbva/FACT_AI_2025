<div align="center">

<h1>
    Reproducibility Study of GNNBoundary: Towards Explain-
    ing Graph Neural Networks through the Lens of Decision
    Boundaries
</h1>

<sup>1</sup>[University of Amsterdam](), &nbsp;

Under review as submission to TMLR

</div>

## 🔥 How to use

### Notebooks
* `results.ipynb` contains the results for the experiments
* `model_training.ipynb` contains the demo for GNN classifier training.

### Model Checkpoints
* You can find the GNN classifier checkpoints in the `ckpts` folder.
* See `model_training.ipynb` for how to load the model checkpoints.

### Datasets
* datasets will be downloaded automatically when running the code. If this fails, a link is available:
* [link](https://drive.google.com/file/d/1O3IRF9mhL2KCCU1eVlCEdssaf6y-pq2h/view?usp=sharing) for downloading the processed datasets.
* After downloading the datasets zip, please `unzip` it in the root folder.

### Results
* Other than in `results.ipynb`, the results can also be found in the `logs` folder.

### Environment
Codes in this repo have been tested on `python3.10` + `pytorch2.1` + `pyg2.5`.

To reproduce the exact python environment, please run:
```bash
conda create -n gnnboundary poetry jupyter
conda activate gnnboundary
poetry install
ipython kernel install --user --name=gnnboundary --display-name="GNNBoundary"
```

Note: In case poetry fails to install the dependencies, you can manually install them using `pip`:
```bash
pip install -r requirements.txt
````