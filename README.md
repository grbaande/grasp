# GRASP

Graph-Based Anomaly Detection Through Self-Supervised Classification

## Install

1. Clone and enter the repo:

```bash
git clone <repo-url>
cd grasp
```

2. Create a virtual environment (Python 3.12.7 recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

3. Install core dependencies:

```bash
pip install -r requirements.txt
```

4. Install PyTorch and PyTorch Geometric for your platform (choose the right wheel for your CUDA/CPU setup, see https://pytorch.org/get-started/locally/):

```bash
pip install torch torchvision torchaudio  # pick version/build matching your CUDA
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv  
```

Example:
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

### Data
We used PostgreSQL dumps, of the DARPA datasets, from the related work: 
```
https://ubc-provenance.github.io/PIDSMaker/
```
Please follow the instructions to setup the PostgreSQL database.
This, and our graph construction, ensures comparable results.

Alternatively, you can use a specialized installation guide to set up only the PostgreSQL database of PIDSMaker.
See [db_setup README](db_setup/README.md) for details.

#### Optional: better runtime through index
Creating an index for the timestamp column improves performance when creating graphs. 
This must be done for each database, if desired.
```
CREATE INDEX time_index ON event_table (timestamp_rec);
```

### .env
Create a .env file. 

```bash
cp .env_example .env
```

Edit the .env_example with the credentials and socket information to reach the db. 
If everything runs on the same host and you followed [db_setup README](db_setup/README.md) the defaults will work.


## Configure

- Default experiment: grasp/experiments/experiment_cadets_e3.yaml. 
- For other experiment configurations, see [grasp/experiments](grasp/experiments).

## Run

```bash
python main.py --experiment-config grasp/experiments/all_experiments/cadets_e3_default/experiment_cadets_e3.yaml
```


## License

See [LICENSE](LICENSE) for details.
