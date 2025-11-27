This repository is meant to store mainly the following `track_utils.py` file in which exist utility functions for analyzing and plotting data coming out of the Tracker software.

The file relies on the use of the following packages, and this code was developed using the following versions:
- `numpy = 2.2.4`
- `pandas = 2.2.3`
- `scipy = 1.15.2`
- `geopandas = 1.0.1`
- `shapely = 2.1.0`
- `plotly = 6.3.0`
- `kaleido = 1.1.0` (optional, for static image export)


# _Installation instructions_

### Directly from GitHub (*non-editable version*)
***
1. Open terminal and either activate an existing environment (`source activate myenv`) or<br>create a new one (`conda create -n tracker python=3.12`) and then activate it.
- Optional: If creating a new env, install all binary packages using conda _before pip installing the package_.<br>If the dependencies that this relies on are not present, they will automatically be downloaded with pip.
- Optional 2b: If you are planning to use jupyter notebooks, make sure to install that now too with `conda install -c conda-forge jupyter`.
2. Install package to env with `pip install git+https://github.com/joezaki/tracker.git`.
3. Verify installation with `python -c "from track_utils import track_utils; print('track_utils imported successfully.')"`.<br>If the package imported successfully, the print statement will be printed.

### From local cloned copy (*editable version*)
***
1. Open terminal and navigate to the folder where you would like track_utils to live (`cd /path/to/folder`).
2. Clone the repository into this folder with `git clone https://github.com/joezaki/track_utils.git`.
3. Navigate into `track_utils` with `cd track_utils`.
4. Activate an existing environment (`source activate myenv`) or create a new one (`conda create -n tracker`) and then activate it.
- Optional: If creating a new env, install all binary packages using conda _before pip installing the package_.
- Optional 2a: If creating a new env, you can use the provided `environment.yaml` file here, with `conda env create -f environment.yaml -n tracker`.
- Optional 2b: If you are planning to use jupyter notebooks, make sure to install that now too with `conda install -c conda-forge jupyter`.
5. Install package to env with `pip install -e .`. This ensures that any changes that are made to the repo will become immediately available.
6. Verify installation with `python -c "from track_utils import track_utils; print('track_utils imported successfully.')"`.<br>If the package imported successfully, the print statement will be printed.