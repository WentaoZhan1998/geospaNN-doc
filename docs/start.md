## Installation
We provide two straightforward installation approaches: via conda and via pip.
Depending on your system setup, it is possible to combine both methods, but be aware that mixing Conda and Pip installations can sometimes lead to dependency conflicts. Proceed with caution and ensure that package versions remain compatible.

### Approach 1: all-in-one through conda (recommended)
1. If you haven't installed anaconda on your machine, refer to this [doc](https://docs.anaconda.com/anaconda/install/) follow the instruction 
and install the right version.
2. Create the conda virtual environment from the [environment.yml](https://github.com/WentaoZhan1998/geospaNN/blob/main/environment.yml) file in this repository. You can specify your environment name by editing "env_name" on the first line of the yml file.
Example:
```
# bash
conda env create -f environment.yml
```
Note:
For Apple Silicon users on a Mac with an Apple M-series (ARM64) chip, you can improve performance by explicitly creating the environment for the ARM architecture instead:
```
# bash
conda env create -f environment.yml --subdir osx-arm64
```
For more details on creating a conda environment, refer to this [doc](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
3. Enter the virtual environment by running:
```
# bash
conda activate [name of your environment]
```

### Approach 2: using pip
(Currently) to avoid running issue, matched PyTorch and PyG libraries are needed, requiring us to install torch and pyg library manually.

1. For pip, installation in the following order is recommended to avoid any compilation issue.
The following chunk has been tested in a python 3.10 environment.
```
# bash
pip install numpy torch==2.7 torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
```
2. Once PyTorch and PyG are successfully installed, use the following command in the terminal for the latest version (version 04/2025):
```
# bash
pip install https://github.com/WentaoZhan1998/geospaNN/archive/main.zip
```
To install the pypi version, use the following command in the terminal (version 04/2025):
```
# bash
pip install geospaNN
```
3. (Skip if you already have R ready to use). The current version of **geospaNN** uses R-package [BRISC](https://github.com/ArkajyotiSaha/BRISC) 
for spatial parameter estimation through rpy2, thus requiring R installed in the environment. To install an R version compatible with your Python and system architecture, Mac users can check their architecture with:
```
# bash
python -c "import platform; print(platform.machine())"
```
Then download the appropriate R installer from [CRAN for macOS](https://cran.r-project.org/bin/macosx/). Windows users can download R from [CRAN for Windows](https://cran.r-project.org/bin/windows/base/). 
4. If rpy2 cannot find your R installation, you may need to set the R home directory manually. First, find R’s home path by running in terminal:
```
# bash
R R_HOME
```
Then, set this directory in your Python environment before importing **geospaNN**:
```
# bash
python -c "import os; os.environ["R_HOME"] = [R home path]"
```
Make sure to use the path to the correct R.

## An easy running sample (functionality verification):
This is a simple running sample to check the functionality of the package.
First, run python in the terminal:
```
python
```
import the modules and set up the parameters
1. Define the Friedman's function, and specify the dimension of input covariates.
2. Set the parameters for the spatial process.
3. Set the hyperparameters of the data.
```
import torch
import geospaNN
import numpy as np

# 1. 
def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

p = 5; funXY = f5

# 2.
sigma = 1
phi = 3/np.sqrt(2)
tau = 0.01
theta = torch.tensor([sigma, phi, tau])

# 3.
n = 1000            # Size of the simulated sample.
nn = 20             # Neighbor size used for NNGP.
```

Next, simulate and split the data.

1. Simulate the spatially correlated data with spatial coordinates randomly sampled on a [0, 10]^2 squared domain.
2. Order the spatial locations by [max-min ordering](https://projecteuclid.org/journals/statistical-science/volume-36/issue-1/A-General-Framework-for-Vecchia-Approximations-of-Gaussian-Processes/10.1214/19-STS755.full).
3. Build the nearest neighbor graph, as a torch_geometric.data.Data object.
4. Split data into training, validation, testing sets.

```
# 1.
torch.manual_seed(2024)
X, Y, coord, cov, corerr = geospaNN.Simulation(n, p, nn, funXY, theta, range=[0, 10])

# 2.
X, Y, coord, _ = geospaNN.spatial_order(X, Y, coord, method = 'max-min')

# 3.
data = geospaNN.make_graph(X, Y, coord, nn)

# 4.
data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, neighbor_size=20,
                                                   test_proportion=0.2)
```    

Compose the mlp structure and train easily.
1. Define the mlp structure (torch.nn) to use.
2. Define the NN-GLS corresponding model.
3. Define the NN-GLS training class with learning rate and tolerance.
4. Train the model.

```
# 1.             
mlp = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

# 2.
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp, theta=torch.tensor([1.5, 5, 0.1]))

# 3.
nngls_model = geospaNN.nngls_train(model, lr =  0.01, min_delta = 0.001)

# 4.
training_log = nngls_model.train(data_train, data_val, data_test,
                                 Update_init = 10, Update_step = 10)
```

Estimation from the model. The variable is a torch.Tensor object of the same dimension
```
train_estimate = model.estimate(data_train.x)
```

Kriging prediction from the model. The first variable is supposed to be the data used for training, and the second 
variable a torch_geometric.data.Data object which can be composed by geospaNN.make_graph()'.
```
test_predict = model.predict(data_train, data_test)
```