## Installation
### Create and enter virtual environment (recommended)
Step 1: If you haven't installed anaconda on your machine, refer to this [doc](https://docs.anaconda.com/anaconda/install/), follow the instruction, 
and install the right version.

Step 2: Create the conda virtual environment. Refer to this [doc](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Example:
```
# bash
conda create -n [name of your environment] python=3.10
```    
Step 3: Enter the virtual environment by running:
```
# bash
conda activate [name of your environment]
```
Step 4: In the current version of geospaNN, to use the R-package [BRISC](https://github.com/ArkajyotiSaha/BRISC) 
for spatial parameter estimation (through rpy2), we need R installed in the environment. In order to install R, simply run:
```
# bash
conda install r-base
```
If you already have native R installed, it's also possible to manually initialize R for rpy2. 
See [here](https://rpy2.github.io/doc/latest/html/overview.html#install-installation) for more details.

### Manual dependency installation
(Currently) to install the development version of the package, a pre-installed PyTorch and PyG libraries are needed.
We provide options to install PyG libraries using conda and pip.

#### Option 1: Using Conda
For conda, installation in the following order is recommended. It may take around 10 minutes for conda to solve the environment for pytorch-sparse.
The following chunk has been tested in a python 3.10 environment.
```
#bash
conda install pytorch torchvision -c pytorch
conda install pyg -c pyg        
conda install pytorch-sparse -c pyg 
```

#### Option 2: Using pip
For pip, installation in the following order is recommended to avoid any compilation issue. It may take around 15 minutes to finish the installation.
The following chunk has been tested in a python 3.10 environment.
```
# bash
pip install numpy==1.26 --no-cache-dir
pip install torch==2.0.0 --no-cache-dir
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0.html --no-cache-dir
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0.html --no-cache-dir
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0.html --no-cache-dir
pip install torch_geometric --no-cache-dir
```
<!---
1. To install PyTorch, find and install the binary suitable for your machine [here](https://pytorch.org/).
2. Then to install the PyG library, find and install the proper binary [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
3. Make sure to also install the dependencies including *pyg_lib*, *torch_scatter*, *torch_sparse*, *torch_cluster*, and *torch_spline_conv*.
-->


### Main installation
Once PyTorch and PyG are successfully installed, use the following command in the terminal for the latest version (version 11/2024):
```
pip install https://github.com/WentaoZhan1998/geospaNN/archive/main.zip
```

To install the pypi version, use the following command in the terminal (version 1/2024):
```
pip install geospaNN
```

## An easy running sample:

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
batch_size = 50     # Batch size for training the neural networks.
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