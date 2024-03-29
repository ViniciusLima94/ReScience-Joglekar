## Introduction

This is a reference implementation of the following model:

Joglekar, Madhura R., et al. "Inter-areal balanced amplification enhances signal propagation in a large-scale circuit model of the primate cortex." Neuron 98.1 (2018): 222-234.

## Platform information

**Platform:** Ubuntu 22.04.2 LTS

**cmake:** 3.26.3

**gcc (GCC):** 11.3.0

**Python:** 3.11.3 

**Matplotlib:** 3.7.1

**NumPy:** 1.24.2

**SciPy:** 1.10.1

**NEST:** 3.4.0 (should be >= 3.0.0)

The machine used to run the codes related to this replication was equiped with a processor **Intel Xeon Gold 6320** 2.1GHz with 20C/20T and 125GB ram. The rate models were simulared using a single-core while the spiking-neuron model was simulated using 20C.

The version of the packages for scientific computing in python might be changed without any downside, however for **NEST** the version has to be 3.0 (or above).

## Package installation

To assure that all data and figures will be generated as presented in the article, we recommend following the instructions below and use the modules in the same versions described here, although the code also works in other versions.

### Python installation

The network simulation is implemented with Python (v.3.11.3).

To install Python 3, type in console:

```
$sudo apt-get update 
$sudo apt-get install python3.11
```

### Installation using Anaconda (**recommended**)

You can install the scientific Python modules with the Anaconda data science platform.

For Linux, Anaconda with Python 3.8 is only available for 64-bit systems. Download link:
https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh

To install open a terminal in the folder containing the downloaded file and type:

```
$chmod +x Anaconda3-2023.07-1-Linux-x86_64.sh

$./Anaconda3-2023.07-1-Linux-x86_64.sh
```

Once Anaconda is installed, the easiest way to install the packages required is via the enviroment file that we provided, using the command from the code directory:

```
conda env create -f rs_joglekar_all.yml
```

Once the enviroment is installed it can be activate with the command

```
conda activate rs_joglekar
```

After this step, the codes in this repository can be run withou dependency issues.

#### Python modules installation using Anaconda

Alternativelly, the user can manually install the requirements using conda, using the command bellow.

```
$conda install PYTHON_PACKAGE_NAME
```

### Installing pip

We use pip, a package management system, to install the Python modules described above.
To install pip in Ubuntu type in a terminal:

```
$sudo apt install python3-pip
```

Upgrade pip to the latest version:

```
$pip3 install --upgrade pip
```

#### Python modules installation using pip

Installation of packages using pip can be done with the following command:

```
$pip3 install --user PYTHON_PACKAGE_NAME
```


To install the required packages type in terminal:

```
$pip3 install --user matplotlib==3.7.1
$pip3 install --user numpy==1.24.2
$pip3 install --user scipy==1.10.1
```
or

```
$pip3 install --user matplotlib==3.7.1 numpy==1.24.2 scipy==1.10.1
```

### Installing NEST version 3 (or higher)

If the user opts for not using the available enviroment .yml file, NEST can be installed using conda by following the instructions in the NEST official website: https://nest-simulator.readthedocs.io/en/stable/installation/index.html


### Building NEST version 3.4

Another option is to install NEST from the source.

To do so first download the tar file with it by using the following link: https://github.com/nest/nest-simulator/archive/refs/tags/v3.4.tar.gz

Next, follow the steps bellow:

#### 1. Decompress the tar file
```
tar -xf nest-simulator-3.4.tar.gz 
```

This will create the folder **nest-simulator-3.4** with the installation files. 

#### 2. Create a build directory and change to it
```
mkdir nest-simulator-3.4-build
cd nest-simulator-3.4-build
```

#### 3. Make and make install NEST
```
cmake -DCMAKE_INSTALL_PREFIX:PATH=$HOME/opt/nest/3.4 ../nest-simulator-3.4
make
make install
```

And optionally:

```
make installcheck
```

The commands in step 3 will build NEST with the default build parameters you may set other configuration accordingly to your needs (see [here](https://nest-simulator.readthedocs.io/en/stable/installation/cmake_options.html)), but for the present code the default parameters are enough.

#### 4. Add NEST to the Path and Pythonpath 

##### Access the bashrc file with your preferred text editor
```
vim ~/.bashrc
```

##### Add the following two lines at the end of the file (you might need to access it as super user)
```
source $HOME/opt/nest/3.4/bin/nest_vars.sh
```

After reseting the terminal (or oppening a new one) you should be able to import nest from the Python console and see the following message:

```
In [1]: import nest

              -- N E S T --
  Copyright (C) 2004 The NEST Initiative

 Version: nest-3.4
 Built: Sep  8 2021 15:35:11

 This program is provided AS IS and comes with
 NO WARRANTY. See the file LICENSE for details.

 Problems or suggestions?
   Visit https://www.nest-simulator.org

 Type 'nest.help()' to find out more about NEST.
```

## Code repository

The code repository contain the following files:

* **rs_joglekar_all.yml ** *: Code with the package requirements used to run the codes bellow. Use conda to install the packages using this file (see Installation instructions above).

* **main.py:** This script is used to run the different "protocols" to generate the results from the original publication.
* **fig2.py:** This script contains the implementation of the two population rate model that generates figure 1 from the original publication.
* **fig3.py:** This script contains the implementation of the 29 population rate model (based on the empirical connectivity matrix) that generates figure 3 from the original publication.
* **fig4_5_6.py:** This script contains the implementation of the 29 population spiking-neuron model that generates figure 5 and figure 6 from the original publication.
* **setParams.py:** This script contains the parameter specifications for each model.
* **plot_figures.py:** This script contains the code to plot the outcomes generated by the simulations.
* **analysis.py:** This script contains functions used to analyse the outcome of the simulations.

The respository also contains a subdirectory **interareal** with the data (e.g., the empirical connectivity matrices) necessary to built the models. All figures are saved in the subdirectory **figures**.

## Running the scripts

To generate the results shown in the replication one ought to run the **main.py** script specifying the line arguments that corresponds to the "protocol" and the number of threads to use:


```
ipython main.py PROTOCOL N_THREADS
```

The outcome of each protocol is detailed below:

* **protocol 0:** Simulate the two population rate model and generate figure 1 from the original paper (figure 2 in the replication paper).
* **protocol 1:** Simulate the 29 population rate model and generate figure 3 from the original paper (figure 3 in the replication paper).
* **protocol 2:** Simulate the spiking-neuron model  for the conditions weak/strong GBA synchronous/assyinchronous and generate figures 5  and 6 from the original paper (figure 4 and 5 in the replication paper).
* **protocol 3:** Simulate the spiking-neuron model  for the condition synchronous/assyinchronous. It generate also figure 6 from the replication paper.
* **protocol 4:** Simulate the spiking-neuron model in the absence of external stimuli applied to the V1 population to assure that the network average frequency rate is in the range specified in the original paper. 

In the machine used to run the protocols we set **N_THREADS** as 1 for protocol 0 and 1, 20  for protocol 2 and 4, and 4 for protocol 3.

**WARNING:** About 10GB of RAM is used for protocols 2 and 4 (using 20 threads).
