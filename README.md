## Introduction

This is a reference implementation of the following model:

Joglekar, Madhura R., et al. "Inter-areal balanced amplification enhances signal propagation in a large-scale circuit model of the primate cortex." Neuron 98.1 (2018): 222-234.

## Platform information

**Platform:** Ubuntu 20.04.2 LTS

**cmake:** 3.16.3

**gcc (GCC):** 9.3.0

**Python:** 3.8.8 

**Matplotlib:** 3.3.4

**NumPy:** 1.18.5

**SciPy:** 1.6.2

**NEST:** 3.0.0

The machine used to run the codes related to this replication was equiped with a processor **Intel Xeon Gold 6320** 2.1GHz with 20C/20T and 125GB ram. The rate models were simulared using a single-core while the spiking-neuron model was simulated using 20C.

The version of the packages for scientific computing in python might be changed without any downside, however for **NEST** should be version 3.0 (or above).

## Package installation

To assure that all data and figures will be generated as presented in the article, we recommend following the instructions below and use the modules in the same versions described here, although the code also works in other versions.

### Python installation

The network simulation is implemented with Python (v.3.8.8).

To install Python 3, type in console:

```
$sudo apt-get update 
$sudo apt-get install python3.8
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

Installation of packages using pip can be done with the following command:

```
$pip3 install --user PYTHON_PACKAGE_NAME
```

#### Python modules installation using pip (recommended)

To install the required packages type in terminal:

```
$pip3 install --user matplotlib==3.3.4
$pip3 install --user numpy==1.18.5
$pip3 install --user scipy==1.6.2
```
or

```
$pip3 install --user matplotlib==3.3.4 numpy==1.18.5 scipy==1.6.2
```

All software packages are also available in the Anaconda distribution.

### Alternative installation (using Anaconda)

Alternatively, you can install the scientific Python modules with the Anaconda data science platform.

For Linux, Anaconda with Python 3.8 is only available for 64-bit systems. Download link:
https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

To install open a terminal in the folder containing the downloaded file and type:

```
$chmod +x Anaconda3-2020.11-Linux-x86_64.sh

$./Anaconda3-2020.11-Linux-x86_64.sh
```

#### Python modules installation using Anaconda

```
$conda install PYTHON_PACKAGE_NAME
```

### Building NEST version 3

To date the current version of NEST that can be installed via Ananconda is the version 2.20.x, therefore it is necessary to build NEST 3 from the source. To do so first download the tar file with it by using the following link: https://github.com/nest/nest-simulator/archive/refs/tags/v3.0.tar.gz

Next, follow the steps bellow:

#### 1. Decompress the tar file
```
tar -xf nest-simulator-3.0.tar.gz 
```

This will create the folder **nest-simulator-3.0** with the installation files. 

#### 2. Create a build directory and change to it
```
mkdir nest-simulator-3.0-build
cd nest-simulator-3.0-build
```

#### 3. Make and make install NEST
```
cmake -DCMAKE_INSTALL_PREFIX:PATH=$HOME/opt/nest/3.0 ../nest-simulator-3.0
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
export PATH="$HOME/opt/nest/3.0:$PATH"
export PYTHONPATH="$HOME/opt/nest/3.0/lib/python3.8/site-packages${PYTHONPATH:+:$PYTHONPATH}
```

After reseting the terminal (or oppening a new one) you should be able to import nest from the Python console and see the following message:

```
In [1]: import nest

              -- N E S T --
  Copyright (C) 2004 The NEST Initiative

 Version: nest-3.0
 Built: Sep  8 2021 15:35:11

 This program is provided AS IS and comes with
 NO WARRANTY. See the file LICENSE for details.

 Problems or suggestions?
   Visit https://www.nest-simulator.org

 Type 'nest.help()' to find out more about NEST.
```

## Code repository

The code subdirectory contain the following files:

* **main.py:** This script is used to run the different "protocols"

### [ReScience C](https://rescience.github.io/) article template

This repository contains the Latex (optional) template for writing a ReScience
C article and the (mandatory) YAML metadata file. For the actual article,
you're free to use any software you like as long as you enforce the proposed
PDF style. A tool is available for the latex template that produces latex
definitions from the metadata file. If you use another software, make sure that
metadata and PDF are always synced.

You can also use overleaf with the [provided template](https://www.overleaf.com/read/kfrwdmygjyqw) but in this case, you'll have to enter `metadata.tex` manually.

#### Usage

For a submission, fill in information in
[metadata.yaml](./metadata.yaml), modify [content.tex](content.tex)
and type:

```bash
$ make 
```

This will produce an `article.pdf` using xelatex and provided font. Note that you must have Python 3 and [PyYAML](https://pyyaml.org/) installed on your computer, in addition to `make`.


After acceptance, you'll need to complete [metadata.yaml](./metadata.yaml) with information provided by the editor and type again:

```bash
$ make
```

(C) 2015-2020, Nicolas Rougier + co-authors GPL-3+, Apache v2+, SIL Open Font License

This set of template files is free-licensed. The files contained in
the sub-directories roboto/; source-code-pro/; source-sans-pro/;
source-serif-pro/; have their free licences indicated with a
"*License.txt" file. All other files, including this one, are licensed
under the GPL version 3 or later, at your choosing, by Nicolas Rougier
and co-authors, 2015-2020. See the file COPYING for details of the
GPL-3 licence.
