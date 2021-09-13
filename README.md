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

The machine used to run the codes related to this replication was equiped with a processor **Intel Xeon Gold 6320** 2.1GHz with 20C/20T. The rate models were simulared using a single-core while the spiking-neuron model was simulated using 20C.

The version of the packages for scientific computing in python might be changed without any downside, however for **NEST** should be version 3.0 (or above).

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
