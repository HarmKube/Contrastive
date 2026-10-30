# Using singularity

Singularity is used to execute the code on HPCs and works with containers. This Container needs to be builded once in the beginning on a Linux machine with sudo privileges.

## Installation

To build the container you have to install singularity. Please refer to the singularity docs under [here](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html) for installation guide.

## Creating a container

This folder contains a container definition file [nip.def](nip.def). This file is used to build the container for singularity.
Create the container by executing the following command in the projects root folder:

```
sudo singularity build env/singularity/nip.sif env/singularity/nip.def
```

It will setup a Ubuntu 22.04 container with a clean conda environment and all requirements for YOLOv8 installed.

## Copy to cluster

After creating the container file it has to be moved to the cluster where you want work on. E.g. with `scp`.

## Use previously build container
A prebuilt container is not distributed with this repository. Build it yourself as described above.

To run code on a cluster, bind your data folder to `/mydata` and your configs to `/myconfigs`, e.g.:
`singularity exec --nv --bind /path/to/data:/mydata env/singularity/nip.sif python nip/nip.py -h`
