MightyCodes
===========

MightyCodes is a software package for constructing optimal short codes for ultra-low-bandwidth communication systems.

Installation
============

```
git clone https://github.com/broadinstitute/MightyCodes.git
pip install -e MightyCodes/
```

Modules
=======

MightCodes includes the following modules:

sa-bac
------

`sa-bac` constructs optimal codebooks for BAC channels using a GPU-accelerated parallel simulated annealing algorithm. The usage is as follows:

```
mighty-codes sa-bac -i <PATH_TO_PARAMS_YAML_FILE>
```

Template input `.yaml` files are provided in `MightyCodes/yaml`.

