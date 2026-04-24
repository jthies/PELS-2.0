#!/bin/bash

module load python/3.12-conda cuda

ENV=${HOME}/env-PELS

if [ ! -d $ENV ]; then
    python3 -m venv ${ENV}
    source ${ENV}/bin/activate
    python -m pip install --upgrade pip
    pip install numpy scipy numba
    pip install numba_cuda[cu13]
    pip install cupy
    pip install pytest parameterized
else
    source ${ENV}/bin/activate
fi;

