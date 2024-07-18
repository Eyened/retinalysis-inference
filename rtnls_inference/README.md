## Retinalysis models inference

This repository implements inference ensembles for retinalysis model releases. These ensembles can be ran directly from Python code or be loaded into torchserve.

Ensemble releases are stored in torchscript format. Preprocessing that uses albumentations transforms and other external libraries cannot be included in torchscript. For this reason the test-time transforms are also included in this repository.