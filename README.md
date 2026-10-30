# NIP

Project Neural Information Processing

Project Group: Harm Kube, Elena Bank, Clara Biedermann, Janis Peters, Gregor Voigts

## Overview
This project tests the impact of contrastive learning on the object detection task.

## Command Line Interface

This Project contains a Command line interface to start scripts. Its located under [nip/nip.py](nip/nip.py). See option `-h` for help.
To extend the command line interface actions and arguments can be added in the [cli_config](nip/cli/cli_config.py). You can find a detailed explanation
in the [README](./nip/README.md).

## Project Organization

```
├── Makefile            <- Convenience commands (currently not used)
├── pyproject.toml      <- Python project configuration (currently not used)
├── config/             <- YAML configuration files
├── data/               <- Data sources and scripts for YOLO training
├── env/                <- Environment setup (requirements, containers)
└── nip/                <- Source code for this project
    ├── benchmarks/      <- Benchmark scripts and performance evaluations
    ├── cli/             <- Command-line interface configs and actions
    ├── config/          <- Module-specific configuration schemas and defaults
    ├── dataset/         <- Dataset loading and management code
    ├── data_analysis/   <- Data analysis utilities and scripts
    ├── detco/           <- DETCO self-supervised framework integrations
    ├── embeddingxai/    <- Embedding-based explainability modules
    ├── interpretation/  <- Interpretation and visualization methods
    ├── kfolds/          <- K-fold cross-validation helper functions
    ├── lightlyssl/      <- Lightly-based self-supervised learning pipelines (SimCLR, Distillation, ...)
    ├── multistage/      <- Code for Generic GERALD dataset creation
    ├── sslcon/          <- Contrastive Self-Supervised Learning modules and utilities
    ├── visualization/   <- Plotting and visualization utilities
    ├── yt_gerald/       <- Pre-Training YOLO on the YouTube dataset
    └── nip.py           <- Main CLI script entry point
```

## License

The source code in this repository is licensed under the MIT License (see `LICENSE`).

This project uses the GERALD dataset (© Philipp Leibner and Fabian Hampel,
https://github.com/ifs-rwth-aachen/GERALD), licensed under CC BY-NC-SA 4.0.
The dataset is not included in this repository.

Leibner, P., Hampel, F., Schindler, C. (2023). GERALD: A novel dataset for the detection of
German mainline railway signals. Proc. IMechE Part F, 237(10), 1332–1342.
https://doi.org/10.1177/09544097231166472

---
