# nuPlan Dataset Documentation Design

## Goal

Document how users can obtain nuPlan v1.1 data and run the official
Diffusion-Planner preprocessing pipeline without downloading data or adding
SDK-dependent preprocessing code to this repository.

## Scope

Update `README.md` with a dataset and preprocessing section that:

- Identifies nuPlan v1.1 trainval databases and maps as the upstream inputs.
- Directs users to create an account and accept the nuPlan terms before
  downloading every required archive from the official portal.
- Shows the expected extracted directory layout.
- Provides the upstream nuPlan devkit and official Diffusion-Planner setup
  commands.
- Shows how to configure and run the official `data_process.sh` script with
  the database, map, and generated-training-data paths.
- States that this repository does not download the dataset, implement the
  preprocessing pipeline, or load the resulting data for training.

## Non-goals

- Downloading data in this repository or during tests.
- Vendoring the nuPlan devkit or upstream preprocessing code.
- Claiming that the generated upstream training data can be consumed directly
  by this standalone implementation.

## Sources

- nuPlan dataset setup: https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html
- Official Diffusion-Planner setup and preprocessing:
  https://github.com/ZhengYinan-AIR/Diffusion-Planner
- Official preprocessing script:
  https://raw.githubusercontent.com/ZhengYinan-AIR/Diffusion-Planner/main/data_process.sh

## Verification

Run the existing test suite after the README-only change. Review the commands
and paths against the linked upstream sources; no dataset download is part of
verification.
