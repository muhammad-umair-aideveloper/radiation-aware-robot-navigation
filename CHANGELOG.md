# Changelog

Generated from conventional commit subjects.

## Features

- `f022e9c` feat(requirements): add initial dependencies for the project
- `c7b4a04` feat(plot): add S12 survey trade-off figures
- `c0cc491` feat(results): add S12 survey study outputs
- `1d178d6` feat(study): add resumable S12 survey runner
- `038f104` feat(online): integrate survey budget into replanning
- `60cbe27` feat(survey): add deterministic waypoint planning
- `8ff595e` feat(visualize): add online estimation animation
- `7e30ef1` feat(visualize): add wall shielding plots
- `294c276` feat(shield_study): implement shielding study with attenuation and path evaluation
- `5c00130` feat(shielding): add attenuation matrix and shielded field functions for radiation modeling
- `26ca8b1` feat(experiments): add S10 and S9 sections for physics-informed estimators and controls
- `9a0d9a7` feat(visualize): add physics-informed estimator plots
- `4affc44` feat(study): add physics-informed estimator study
- `f9a5e27` feat(online): support parametric estimator in the online simulator
- `38e151b` feat(estimators): add parametric inverse-square source estimator with tests
- `03315e9` feat(visualize): add sensor range study plots
- `6454095` feat(study): add sensor range and inflation study
- `97c1c57` feat(study): add sensor range and inflation study
- `28aba0e` feat(study): add simulator controls
- `de8e990` feat(online): add controls, noise-free mode and unknown-cell inflation
- `3042934` feat(measurement): add disc measurement patterns
- `ba482b3` feat(visualize): add online study plots
- `87473c3` feat(study): add online study summary script
- `0617d0f` feat(study): add online measure-estimate-replan study
- `d204b0d` feat(study): add online measure-estimate-replan study
- `2eadb6d` feat(study): add online study references
- `cdefe2a` feat(online): add online measure-estimate-replan episode
- `bc32b4b` feat(measurement): add measurement patterns
- `255f612` feat(visualize): add estimation study plots
- `c907e1b` feat(estimation): add S7 measure-estimate-plan study runner
- `a4ede94` feat(estimators): add kernel regression and idw field estimators
- `6a0d75a` feat(measurement): add poisson geiger model
- `0fa1420` feat: add plotting script for study s6 evaluation results and learning curves
- `c1bb0bd` feat(visualize): add s6 plots
- `db8ebce` feat(evaluate): add pair set selection and test guard
- `0c0a3b3` feat(train): add stack, curriculum and goal pool options
- `3166447` feat(env): add observation frame stacking
- `90445f1` feat(analysis): add sensor agent failure diagnosis
- `359db7d` feat(pairs): add validation, test and diagnostic pair sets
- `fa671a1` feat: add sensor plotting script for DQN learning curves and evaluation trajectories
- `31fdc03` feat(visualize): add sensor agent plots
- `ba2cde3` feat(evaluate): add sensor agent evaluation
- `78a3ba6` feat: add DQN shaped training log for seed 0
- `71e2dae` feat: add DQN sensor run results and logs
- `9d9ce59` feat: add DQN shaped reward sensor run log for lambda 2.5 seed 2
- `89ab83b` feat: implement DQN agent and sensor environment for radiation-aware robot navigation
- `a2e3504` feat(train): add sensor agent training script
- `952ceeb` feat(agents): add dqn agent in numpy
- `01f3146` feat(nn): add numpy mlp with adam
- `9cf652a` feat(env): add sensor-based navigation environment
- `b5bc87d` feat(pairs): add seeded goal pools and evaluation pairs
- `8fbaddf` feat(sensors): add lidar and geiger sensors
- `3eb7137` feat: add CI workflow for automated testing and unit tests for optimality gap
- `6fced25` feat(study): add per-layout q-learning summary
- `f04d896` feat: add experiment results, layout configuration data, and evaluation scripts
- `cfacf94` feat(analysis): add optimality gap versus dijkstra
- `8775b98` feat(compare): add q-learning vs sarsa comparison
- `348128f` feat(visualize): add greedy success convergence plot
- `7350417` feat(sweep): add sweep summary script
- `bd20329` feat: add evaluation results for layout Q seeds 102 through 108
- `f044473` feat: add simulation results for layout and sweep experiments
- `15a538f` feat: add Q-learning sweep results for lambda 2.5 and 5.0
- `ed1647a` feat(study): add per-layout q-learning
- `35f7ac2` feat(visualize): add layouts overview and robustness plots
- `2c86f20` feat(study): add map-error robustness study
- `d9c74ef` feat(study): add multi-layout astar study
- `0a473d1` feat(layouts): add seeded random layout generator
- `be81dd0` feat: add exact Dijkstra reference solver and Pareto frontier visualization script
- `c62159b` feat(vis): add Pareto trade-off curve generator for dose vs path length
- `d7c7924` feat(sweep): add sweep runner for A* and Q-learning across radiation weights
- `0f5b609` feat(sweep): implement radiation weight sweep script for A* and Q-learning
- `06916bd` feat: implement A* search planners and Tabular Q-Learning agent
- `fe5b12d` feat(dp): implement exact Dijkstra reference solver
- `7adc479` feat(bench): add edge-inference latency and memory profiler with mean and P99 latency over 1000 states
- `23f22be` feat(metrics): add Pareto metrics evaluator for success rate, path efficiency ratio, and radiation exposure
- `628d798` feat(config): extract reward weights into typed RewardConfig dataclass with formal docstring equations
- `a711545` feat(physics): add inverse-square radiation physics module with softened singularity and dose accumulation
- `ecc2e3d` feat: add benchmark script and metrics evaluator for Q-table agent
- `8ba3ef7` feat: add pretrained optimal Q-table weights
- `1e8b5dc` feat: add standalone evaluation script for pre-trained Q-table inference
- `4e09fe3` feat: save best performing Q-table model to disk
- `77400ea` feat(results): add real evaluation plots and navigation animation
- `47dfabd` feat(eval): add benchmark evaluation on 50 starts and canonical path
- `e534063` feat(vis): add trajectory comparison, RL convergence, and GIF generator
- `8592a1e` feat(planners): implement shortest A*, aware A*, and Q-learning agent
- `cd88cc7` feat(env): implement 50x50 radiation grid world environment

## Fixes

- `59c6c29` fix(study): fix table row breaks in online summary
- `0b02074` fix(tests): use ascii output in core test
- `03338be` fix(evaluate): support w_0 weight key

## Documentation

- `286f97f` docs(index): add reviewer reading paths
- `4b86c9e` docs(review): add project terminology glossary
- `9c4cb01` docs(report): add generated technical synthesis
- `4edee1d` docs(results): correct and allowlist audit findings
- `9c06bd7` docs(results): index result provenance
- `3096bfe` docs(study): document S12 survey results
- `254bd6e` docs(plan): declare the S12 survey study
- `92c1be5` docs: move detailed studies from the README into docs pages
- `48bc92a` docs: add wall shielding results to README
- `4b27900` docs(results): add wall shielding study results
- `0b385a7` docs: add physics-informed estimator results to README
- `077a0bf` docs(results): add physics-informed estimator study results
- `5eb0fa2` docs: pre-declare s10 physics-informed estimator study
- `213943f` docs: add sensor range and inflation results to README
- `0cb851c` docs(results): add sensor range and inflation study results
- `53a4c28` docs(results): add simulator control results
- `07e251c` docs: pre-declare s9 controls and sensor range study
- `0982fbc` docs: correct online reduction values
- `3feba8e` docs: add online replanning results to README
- `8ec9e71` docs(results): add online study figures
- `4819879` docs(results): add online study results
- `56c2fb9` docs: pre-declare online measure-estimate-replan study
- `1b9fb17` docs: add measure-estimate-plan results
- `c692d49` docs(results): add estimation study figures
- `96d8cec` docs(results): add measure-estimate-plan results
- `40e61f0` docs: add S7 experimental protocol for radiation-aware navigation study
- `a5029da` docs: pre-declare measure-estimate-plan study
- `9fa42f1` docs: add s6 sensor agent results to README
- `9ddb112` docs(results): add s6 figures
- `d229365` docs(results): add s6 final runs and test evaluation
- `6e54283` docs(results): add s6 variant runs and selection
- `38f11e7` docs(results): add sensor agent failure diagnosis
- `3e592ff` docs(results): add s6 pair sets
- `a5dec8f` docs: pre-declare s6 sensor agent improvement study
- `fa62e69` docs: add sensor agent results to README
- `e9022eb` docs(results): add sensor agent figures
- `338e440` docs(results): add sensor agent runs and evaluation
- `4c3518f` docs: declare sensor agent training budget from measured speed
- `93f9809` docs(results): add sensor evaluation pairs
- `7f1a77f` docs: pre-declare sensor agent study
- `ee2e110` docs: correct readme wording on sarsa and optimality
- `ddf2991` docs: update benchmark table labels and add explanatory notes in README
- `df59d71` docs: add sweep, sarsa and optimality results to README
- `345b55c` docs(results): add per-layout q-learning results
- `836949d` docs(results): add optimality gap results
- `2b05477` docs(results): add q-learning vs sarsa results
- `9fd1d2f` docs(results): add pareto plot with q-learning results
- `84d895b` docs(results): add lambda sweep summary and run files
- `2e17b59` docs: add layout and robustness results to README
- `43bb6ce` docs(results): add map-error robustness results
- `39a7be3` docs(results): add multi-layout astar results
- `aa0a243` docs: pre-declare layout and map-error studies
- `699e734` docs: add radiation weight sweep section and Pareto analysis to README
- `af7376b` docs(results): record A* radiation weight sweep measurements
- `5512ff5` docs: add radiation weight sweep plan to EXPERIMENTS.md
- `fa96fb8` docs: add MIT license
- `0c8ca66` docs: add development notes
- `431a42f` docs: add quick inference instructions for pre-trained weights in README
- `584cdf4` docs: add comprehensive README with measured results and equations
- `9da44c3` docs: add project specifications, agent rules and setup configuration

## Tests

- `ffe9af7` test(coverage): add core coverage and focused tests
- `9776aaf` test(docs): add numeric result audit tool
- `0b012c5` test(core): add golden regression references
- `d2027aa` test(online): check controls, noise-free mode and inflation
- `7b88996` test(measurement): check disc patterns
- `2a0f53b` test(online): check online episode invariants
- `27e0797` test(measurement): check measurement patterns
- `d4ad107` test(estimators): check estimator properties
- `79a5107` test(measurement): check poisson noise model
- `1fafafa` test(train): check curriculum schedule and defaults
- `9473723` test(env): check frame stacking
- `75cfed1` test(pairs): check pair set separation
- `30d8f86` test(agents): check replay buffer and schedules
- `e0adcc8` test(nn): check gradients against finite differences
- `fe0bf2f` test(env): check sensor environment rewards and shaping
- `ffe72fe` test(sensors): check lidar and geiger readings
- `3081f99` test(analysis): check optimality gap helper
- `5342c63` test(layouts): check generator properties
- `f7970ba` test(sweep): add tests for lambda sweep helpers and Pareto data integrity
- `f0aba06` test(dp): add verification tests for Dijkstra vs A* on 50 starts
- `038215f` test: add core unit tests and scientific invariant checks for radiation navigation
- `79575c7` test: add core unit tests, scientific invariants, and artifact checks
- `7e7a393` test: add core unit tests and scientific invariants for environment and planners
- `ab7143e` test: add mathematical invariant and artifact integrity checks
- `e5ca192` test: add core tests for radiation field, astar, and collision

## Continuous Integration

- `31ce9be` ci(gate): add one-command quality verification
- `361b937` ci(test): run every test on Python 3.10 to 3.12
- `9a6f06a` ci: run online study test files in the workflow
- `325322e` ci: run new test files in the workflow
- `87d3c0b` ci: run all test files in the workflow
- `db5c218` ci: add test workflow

## Build and Tooling

- `472f1ab` build(lint): enforce core Ruff checks
- `c1f7eef` build(deps): pin tested dependency versions

## Refactoring

- `3fd0fbd` refactor(core): clean central module health

## Chores

- `33e3b2e` chore: ignore tmp folder
- `f69197f` chore: add dqn shaped lambda 2.5 seed 2 sensor run log
- `5747456` chore: add citation metadata
- `df6003f` chore: add A* and Q-learning sweep results and metrics data
- `864245c` chore: add newline at end of LICENSE file
- `562a6af` chore: update configuration and environment setup scripts
- `6977a0a` chore: remove planning and tooling files from the repository

## Other

- `4394986` Add kernel estimation results for varying dwell times and sample sizes
- `ec562f8` results(study): add S10 parametric estimator study outputs
- `62cb993` Add new kernel estimation results for various configurations
- `8001bf2` Add new JSON result files for IDW and Kernel estimators across various patterns and dwell times
- `abb04d2` Add kernel estimation results for varying dwell times and sample sizes
- `c0553a0` style: normalize line endings on study modules and tests
- `543384a` style: add trailing newline to evaluate.py
- `b7bfb3e` Initial commit
