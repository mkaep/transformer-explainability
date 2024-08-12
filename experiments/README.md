# Author info:
The configuration files in the sub-directories created most of the files and directories in:
- [explaining](../explaining) - contains the XAI and conceptual-metric results
- [training](../training) - contains the trained transformers and attention-metric results

# Experiments

Please read!

This file contains information on how to reproduce everything:

* Preprocessing of Event Log files.
* Training a Transformer on the preprocessed event log files.
* Explaining a certain trace based on a trained model.
* The different transformer versions along with their Git hash/tag.

All the resulting data is stored within the repository as well.
You should be able to run the configuration files yourself and see the same results,
from preprocessing over training to explaining.

The file contains a lot of backwards-links, also to the [Data Readme](../PLG2/generated/README.md).
The text displayed for the links may not be consistent, therefore please navigate via the links.

Main contents:

* [Preprocessing](#preprocessing)
    * Experiments start with "[P]" as in **P**reprocessing.
* [Training](#training)
    * Experiments start with "[T]" as in **T**raining.
* [Explaining](#explaining)
    * Experiments start with "[X]" as in e**X**plaining.
* [Transformer Versions](#transformer-versions)
    * Where to find the different transformer version in Git.
* [Noteworthy](#noteworthy-explanations)
    * Noteworthy explanations (i.e. possibly interesting)

### Data

See [PLG2/generated/README.md](./../PLG2/generated/README.md)

## Preprocessing

#### [P] exp_small

* [preprocessing/exp_small.json](preprocessing/exp_small.json)
    * Data: [Simple Sequence](./../PLG2/generated/README.md#simple-sequence)
    * Splitting by: time

#### [P] exp_001

* [preprocessing/exp_001.json](preprocessing/exp_001.json)
    * Data: [Simple Sequence](./../PLG2/generated/README.md#simple-sequence) and
      [Longer Sequence](./../PLG2/generated/README.md#longer-sequence)
    * Splitting by: time

#### [P] exp_002

* [preprocessing/exp_002.json](preprocessing/exp_002.json)
    * Data: [Simple XOR and Sequence Process](./../PLG2/generated/README.md#simple-xor-and-sequence-process)
    * Splitting by: time

#### [P] exp_003

* [preprocessing/exp_003.json](preprocessing/exp_003.json)
    * Data: [Simple AND and Sequence Process](./../PLG2/generated/README.md#simple-and-and-sequence-process)
    * Splitting by: time

#### [P] exp_006

* [preprocessing/exp_006.json](preprocessing/exp_006.json)
    * Data: [Long Running Dependency](./../PLG2/generated/README.md#long-running-dependency)
    * Splitting by: time

#### [P] exp_007

* [preprocessing/exp_007.json](preprocessing/exp_007.json)
    * Data: [XOR then SEQ](./../PLG2/generated/README.md#xor-then-seq)
    * Splitting by: time

#### [P] exp_008

* [preprocessing/exp_008.json](preprocessing/exp_008.json)
    * Data: [AND then SEQ](./../PLG2/generated/README.md#and-then-seq)
    * Splitting by: time

#### [P] exp_009

* [preprocessing/exp_009.json](preprocessing/exp_009.json)
    * Data: [Longer AND Branches](./../PLG2/generated/README.md#longer-and-branches) and
    * Splitting by: time

#### [P] exp_010

* [preprocessing/exp_010.json](preprocessing/exp_010.json)
    * Data: [Even Longer AND Branches](./../PLG2/generated/README.md#even-longer-and-branches)
    * Splitting by: time

#### [P] exp_011

* [preprocessing/exp_011.json](preprocessing/exp_011.json)
    * Data: [Long Running Dependency 2](./../PLG2/generated/README.md#long-running-dependency-2)
    * Splitting by: time

#### [P] exp_012

* [preprocessing/exp_012.json](preprocessing/exp_012.json)
    * Data: [Long Running Dependency 3](./../PLG2/generated/README.md#long-running-dependency-3)
    * Splitting by: time

#### [P] exp_013

* [preprocessing/exp_013.json](preprocessing/exp_013.json)
    * Data: [Long Running Dependency 3](./../PLG2/generated/README.md#long-running-dependency-3)
    * Splitting by: random

#### [P] exp_014

* [preprocessing/exp_014.json](preprocessing/exp_014.json)
    * Data: [Longer Sequence](./../PLG2/generated/README.md#longer-sequence)
    * Splitting by: random

#### [P] exp_015

* [preprocessing/exp_015.json](preprocessing/exp_015.json)
    * Data: [Long Running Dependency 2](./../PLG2/generated/README.md#long-running-dependency-2)
    * Splitting by: random

#### [P] exp_016

* [preprocessing/exp_016.json](preprocessing/exp_016.json)
    * Data: [AND then SEQ](./../PLG2/generated/README.md#and-then-seq)
    * Splitting by: random

#### [P] exp_017

* [preprocessing/exp_017.json](preprocessing/exp_017.json)
    * Data: [Even Longer AND Branches](./../PLG2/generated/README.md#even-longer-and-branches)
    * Splitting by: random

#### [P] exp_018

* [preprocessing/exp_018.json](preprocessing/exp_018.json)
    * Data: [Longer Sequence](./../PLG2/generated/README.md#longer-sequence)
    * Splitting by: random

#### [P] exp_019

* [preprocessing/exp_019.json](preprocessing/exp_019.json)
    * Data: [Complex Start then Sequence](./../PLG2/generated/README.md#complex-start-then-sequence)
    * Splitting by: random

#### [P] exp_020

* [preprocessing/exp_020.json](preprocessing/exp_020.json)
    * Data: [Even Longer AND Branches](./../PLG2/generated/README.md#even-longer-and-branches)
    * Splitting by: random

#### [P] exp_021

* [preprocessing/exp_021.json](preprocessing/exp_021.json)
    * Data: [Complex Model 001](./../PLG2/generated/README.md#complex-model-001)
    * Splitting by: random
    * Note: Further preprocessing to make case-IDs unique

#### [P] exp_022

* [preprocessing/exp_022.json](preprocessing/exp_022.json)
    * Data: [Complex Model 001](./../PLG2/generated/README.md#complex-model-001)
    * Splitting by: random
    * Note: Used own generator:
        * `--bpmn_path PLG2\generated\14_complex_models\complex_model_001\complex_model_001.bpmn
          --output_path PLG2\generated\14_complex_models\complex_model_001\log3_complex_model_001.xes
          --trace_count 5000`

#### [P] exp_023

* [preprocessing/exp_023.json](preprocessing/exp_023.json)
    * Data: [Looped And](./../PLG2/generated/README.md#looped-and)
    * Splitting by: random
    * Note: Used own generator:
        * `--bpmn_path PLG2\generated\15_looped_AND\looped_AND.bpmn
          -output_path PLG2\generated\15_looped_AND\log1_looped_AND_001.xes
          --trace_count 1000
          --balanced`

`
--bpmn_path
PLG2\generated\16_loop_tests\xor_and\xor_and.bpmn
--output_path
PLG2\generated\16_loop_tests\xor_and\log1_xor_and_001.xes
--trace_count
100
--balanced
--hard_restrict`

`--bpmn_path
PLG2\generated\16_loop_tests\xor_xor\xor_xor.bpmn
--output_path
PLG2\generated\16_loop_tests\xor_xor\log1_xor_xor_001.xes
--trace_count
100
--balanced
--hard_restrict`

---

## Training

#### [T] exp_001

* [exp_001](training/exp_001.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Simple Sequence - Exp001](#p-exp001)

#### [T] exp_002

* [exp_002](training/exp_002.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Longer Sequence - Exp001](#p-exp001)

#### [T] exp_003

* [exp_003](training/exp_003.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Simple XOR and Sequence Process - Exp002](#p-exp002)

#### [T] exp_004

* [exp_004](training/exp_004.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Simple AND and Sequence Process - Exp003](#p-exp003)

#### [T] exp_006

* [exp_006](training/exp_006.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Long running dependency - Exp006](#p-exp006)

#### [T] exp_007

* [exp_007](training/exp_007.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [XOR then sequence - Exp007](#p-exp007)

#### [T] exp_008

* [exp_008](training/exp_008.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [AND then SEQ - Exp008](#p-exp008)

#### [T] exp_009

* [exp_009](training/exp_009.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Longer AND Branches - Exp009](#p-exp009)

#### [T] exp_010

* [exp_010](training/exp_010.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Even Longer AND Branches - Exp010](#p-exp010)

#### [T] exp_011

* [exp_011](training/exp_011.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Long Running Dependency 2 - Exp011](#p-exp011)

#### [T] exp_012

* [exp_012](training/exp_012.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Long Running Dependency 3 - Exp012](#p-exp012)

#### [T] exp_013

* [exp_013](training/exp_013.json)
    * Transformer Version: [Transformer V1](#transformer-v1)
    * Training data: [Long Running Dependency 3 - Exp013](#p-exp013)

#### [T] exp_014

* [exp_014](training/exp_014.json)
    * Transformer Version: [Transformer V2](#transformer-v2)
    * Training data: [Long Running Dependency 3 - Exp013](#p-exp013)

#### [T] exp_015

* [exp_015](training/exp_015.json)
    * Transformer Version: [Transformer V2](#transformer-v2)
    * Training data: [Longer Sequence - Exp014](#p-exp014)

#### [T] exp_016

* [exp_016](training/exp_016.json)
    * Transformer Version: [Transformer V3](#transformer-v3)
    * Training data: [Long Running Dependency 3 - Exp013](#p-exp013)

#### [T] exp_017

* [exp_017](training/exp_017.json)
    * Transformer Version: [Transformer V3](#transformer-v3)
    * Training data: [Long Running Dependency 2 - Exp015](#p-exp015)

#### [T] exp_018

* [exp_018](training/exp_018.json)
    * Transformer Version: [Transformer V3](#transformer-v3)
    * Training data: [AND then SEQ - Exp016](#p-exp016)

#### [T] exp_019

* [exp_019](training/exp_019.json)
    * Transformer Version: [Transformer V3](#transformer-v3)
    * Training data: [Even Longer AND Branches - Exp017](#p-exp017)

#### [T] exp_020

* [exp_020](training/exp_020.json)
    * Transformer Version: [Transformer V4](#transformer-v4)
    * Training data: [Even Longer AND Branches - Exp017](#p-exp017)

#### [T] exp_021

* [exp_021](training/exp_021.json)
    * Transformer Version: [Transformer V4](#transformer-v4)
    * Training data: [Longer Sequence - Exp018](#p-exp018)

#### [T] exp_022

* [exp_022](training/exp_022.json)
    * Transformer Version: [Transformer V4](#transformer-v4)
    * Training data: [Complex Start then Sequence - Exp019](#p-exp019)

#### [T] exp_023

* [exp_023](training/exp_023.json)
    * Transformer Version: [Transformer V4](#transformer-v4)
    * Training data: [Even Longer AND Branches - Exp020](#p-exp020)

#### [T] exp_024

* [exp_024](training/exp_024.json)
    * Transformer Version: [Transformer V4](#transformer-v4)
    * Training data: [Complex Model 001 - Exp021](#p-exp021)

#### [T] exp_025

* [exp_025](training/exp_025.json)
    * Transformer Version: [Transformer V4](#transformer-v4)
    * Training data: [Complex Model 001 - Exp022](#p-exp021)

#### [T] exp_026

* [exp_026](training/exp_026.json)
    * Transformer Version: [Transformer V4](#transformer-v4)
    * Training data: [Long Running Dependency 3 - Exp013](#p-exp013)

#### [T] exp_027

* [exp_027](training/exp_027.json)
    * Transformer Version: [Transformer V4](#transformer-v4)
    * Training data: [Looped AND - Exp023](#p-exp023)

---

## Explaining

### General

#### [X] exp_001

* [exp_001.json](explaining/exp_001.json)
    * Training: [NN for Simple Sequence - Exp001](#t-exp001)
    * Trace: [XAI Trace](./../PLG2/generated/README.md#simple-sequence)
    * Results: [explaining_001](../explaining/explaining_001)

#### [X] exp_002

* [exp_002.json](explaining/exp_002.json)
    * Training: [NN for Longer Sequence - Exp002](#t-exp002)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#longer-sequence)
    * Results: [explaining_002](../explaining/explaining_002)
    * Comments:
        * Sequence. Prefix A, B, ..., K. L to be predicted.
        * Strong attention on last column. Especially on A (within last column, so top right corner).
        * [Attention Heads](../explaining/explaining_002/pre_prediction.svg)

#### [X] exp_003

* [exp_003.json](explaining/exp_003.json)
    * Training: [NN for Longer Sequence - Exp002](#t-exp002)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#longer-sequence)
    * Results: [explaining_003](../explaining/explaining_003)
    * Comments:
        * Sequence. Prefix A, B, C, D. E to be predicted.
        * Strong attention on both last columns.
        * [Attention Heads](../explaining/explaining_003/pre_prediction.svg)

#### [X] exp_004

* [exp_004.json](explaining/exp_004.json)
    * Training: [NN for Simple XOR and Sequence Process - Exp003](#t-exp003)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#simple-xor-and-sequence-process)
    * Results: [explaining_004](../explaining/explaining_004)
    * Comments:
        * Simple XOR. Prefix A, B, C. E to be predicted.
        * Attention on C (last event before XOR-split).
        * [Attention Heads](../explaining/explaining_004/pre_prediction.svg)

#### [X] exp_005

* [exp_005.json](explaining/exp_005.json)
    * Training: [NN for Simple XOR and Sequence Process - Exp003](#t-exp003)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#simple-xor-and-sequence-process)
    * Results: [explaining_005](../explaining/explaining_005)
    * Comments:
        * Simple XOR. Prefix A, B, C. F to be predicted.
        * Attention on C (last event before XOR-split).
        * [Attention Heads](../explaining/explaining_005/pre_prediction.svg)

#### [X] exp_006

* [exp_006.json](explaining/exp_006.json)
    * Training: [NN for Simple AND and Sequence Process - Exp004](#t-exp004)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#simple-and-and-sequence-process)
    * Results: [explaining_006](../explaining/explaining_006)
    * Comments:
        * Simple AND. Prefix A, D. C to be predicted (within other AND-branch).
        * [Attention Heads](../explaining/explaining_006/pre_prediction.svg)

#### [X] exp_007

* [exp_007.json](explaining/exp_007.json)
    * Training: [NN for Simple AND and Sequence Process - Exp004](#t-exp004)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#simple-and-and-sequence-process)
    * Results: [explaining_007](../explaining/explaining_007)
    * Comments:
        * Simple AND. Prefix: A, C. D to be predicted.
        * Analog to the [above](#x-exp006).
        * [Attention Heads](../explaining/explaining_007/pre_prediction.svg)

#### [X] exp_008

* [exp_008.json](explaining/exp_008.json)
    * Training: [NN for Simple AND and Sequence Process - Exp004](#t-exp004)
    * Trace: [XAI Trace 003](./../PLG2/generated/README.md#simple-and-and-sequence-process)
    * Results: [explaining_008](../explaining/explaining_008)
    * Comments:
        * Simple AND. Prefix: A, D, C. B to be predicted (first activity after AND-merge).
        * Attention on both C and D (both AND branches "finished").
        * [Attention Heads](../explaining/explaining_008/pre_prediction.svg)

#### [X] exp_009

* [exp_009.json](explaining/exp_009.json)
    * Training: [NN for Simple XOR and Sequence Process - Exp003](#t-exp003)
    * Trace: [XAI Trace 003](./../PLG2/generated/README.md#simple-xor-and-sequence-process)
    * Results: [explaining_009](../explaining/explaining_009)
    * Comments:
        * Simple XOR. Prefix: A, B, C, F. D to be predicted.
        * Attention on F as either F or E is always the event before D.
        * [Attention Heads](../explaining/explaining_009/pre_prediction.svg)

#### [X] exp_010

* [exp_010.json](explaining/exp_010.json)
    * Training: [NN for Simple XOR and Sequence Process - Exp003](#t-exp003)
    * Trace: [XAI Trace 004](./../PLG2/generated/README.md#simple-xor-and-sequence-process)
    * Results: [explaining_010](../explaining/explaining_010)
    * Comments:
        * Simple XOR. Prefix: A, B, C, E. D to be predicted.
        * Analog to the [above](#x-exp009).
        * [Attention Heads](../explaining/explaining_010/pre_prediction.svg)

### AND then Sequence

#### [X] AND_then_SEQ_001

* [AND_then_SEQ_001.json](explaining/branch_then_SEQ/AND_then_SEQ_001.json)
    * Training: [NN for AND then SEQ - Exp008](#t-exp008)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#and-then-seq)
    * Results: [AND_then_SEQ_001](../explaining/branch_then_SEQ/AND_then_SEQ_001)
    * Comments:
        * AND then Sequence. Prefix: X, Y, A, B, C, D, E. F to be predicted.
        * Attention on last row, especially A and X for some reason.
        * [Attention Heads](../explaining/branch_then_SEQ/AND_then_SEQ_001/pre_prediction.svg)

#### [X] AND_then_SEQ_002

* [AND_then_SEQ_002.json](explaining/branch_then_SEQ/AND_then_SEQ_002.json)
    * Training: [NN for AND then SEQ - Exp008](#t-exp008)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#and-then-seq)
    * Results: [AND_then_SEQ_002](../explaining/branch_then_SEQ/AND_then_SEQ_002)
    * Comments:
        * AND then Sequence. Prefix: Y, X, A, B, C, D, E. F to be predicted.
        * Similar to the [above](#x-andthenseq001). Attention also on X for some reason.
        * [Attention Heads](../explaining/branch_then_SEQ/AND_then_SEQ_002/pre_prediction.svg)

#### [X] AND_then_SEQ_003

* [AND_then_SEQ_003.json](explaining/branch_then_SEQ/AND_then_SEQ_003.json)
    * Training: [NN for AND then SEQ - Exp017](#t-exp017)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#and-then-seq)
    * Results: [AND_then_SEQ_003](../explaining/branch_then_SEQ/AND_then_SEQ_003)
    * Comments:
        * AND then Sequence. Prefix: Y, X, A, B, C, D, E. F to be predicted.
        * Strong attention on E, however also on Y for whatever reason (but not on X!).
        * [Attention Heads](../explaining/branch_then_SEQ/AND_then_SEQ_003/pre_prediction.svg)

#### [X] AND_then_SEQ_004

* [AND_then_SEQ_004.json](explaining/branch_then_SEQ/AND_then_SEQ_004.json)
    * Training: [NN for AND then SEQ - Exp017](#t-exp017)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#and-then-seq)
    * Results: [AND_then_SEQ_004](../explaining/branch_then_SEQ/AND_then_SEQ_004)
    * Comments:
        * AND then Sequence. Prefix: X, Y, A, B, C, D, E. F to be predicted.
        * Strong attention on E, however also on Y for whatever reason (but not on X!).
        * [Attention Heads](../explaining/branch_then_SEQ/AND_then_SEQ_004/pre_prediction.svg)

### XOR then Sequence

#### [X] XOR_then_SEQ_001

* [XOR_then_SEQ_001.json](explaining/branch_then_SEQ/XOR_then_SEQ_001.json)
    * Training: [NN for XOR then SEQ - Exp007](#t-exp007)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#xor-then-seq)
    * Results: [XOR_then_SEQ_001](../explaining/branch_then_SEQ/XOR_then_SEQ_001)
    * Comments:
        * XOR then Sequence: Prefix: X, A, B, C, D, E. F to be predicted.
        * Attention on last column. Especially on B.
        * [Attention Heads](../explaining/branch_then_SEQ/XOR_then_SEQ_001/pre_prediction.svg)

#### [X] XOR_then_SEQ_002

* [XOR_then_SEQ_002.json](explaining/branch_then_SEQ/XOR_then_SEQ_002.json)
    * Training: [NN for XOR then SEQ - Exp007](#t-exp007)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#xor-then-seq)
    * Results: [XOR_then_SEQ_002](../explaining/branch_then_SEQ/XOR_then_SEQ_002)
    * Comments:
        * XOR then Sequence: Prefix: Y, A, B, C, D, E. F to be predicted.
        * Attention on last column. Especially on B.
        * [Attention Heads](../explaining/branch_then_SEQ/XOR_then_SEQ_002/pre_prediction.svg)

### Even Longer AND Branch

#### [X] even_longer_AND_branch_001

* [even_longer_AND_branch_001.json](explaining/complex_AND_branches/even_longer_AND_branch_001.json)
    * Training: [NN for Even Longer AND Branch - Exp010](#t-exp010)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_001](../explaining/complex_AND_branches/even_longer_AND_branch_001)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, Y, M. B to be predicted.
        * Attention more or less all over the place.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_001/pre_prediction.svg)

#### [X] even_longer_AND_branch_002

* [even_longer_AND_branch_002.json](explaining/complex_AND_branches/even_longer_AND_branch_002.json)
    * Training: [NN for Even Longer AND Branch - Exp010](#t-exp010)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_002](../explaining/complex_AND_branches/even_longer_AND_branch_002)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K. X to be predicted.
        * Attention more or less all over the place. But some order can be seen.
        * Not an intuitive explanation though except that lots of things prepend an event in AND_branches.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_002/pre_prediction.svg)

#### [X] even_longer_AND_branch_003

* [even_longer_AND_branch_003.json](explaining/complex_AND_branches/even_longer_AND_branch_003.json)
    * Training: [NN for Even Longer AND Branch - Exp010](#t-exp010)
    * Trace: [XAI Trace 003](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_003](../explaining/complex_AND_branches/even_longer_AND_branch_003)
    * Comments:
        * Two longer AND-branches. Prefix: A, I, J, U, V, W, K. L to be predicted.
        * Analog to the [above](#x-evenlongerandbranch002).
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_003/pre_prediction.svg)

#### [X] even_longer_AND_branch_004

* [even_longer_AND_branch_004.json](explaining/complex_AND_branches/even_longer_AND_branch_004.json)
    * Training: [NN for Even Longer AND Branch - Exp010](#t-exp010)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_004](../explaining/complex_AND_branches/even_longer_AND_branch_004)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, Y, M. B to be predicted.
        * Analog to the [above](#x-evenlongerandbranch001) - PAD visualized.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_004/pre_prediction.svg)

#### [X] even_longer_AND_branch_005

* [even_longer_AND_branch_005.json](explaining/complex_AND_branches/even_longer_AND_branch_005.json)
    * Training: [NN for Even Longer AND Branch - Exp010](#t-exp010)
    * Trace: [XAI Trace 004](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_005](../explaining/complex_AND_branches/even_longer_AND_branch_005)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, M, Y. B to be predicted.
        * Similar to the [above](#x-evenlongerandbranch001) - PAD visualized.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_005/pre_prediction.svg)

#### [X] even_longer_AND_branch_006

* [even_longer_AND_branch_006.json](explaining/complex_AND_branches/even_longer_AND_branch_006.json)
    * Training: [NN for Even Longer AND Branch - Exp010](#t-exp010)
    * Trace: [XAI Trace 005](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_006](../explaining/complex_AND_branches/even_longer_AND_branch_006)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, M. Y to be predicted.
        * Similar to the [above](#x-evenlongerandbranch001) - PAD visualized.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_006/pre_prediction.svg)

#### [X] even_longer_AND_branch_007

* [even_longer_AND_branch_007.json](explaining/complex_AND_branches/even_longer_AND_branch_007.json)
    * Training: [NN for Even Longer AND Branch - Exp010](#t-exp010)
    * Trace: [XAI Trace 006](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_007](../explaining/complex_AND_branches/even_longer_AND_branch_007)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, Y. M to be predicted.
        * Similar to the [above](#x-evenlongerandbranch001) - PAD visualized.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_007/pre_prediction.svg)

#### [X] even_longer_AND_branch_008

* [even_longer_AND_branch_008.json](explaining/complex_AND_branches/even_longer_AND_branch_008.json)
    * Training: [NN for Even Longer AND Branch - Exp019](#t-exp019)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_008](../explaining/complex_AND_branches/even_longer_AND_branch_008)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, Y, M. B to be predicted.
        * Basically only attention on Y, even though M should have about equal attention, too.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_008/pre_prediction.svg)

#### [X] even_longer_AND_branch_009

* [even_longer_AND_branch_009.json](explaining/complex_AND_branches/even_longer_AND_branch_009.json)
    * Training: [NN for Even Longer AND Branch - Exp019](#t-exp019)
    * Trace: [XAI Trace 004](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_009](../explaining/complex_AND_branches/even_longer_AND_branch_009)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, M, Y. B to be predicted.
        * Analog to the [above](#x-evenlongerandbranch008) but with Y and M attention switched.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_009/pre_prediction.svg)

#### [X] even_longer_AND_branch_010

* [even_longer_AND_branch_010.json](explaining/complex_AND_branches/even_longer_AND_branch_010.json)
    * Training: [NN for Even Longer AND Branch - Exp019](#t-exp019)
    * Trace: [XAI Trace 005](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_010](../explaining/complex_AND_branches/even_longer_AND_branch_010)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, M. Y to be predicted.
        * Wrong prediction! Predicted M instead of Y.
        * Attention on L (pre-last) for whatever reason. Neither on M nor X!
        * It seems as though the model learned only to look at the pre-last event...
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_010/pre_prediction.svg)

#### [X] even_longer_AND_branch_011

* [even_longer_AND_branch_011.json](explaining/complex_AND_branches/even_longer_AND_branch_011.json)
    * Training: [NN for Even Longer AND Branch - Exp019](#t-exp019)
    * Trace: [XAI Trace 006](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_011](../explaining/complex_AND_branches/even_longer_AND_branch_011)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, Y. M to be predicted.
        * Looks exactly like the [above](#x-evenlongerandbranch010) - prediction is correct now.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_011/pre_prediction.svg)

#### [X] even_longer_AND_branch_012

* [even_longer_AND_branch_012.json](explaining/complex_AND_branches/even_longer_AND_branch_012.json)
    * Training: [NN for Even Longer AND Branch - Exp020](#t-exp020)
    * Trace: [XAI Trace 005](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_012](../explaining/complex_AND_branches/even_longer_AND_branch_012)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, M. Y to be predicted.
        * Attention on L and M. Partially expected, attention on X is missing. Correct prediction.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_012/pre_prediction.svg)

#### [X] even_longer_AND_branch_013

* [even_longer_AND_branch_013.json](explaining/complex_AND_branches/even_longer_AND_branch_013.json)
    * Training: [NN for Even Longer AND Branch - Exp020](#t-exp020)
    * Trace: [XAI Trace 006](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_013](../explaining/complex_AND_branches/even_longer_AND_branch_013)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, Y. M to be predicted.
        * Attention on V, L, Y. L and Y expected. Correct prediction.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_013/pre_prediction.svg)

#### [X] even_longer_AND_branch_014

* [even_longer_AND_branch_014.json](explaining/complex_AND_branches/even_longer_AND_branch_014.json)
    * Training: [NN for Even Longer AND Branch - Exp020](#t-exp020)
    * Trace: [XAI Trace 007](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_014](../explaining/complex_AND_branches/even_longer_AND_branch_014)
    * Comments:
        * Two longer AND-branches. Prefix: A, I, J, K, L, M, U, V. W to be predicted.
        * Wrong prediction, therefore not of further interest.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_014/pre_prediction.svg)

#### [X] even_longer_AND_branch_015

* [even_longer_AND_branch_015.json](explaining/complex_AND_branches/even_longer_AND_branch_015.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 007](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_015](../explaining/complex_AND_branches/even_longer_AND_branch_015)
    * Comments:
        * Two longer AND-branches. Prefix: A, I, J, K, L, M, U, V. W to be predicted.
        * "Only" changed the training data. PLG2 generated only two different trace variants.
        * The new online tool ["PURPLE"](http://pros.unicam.it:4300/discovery/trace_frequencies) generates 137 different
          trace variants.
        * Correct prediction!
        * Attention on A (ok), L (irritating), M (good as it marks the end of one AND-branch) and V (good as it is the
          direct predecessor of W).
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_015/pre_prediction.svg)

#### [X] even_longer_AND_branch_016

* [even_longer_AND_branch_016.json](explaining/complex_AND_branches/even_longer_AND_branch_016.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 008](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_016](../explaining/complex_AND_branches/even_longer_AND_branch_016)
    * Comments:
        * Two longer AND-branches. Prefix: A, I, J, K, MASK[L], M, U, V. W to be predicted.
        * L masked out -> attention on "correct" events. Prediction: W with 90%.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_016/pre_prediction.svg)

#### [X] even_longer_AND_branch_017

* [even_longer_AND_branch_017.json](explaining/complex_AND_branches/even_longer_AND_branch_017.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 009](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_017](../explaining/complex_AND_branches/even_longer_AND_branch_017)
    * Comments:
        * Two longer AND-branches. Prefix: MASK[A], I, J, K, L, M, U, V. W to be predicted.
        * A masked out -> focus shifts to I. Prediction: W with 95%.
        * Not what we expect, as this marks A as "not important". May be circumvented somehow (common prefix).
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_017/pre_prediction.svg)

#### [X] even_longer_AND_branch_018

* [even_longer_AND_branch_018.json](explaining/complex_AND_branches/even_longer_AND_branch_018.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 010](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_018](../explaining/complex_AND_branches/even_longer_AND_branch_018)
    * Comments:
        * Two longer AND-branches. Prefix: A, I, J, K, L, MASK[M], U, V. W to be predicted.
        * M masked out -> focus on L and K (previous events in branch). Prediction: W (58%) and M (42%).
        * I.e., M should not be masked out. Matches with our expectation.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_018/pre_prediction.svg)

#### [X] even_longer_AND_branch_019

* [even_longer_AND_branch_019.json](explaining/complex_AND_branches/even_longer_AND_branch_019.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 011](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_019](../explaining/complex_AND_branches/even_longer_AND_branch_019)
    * Comments:
        * Two longer AND-branches. Prefix: A, MASK[I], MASK[J], MASK[K], MASK[L], M, MASK[U], V. W to be predicted.
        * Masked out I, J, K, L, U, i.e. all unimportant ones -> focus on A, M, V. Prediction: W with 98%.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_019/pre_prediction.svg)

#### [X] even_longer_AND_branch_020

* [even_longer_AND_branch_020.json](explaining/complex_AND_branches/even_longer_AND_branch_020.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 012](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_020](../explaining/complex_AND_branches/even_longer_AND_branch_020)
    * Comments:
        * Two longer AND-branches. Prefix: MASK[A], I, J, K, L, MASK[M], U, MASK[V]. W to be predicted.
        * Masked out A, M, V, i.e. all important (!) ones -> focus all over the place. Prediction wrong.
        * -> predicted U, V and M, i.e. two of the three masked out ones.
        * -> can use this to underline that A, M and V are important.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_020/pre_prediction.svg)

#### [X] even_longer_AND_branch_021

* [even_longer_AND_branch_021.json](explaining/complex_AND_branches/even_longer_AND_branch_021.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_021](../explaining/complex_AND_branches/even_longer_AND_branch_021)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, Y, M, B to be predicted.
        * Focus on A, L, M, X, Y. Should only be A, M, Y. Prediction correct.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_021/pre_prediction.svg)

#### [X] even_longer_AND_branch_022

* [even_longer_AND_branch_022.json](explaining/complex_AND_branches/even_longer_AND_branch_022.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 013](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_022](../explaining/complex_AND_branches/even_longer_AND_branch_022)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, MASK[X], MASK[L], Y, M, B to be predicted.
        * Masked out X and L -> still correct prediction. Confirmed that A, M, Y are important.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_022/pre_prediction.svg)

#### [X] even_longer_AND_branch_023

* [even_longer_AND_branch_023.json](explaining/complex_AND_branches/even_longer_AND_branch_023.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 014](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_023](../explaining/complex_AND_branches/even_longer_AND_branch_023)
    * Comments:
        * Two longer AND-branches. Prefix: A, U, I, V, J, W, K, X, L, MASK[Y], MASK[M], B to be predicted.
        * Masked out Y and M, got different prediction (M 53%, Y 49%). Confirmed that M and Y are important.
        * [Attention Heads](../explaining/complex_AND_branches/even_longer_AND_branch_023/pre_prediction.svg)

#### [X] even_longer_AND_branch_024

* [even_longer_AND_branch_024.json](explaining/complex_AND_branches/even_longer_AND_branch_024.json)
    * Training: [NN for Even Longer AND Branch - Exp023](#t-exp023)
    * Trace: [XAI Trace 007](./../PLG2/generated/README.md#even-longer-and-branches)
    * Results: [even_longer_AND_branch_024](../explaining/complex_AND_branches/even_longer_AND_branch_024)
    * Comments:
        * Two longer AND-branches. Prefix: A, I, J, K, L, M, U, V. W to be predicted.
        * Explainer: AttentionRestrictedDirectlyFollowsGraph.
        * Model almost correctly reproduced. Missing are A->I and A->U. NN does not lay attention on A here.
        * Yes, with an empty trace, I and U have a high prediction chance (total about 99%).
        * [Graph](../explaining/complex_AND_branches/even_longer_AND_branch_024/graph.svg)
        * [Original](../PLG2/generated/10_even_longer_AND_branches/even_longer_AND_branches.svg)

### Longer AND branch

#### [X] longer_AND_branch_001

* [longer_AND_branch_001.json](explaining/complex_AND_branches/longer_AND_branch_001.json)
    * Training: [NN for Longer AND Branch - Exp009](#t-exp009)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#longer-and-branches)
    * Results: [longer_AND_branch_001](../explaining/complex_AND_branches/longer_AND_branch_001)
    * Comments:
        * Two AND-branches, one with two events. Prefix: A, U, I, J. B to be predicted.
        * Shows what events contribute to the AND-merge (and event B afterward).
        * [Attention Heads](../explaining/complex_AND_branches/longer_AND_branch_001/pre_prediction.svg)

#### [X] longer_AND_branch_002

* [longer_AND_branch_002.json](explaining/complex_AND_branches/longer_AND_branch_002.json)
    * Training: [NN for Longer AND Branch - Exp009](#t-exp009)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#longer-and-branches)
    * Results: [longer_AND_branch_002](../explaining/complex_AND_branches/longer_AND_branch_002)
    * Comments:
        * Two AND-branches, one with two events. Prefix: A, I, U, J. B to be predicted.
        * Similar to the [above](#x-longerandbranch001).
        * [Attention Heads](../explaining/complex_AND_branches/longer_AND_branch_002/pre_prediction.svg)

#### [X] longer_AND_branch_003

* [longer_AND_branch_003.json](explaining/complex_AND_branches/longer_AND_branch_003.json)
    * Training: [NN for Longer AND Branch - Exp009](#t-exp009)
    * Trace: [XAI Trace 003](./../PLG2/generated/README.md#longer-and-branches)
    * Results: [longer_AND_branch_003](../explaining/complex_AND_branches/longer_AND_branch_003)
    * Comments:
        * Two AND-branches, one with two events. Prefix: A, I, J, U. B to be predicted.
        * Similar to the [above](#x-longerandbranch001).
        * [Attention Heads](../explaining/complex_AND_branches/longer_AND_branch_003/pre_prediction.svg)

#### [X] longer_AND_branch_004

* [longer_AND_branch_004.json](explaining/complex_AND_branches/longer_AND_branch_004.json)
    * Training: [NN for Longer AND Branch - Exp009](#t-exp009)
    * Trace: [XAI Trace 004](./../PLG2/generated/README.md#longer-and-branches)
    * Results: [longer_AND_branch_004](../explaining/complex_AND_branches/longer_AND_branch_004)
    * Comments:
        * Two AND-branches, one with two events. Prefix: A, I, J. U to be predicted.
        * Similar to the [above](#x-longerandbranch001) but within the AND-branches.
        * [Attention Heads](../explaining/complex_AND_branches/longer_AND_branch_004/pre_prediction.svg)

#### [X] longer_AND_branch_005

* [longer_AND_branch_005.json](explaining/complex_AND_branches/longer_AND_branch_005.json)
    * Training: [NN for Longer AND Branch - Exp009](#t-exp009)
    * Trace: [XAI Trace 005](./../PLG2/generated/README.md#longer-and-branches)
    * Results: [longer_AND_branch_005](../explaining/complex_AND_branches/longer_AND_branch_005)
    * Comments:
        * Two AND-branches, one with two events. Prefix: A, I, U. J to be predicted.
        * Similar to the [above](#x-longerandbranch001) but within the AND-branches.
        * [Attention Heads](../explaining/complex_AND_branches/longer_AND_branch_005/pre_prediction.svg)

#### [X] longer_AND_branch_006

* [longer_AND_branch_006.json](explaining/complex_AND_branches/longer_AND_branch_006.json)
    * Training: [NN for Longer AND Branch - Exp009](#t-exp009)
    * Trace: [XAI Trace 006](./../PLG2/generated/README.md#longer-and-branches)
    * Results: [longer_AND_branch_006](../explaining/complex_AND_branches/longer_AND_branch_006)
    * Comments:
        * Two AND-branches, one with two events. Prefix: A, I. U to be predicted.
        * Similar to the [above](#x-longerandbranch001) but within the AND-branches.
        * [Attention Heads](../explaining/complex_AND_branches/longer_AND_branch_006/pre_prediction.svg)

#### [X] longer_AND_branch_007

* [longer_AND_branch_007.json](explaining/complex_AND_branches/longer_AND_branch_007.json)
    * Training: [NN for Longer AND Branch - Exp009](#t-exp009)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#longer-and-branches)
    * Results: [longer_AND_branch_007](../explaining/complex_AND_branches/longer_AND_branch_007)
    * Comments:
        * Two AND-branches, one with two events. Prefix: A, I. U to be predicted.
        * Attention Matrix contains padding (other than that equal to [001](#x-longerandbranch001))
        * [Attention Heads](../explaining/complex_AND_branches/longer_AND_branch_007/pre_prediction.svg)

### Long Running Dependency

#### [X] long_running_dep_001

* [long_running_dep_001.json](explaining/long_running_dependency/long_running_dep_001.json)
    * Training: [NN for Long Running Dependency - Exp006](#t-exp006)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#long-running-dependency)
    * Results: [long_running_dep_001](../explaining/long_running_dep/long_running_dep_001)
    * Comments:
        * Smallest long-running dependency. Prefix: B, C. E to be predicted.
        * Attention strongly on B as expected.
        * [Attention Heads](../explaining/long_running_dep/long_running_dep_001/pre_prediction.svg)

#### [X] long_running_dep_002

* [long_running_dep_002.json](explaining/long_running_dependency/long_running_dep_002.json)
    * Training: [NN for Long Running Dependency - Exp006](#t-exp006)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#long-running-dependency)
    * Results: [long_running_dep_002](../explaining/long_running_dep/long_running_dep_002)
    * Comments:
        * Smallest long-running dependency. Prefix: A, C. E to be predicted.
        * Attention strongly on A as expected. Also on C though, which is ok (as we merge).
        * [Attention Heads](../explaining/long_running_dep/long_running_dep_002/pre_prediction.svg)

### Long Running Dependency 2

#### [X] long_running_dep2_001

* [long_running_dep2_001.json](explaining/long_running_dependency2/long_running_dep2_001.json)
    * Training: [NN for Long Running Dependency 2 - Exp011](#t-exp011)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#long-running-dependency-2)
    * Results: [long_running_dep2_001](../explaining/long_running_dep2/long_running_dep2_001)
    * Comments:
        * Longer long-running dependency. Prefix: A, U, V, W, X, Y, Z. D to be predicted.
        * Attention strongly on Z as expected. Only minor attention on A.
        * [Attention Heads](../explaining/long_running_dep2/long_running_dep2_001/pre_prediction.svg)

#### [X] long_running_dep2_002

* [long_running_dep2_002.json](explaining/long_running_dependency2/long_running_dep2_002.json)
    * Training: [NN for Long Running Dependency 2 - Exp011](#t-exp011)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#long-running-dependency-2)
    * Results: [long_running_dep2_002](../explaining/long_running_dep2/long_running_dep2_002)
    * Comments:
        * Longer long-running dependency. Prefix: B, U, V, W, X, Y, Z. E to be predicted.
        * Analog to the [above](#x-longrunningdep2001).
        * [Attention Heads](../explaining/long_running_dep2/long_running_dep2_002/pre_prediction.svg)

#### [X] long_running_dep2_003

* [long_running_dep2_003.json](explaining/long_running_dependency2/long_running_dep2_003.json)
    * Training: [NN for Long Running Dependency 2 - Exp011](#t-exp011)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#long-running-dependency-2)
    * Results: [long_running_dep2_003](../explaining/long_running_dep2/long_running_dep2_003)
    * Comments:
        * Attention Matrix contains padding (other than that equal to [002](#x-longrunningdep2002)).
        * [Attention Heads](../explaining/long_running_dep2/long_running_dep2_003/pre_prediction.svg)

#### [X] long_running_dep2_004

* [long_running_dep2_004.json](explaining/long_running_dependency2/long_running_dep2_004.json)
    * Training: [NN for Long Running Dependency 2 - Exp011](#t-exp011)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#long-running-dependency-2)
    * Results: [long_running_dep2_004](../explaining/long_running_dep2/long_running_dep2_004)
    * Comments:
        * Attention Matrix contains padding (other than that equal to [002](#x-longrunningdep2001)).
        * [Attention Heads](../explaining/long_running_dep2/long_running_dep2_004/pre_prediction.svg)

#### [X] long_running_dep2_005

* [long_running_dep2_005.json](explaining/long_running_dependency2/long_running_dep2_005.json)
    * Training: [NN for Long Running Dependency 2 - Exp017](#t-exp017)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#long-running-dependency-2)
    * Results: [long_running_dep2_005](../explaining/long_running_dep2/long_running_dep2_005)
    * Comments:
        * Longer long-running dependency. Prefix: B, U, V, W, X, Y, Z. E to be predicted.
        * Attention on B, without doubt. However, only weak on Z.
        * [Attention Heads](../explaining/long_running_dep2/long_running_dep2_005/pre_prediction.svg)

### Long Running Dependency 3

#### [X] long_running_dep3_001

* [long_running_dep3_001.json](explaining/long_running_dependency3/long_running_dep3_001.json)
    * Training: [NN for Long Running Dependency 3 - Exp012](#t-exp012)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#long-running-dependency-3)
    * Results: [long_running_dep3_001](../explaining/long_running_dep3/long_running_dep3_001)
    * Comments:
        * Longer long-running dependency. Prefix: A, R, S, T, U, V, W, X, Y, Z. D to be predicted.
        * Similar to [long_running_dep2_001](#x-longrunningdep2001) - less structured though (more chaos).
        * [Attention Heads](../explaining/long_running_dep3/long_running_dep3_001/pre_prediction.svg)

#### [X] long_running_dep3_002

* [long_running_dep3_002.json](explaining/long_running_dependency3/long_running_dep3_002.json)
    * Training: [NN for Long Running Dependency 3 - Exp012](#t-exp012)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#long-running-dependency-3)
    * Results: [long_running_dep3_002](../explaining/long_running_dep3/long_running_dep3_002)
    * Comments:
        * Longer long-running dependency. Prefix: B, R, S, T, U, V, W, X, Y, Z. E to be predicted.
        * Analog to [long_running_dep2_001](#x-longrunningdep3001).
        * [Attention Heads](../explaining/long_running_dep3/long_running_dep3_002/pre_prediction.svg)

#### [X] long_running_dep3_003

* [long_running_dep3_003.json](explaining/long_running_dependency3/long_running_dep3_003.json)
    * Training: [NN for Long Running Dependency 3 - Exp013](#t-exp013)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#long-running-dependency-3)
    * Results: [long_running_dep3_003](../explaining/long_running_dep3/long_running_dep3_003)
    * Comments:
        * Longer long-running dependency. Prefix: A, R, S, T, U, V, W, X, Y, Z. D to be predicted.
        * Other training data than [X-long_running_dep3_003](#x-longrunningdep3001)!
        * Can see dependency on both A and Z quite well!
        * [Attention Heads](../explaining/long_running_dep3/long_running_dep3_003/pre_prediction.svg)

#### [X] long_running_dep3_004

* [long_running_dep3_004.json](explaining/long_running_dependency3/long_running_dep3_004.json)
    * Training: [NN for Long Running Dependency 3 - Exp013](#t-exp013)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#long-running-dependency-3)
    * Results: [long_running_dep3_004](../explaining/long_running_dep3/long_running_dep3_004)
    * Comments:
        * Longer long-running dependency. Prefix: B, R, S, T, U, V, W, X, Y, Z. E to be predicted.
        * Same training data as [X-long_running_dep3_003](#x-longrunningdep3003) but worse results again!
        * [Attention Heads](../explaining/long_running_dep3/long_running_dep3_004/pre_prediction.svg)

#### [X] long_running_dep3_005

* [long_running_dep3_005.json](explaining/long_running_dependency3/long_running_dep3_005.json)
    * Training: [NN for Long Running Dependency 3 - Exp014](#t-exp014)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#long-running-dependency-3)
    * Results: [long_running_dep3_005](../explaining/long_running_dep3/long_running_dep3_005)
    * Comments:
        * Longer long-running dependency. Prefix: A, R, S, T, U, V, W, X, Y, Z. D to be predicted.
        * Other training data (random-split).
        * Other NN: Transformer V2.
        * Can see A and Z very well!
        * [Attention Heads](../explaining/long_running_dep3/long_running_dep3_005/pre_prediction.svg)

#### [X] long_running_dep3_006

* [long_running_dep3_006.json](explaining/long_running_dependency3/long_running_dep3_006.json)
    * Training: [NN for Long Running Dependency 3 - Exp016](#t-exp016)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#long-running-dependency-3)
    * Results: [long_running_dep3_006](../explaining/long_running_dep3/long_running_dep3_006)
    * Comments:
        * Longer long-running dependency. Prefix: A, R, S, T, U, V, W, X, Y, Z. D to be predicted.
        * Other training data (random-split).
        * Other NN: Transformer V3.
        * Can see A and Z extremely well!
        * [Attention Heads](../explaining/long_running_dep3/long_running_dep3_006/pre_prediction.svg)

#### [X] long_running_dep3_007

* [long_running_dep3_007.json](explaining/long_running_dependency3/long_running_dep3_007.json)
    * Training: [NN for Long Running Dependency 3 - Exp016](#t-exp016)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#long-running-dependency-3)
    * Results: [long_running_dep3_007](../explaining/long_running_dep3/long_running_dep3_007)
    * Comments:
        * Longer long-running dependency. Prefix: A, R, S, T, U, V, W, X, Y, Z. D to be predicted.
        * Other training data (random-split).
        * Other NN: Transformer V3.
        * Can see Z well. B cannot be seen as well. Additionally, U infers for whatever reason.
        * [Attention Heads](../explaining/long_running_dep3/long_running_dep3_007/pre_prediction.svg)

#### [X] long_running_dep3_008

* [long_running_dep3_008.json](explaining/long_running_dependency3/long_running_dep3_008.json)
    * Training: [NN for Long Running Dependency 3 - Exp026](#t-exp026)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#long-running-dependency-3)
    * Results: [long_running_dep3_008](../explaining/long_running_dep3/long_running_dep3_008)
    * Comments:
        * Longer long-running dependency. Prefix: A, R, S, T, U, V, W, X, Y, Z. D to be predicted.
        * Explainer: AttentionRestrictedDirectlyFollowsGraph.
        * Model almost correctly reproduced. Some missing. Attention correct.
        * [Graph](../explaining/long_running_dep3/long_running_dep3_008/graph.svg)
        * [Original](../PLG2/generated/12_long_running_dependency3/long_running_dependency3.svg)

### Shortened Prefix Sequence

#### [X] shortened_prefix_sequence_001

* [shortened_prefix_sequence_001.json](explaining/shortened_prefixes/shortened_prefix_sequence_001.json)
    * Training: [NN for Longer Sequence - Exp002](#t-exp002)
    * Trace: [XAI Trace 003](./../PLG2/generated/README.md#longer-sequence)
    * Results: [shortened_prefix_sequence_001](../explaining/shortened_prefixes/shortened_prefix_sequence_001)
    * Comments:
        * Sequence. Prefix (start cut off): D, E, F, G, H, I, J, K. L to be predicted.
        * Focus on last column.
        * [Attention Heads](../explaining/shortened_prefixes/shortened_prefix_sequence_001/pre_prediction.svg)

#### [X] shortened_prefix_sequence_002

* [shortened_prefix_sequence_002.json](explaining/shortened_prefixes/shortened_prefix_sequence_002.json)
    * Training: [NN for Longer Sequence - Exp002](#t-exp002)
    * Trace: [XAI Trace 004](./../PLG2/generated/README.md#longer-sequence)
    * Results: [shortened_prefix_sequence_002](../explaining/shortened_prefixes/shortened_prefix_sequence_002)
    * Comments:
        * Sequence. Prefix (start cut off): H, I, J, K. L to be predicted.
        * Focus on last column.
        * [Attention Heads](../explaining/shortened_prefixes/shortened_prefix_sequence_002/pre_prediction.svg)

#### [X] shortened_prefix_sequence_003

* [shortened_prefix_sequence_003.json](explaining/shortened_prefixes/shortened_prefix_sequence_003.json)
    * Training: [NN for Longer Sequence - Exp002](#t-exp002)
    * Trace: [XAI Trace 003](./../PLG2/generated/README.md#longer-sequence)
    * Results: [shortened_prefix_sequence_003](../explaining/shortened_prefixes/shortened_prefix_sequence_003)
    * Comments:
        * Attention matrix contains padding (other that that equal to [001](#x-shortenedprefixsequence001)).
        * [Attention Heads](../explaining/shortened_prefixes/shortened_prefix_sequence_003/pre_prediction.svg)

#### [X] shortened_prefix_sequence_004

* [shortened_prefix_sequence_004.json](explaining/shortened_prefixes/shortened_prefix_sequence_004.json)
    * Training: [NN for Longer Sequence - Exp002](#t-exp002)
    * Trace: [XAI Trace 004](./../PLG2/generated/README.md#longer-sequence)
    * Results: [shortened_prefix_sequence_004](../explaining/shortened_prefixes/shortened_prefix_sequence_004)
    * Comments:
        * Attention matrix contains padding (other that that equal to [002](#x-shortenedprefixsequence002))
        * [Attention Heads](../explaining/shortened_prefixes/shortened_prefix_sequence_004/pre_prediction.svg)

#### [X] shortened_prefix_sequence_005

* [shortened_prefix_sequence_005.json](explaining/shortened_prefixes/shortened_prefix_sequence_005.json)
    * Training: [NN for Longer Sequence - Exp015](#t-exp015)
    * Trace: [XAI Trace 004](./../PLG2/generated/README.md#longer-sequence)
    * Results: [shortened_prefix_sequence_005](../explaining/shortened_prefixes/shortened_prefix_sequence_005)
    * Comments:
        * Sequence. Prefix (start cut off): H, I, J, K. L to be predicted.
        * Other NN: Transformer V2.
        * Possibly better visualization/explanation of sequence.
        * [Attention Heads](../explaining/shortened_prefixes/shortened_prefix_sequence_005/pre_prediction.svg)

#### [X] shortened_prefix_sequence_006

* [shortened_prefix_sequence_006.json](explaining/shortened_prefixes/shortened_prefix_sequence_006.json)
    * Training: [NN for Longer Sequence - Exp021](#t-exp021)
    * Trace: [XAI Trace 005](./../PLG2/generated/README.md#longer-sequence)
    * Results: [shortened_prefix_sequence_006](../explaining/shortened_prefixes/shortened_prefix_sequence_006)
    * Comments:
        * Sequence. Prefix (start cut off): H, J, K, I. L to be predicted.
        * Wrong prediction: J instead of L. I.e. the transformer learns to attend to the last event and
          output the following activity (I -> J).
        * [Attention Heads](../explaining/shortened_prefixes/shortened_prefix_sequence_006/pre_prediction.svg)

### Complex Start then Sequence

#### [X] complex_start_then_SEQ_001

* [complex_start_then_SEQ_001](explaining/complex_start_then_SEQ/complex_start_then_SEQ_001.json)
    * Training: [NN for Complex Start then Sequence - Exp022](#t-exp022)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#complex-start-then-sequence)
    * Results: [complex_start_then_SEQ_001](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_001)
    * Comments:
        * Complex XOR-branching at the start, then a short Sequence. Prefix: A, B, C, D, H, X. Y to be predicted.
        * Focus strongly on X, a bit of attention on C. Prediction correct.
        * [Attention Heads](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_001/pre_prediction.svg)

#### [X] complex_start_then_SEQ_002

* [complex_start_then_SEQ_002](explaining/complex_start_then_SEQ/complex_start_then_SEQ_002.json)
    * Training: [NN for Complex Start then Sequence - Exp022](#t-exp022)
    * Trace: [XAI Trace 002](./../PLG2/generated/README.md#complex-start-then-sequence)
    * Results: [complex_start_then_SEQ_002](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_002)
    * Comments:
        * Complex XOR-branching at the start, then a short Sequence. Prefix: A, B, C, D. H to be predicted.
        * Focus strongly on D and a bit of A. As expected. Prediction correct.
        * [Attention Heads](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_002/pre_prediction.svg)

#### [X] complex_start_then_SEQ_003

* [complex_start_then_SEQ_003](explaining/complex_start_then_SEQ/complex_start_then_SEQ_003.json)
    * Training: [NN for Complex Start then Sequence - Exp022](#t-exp022)
    * Trace: [XAI Trace 003](./../PLG2/generated/README.md#complex-start-then-sequence)
    * Results: [complex_start_then_SEQ_003](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_003)
    * Comments:
        * Complex XOR-branching at the start, then a short Sequence. Prefix: A, B. C to be predicted.
        * Focus on A. Focus on B only where there is padding. Correct prediction though.
        * [Attention Heads](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_003/pre_prediction.svg)

#### [X] complex_start_then_SEQ_004

* [complex_start_then_SEQ_004](explaining/complex_start_then_SEQ/complex_start_then_SEQ_004.json)
    * Training: [NN for Complex Start then Sequence - Exp022](#t-exp022)
    * Trace: [XAI Trace 004](./../PLG2/generated/README.md#complex-start-then-sequence)
    * Results: [complex_start_then_SEQ_004](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_004)
    * Comments:
        * Complex XOR-branching at the start, then a short Sequence. Prefix: A, B, C, MASK[D]. H to be predicted.
        * Focus now on A instead of D. Prediction is D, obviously.
        * [Attention Heads](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_004/pre_prediction.svg)

#### [X] complex_start_then_SEQ_005

* [complex_start_then_SEQ_005](explaining/complex_start_then_SEQ/complex_start_then_SEQ_005.json)
    * Training: [NN for Complex Start then Sequence - Exp022](#t-exp022)
    * Trace: [XAI Trace 005](./../PLG2/generated/README.md#complex-start-then-sequence)
    * Results: [complex_start_then_SEQ_005](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_005)
    * Comments:
        * Complex XOR-branching at the start, then a short Sequence. Prefix: A, B, MASK[C], D. H to be predicted.
        * Pretty much the same as [002](#x-complexstartthenseq002).
        * [Attention Heads](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_005/pre_prediction.svg)

#### [X] complex_start_then_SEQ_006

* [complex_start_then_SEQ_006](explaining/complex_start_then_SEQ/complex_start_then_SEQ_006.json)
    * Training: [NN for Complex Start then Sequence - Exp022](#t-exp022)
    * Trace: [XAI Trace 006](./../PLG2/generated/README.md#complex-start-then-sequence)
    * Results: [complex_start_then_SEQ_006](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_006)
    * Comments:
        * Complex XOR-branching at the start, then a short Sequence. Prefix: A, MASK[B], C, D. H to be predicted.
        * Pretty much the same as [002](#x-complexstartthenseq002).
        * [Attention Heads](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_006/pre_prediction.svg)

#### [X] complex_start_then_SEQ_007

* [complex_start_then_SEQ_007](explaining/complex_start_then_SEQ/complex_start_then_SEQ_007.json)
    * Training: [NN for Complex Start then Sequence - Exp022](#t-exp022)
    * Trace: [XAI Trace 007](./../PLG2/generated/README.md#complex-start-then-sequence)
    * Results: [complex_start_then_SEQ_007](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_007)
    * Comments:
        * Complex XOR-branching at the start, then a short Sequence. Prefix: MASK[A], B, C, D. H to be predicted.
        * Pretty much the same as [002](#x-complexstartthenseq002).
        * [Attention Heads](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_007/pre_prediction.svg)

#### [X] complex_start_then_SEQ_008

* [complex_start_then_SEQ_008](explaining/complex_start_then_SEQ/complex_start_then_SEQ_008.json)
    * Training: [NN for Complex Start then Sequence - Exp022](#t-exp022)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#complex-start-then-sequence)
    * Results: [complex_start_then_SEQ_008](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_008)
    * Comments:
        * Complex XOR-branching at the start, then a short Sequence. Prefix: A, B, C, D, H, X. Y to be predicted.
        * Explainer: AttentionRestrictedDirectlyFollowsGraph.
        * Attention ok. Some edges are missing.
        * [Graph](../explaining/complex_start_then_SEQ/complex_start_then_SEQ_008/graph.svg)
        * [Original](../PLG2/generated/13_complex_start_then_SEQ/complex_start_then_sequence.svg)

### Complex Model 001

#### [X] complex_model_001_001

* [complex_model_001_001](explaining/14_complex_models/complex_model_001/complex_model_001_001.json)
    * Training: [NN for Complex Model 001 - Exp024](#t-exp024)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#complex-model-001)
    * Results: [complex_model_001_001](../explaining/14_complex_models/complex_model_001/complex_model_001_001)
    * Comments:
        * Complex model with loop, AND, and XOR. Prefix: A, D, P, R, E, H, S, F, K, Q. T to be predicted.
        * Somewhat as expected: T is the last possible event (99% predicted). Of the two other parallel branches,
          K gives attention from the first branch. F gives attention from the second branch.
        * From its own, third branch, Q and S give attention but not R. Additionally, A (start) gives attention.
        * Looks good, but cannot infer any control-flow-logic whatsoever.
        * [Attention Heads](../explaining/14_complex_models/complex_model_001/complex_model_001_001/pre_prediction.svg)

#### [X] complex_model_001_008

* [complex_model_001_008](explaining/14_complex_models/complex_model_001/complex_model_001_008.json)
    * Training: [NN for Complex Model 001 - Exp025](#t-exp025)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#complex-model-001)
    * Results: [complex_model_001_008](../explaining/14_complex_models/complex_model_001/complex_model_001_008)
    * Comments:
        * Complex model with loop, AND, and XOR. Prefix: A, D, P, R, E, H, S, F, K, Q. T to be predicted.
        * Explainer: AttentionRestrictedDirectlyFollowsGraph.
        * Has lots of missing/wrong edges. NN does not perform that good though.
        * [Graph](../explaining/14_complex_models/complex_model_001/complex_model_001_008/graph.svg)
        * [Original](../PLG2/generated/14_complex_models/complex_model_001/complex_model_001.svg)

#### [X] complex_model_001_009

* [complex_model_001_009](explaining/14_complex_models/complex_model_001/complex_model_001_009.json)
    * Training: [NN for Complex Model 001 - Exp025](#t-exp025)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#complex-model-001)
    * Results: [complex_model_001_009](../explaining/14_complex_models/complex_model_001/complex_model_001_009)
    * Comments:
        * Complex model with loop, AND, and XOR. Prefix: A, D, P, R, E, H, S, F, K, Q. T to be predicted.
        * Explainer: AttentionRestrictedDirectlyFollowsGraph.
        * Even worse than the [above](#x-complexmodel001008), just changed a few parameters.
        * [Graph](../explaining/14_complex_models/complex_model_001/complex_model_001_009/graph.svg)
        * [Original](../PLG2/generated/14_complex_models/complex_model_001/complex_model_001.svg)

#### [X] complex_model_001_009

* [complex_model_001_009](explaining/14_complex_models/complex_model_001/complex_model_001_009.json)
    * Training: [NN for Complex Model 001 - Exp025](#t-exp025)
    * Trace: [XAI Trace 001](./../PLG2/generated/README.md#complex-model-001)
    * Results: [complex_model_001_009](../explaining/14_complex_models/complex_model_001/complex_model_001_009)
    * Comments:
        * Complex model with loop, AND, and XOR. Prefix: A, D, P, R, E, H, S, F, K, Q. T to be predicted.
        * Explainer: AttentionRestrictedDirectlyFollowsGraph.
        * Even worse than the [above](#x-complexmodel001008), just changed a few parameters.
        * [Graph](../explaining/14_complex_models/complex_model_001/complex_model_001_009/graph.svg)
        * [Original](../PLG2/generated/14_complex_models/complex_model_001/complex_model_001.svg)

### Looped AND

#### [X] looped_AND_001

* [looped_AND_001](explaining/15_looped_AND/looped_and_001.json)
  * Training: [NN for Looped AND Model - Exp027](#t-exp027)
  * Trace: [XAI Trace 001](./../PLG2/generated/README.md#looped-and)
  * Results: [looped_and_001](../explaining/15_looped_AND/looped_and_001)
  * Comments:
    * Model with AND within loop. Prefix: A, I, M, J, N, M, N, I, J, M, N, I, J, M, I, J, N, M, N, I, J, I, J, M, N. Z to be predicted
    * Explainer: AttentionRestrictedDirectlyFollowsGraph.
    * Too many edges, of course
    * [Graph](../explaining/15_looped_AND/looped_and_001/graph.svg)
    * [Original](../PLG2/generated/15_looped_AND/looped_AND.bpmn)

---

## Transformer Versions

#### Transformer V1

Original from paper.
The changes only affect what is stored within the Transformer (e.g. attention scores).

Git tag: `transformer-v1`

Git hash: `eff4628f69ec290a6368d7706cb1919e7fabce50`

#### Transformer V2

With "vertical" masking (i.e. only PAD-columns are masked, not PAD-rows).

Git tag: `transformer-v2`

Git hash: `0e9df5a9c3c07d7669aa96c8ef95a6d859a7e0e5`

#### Transformer V3

With "full" masking (i.e. PAD-columns and PAD-rows are masked)

Git tag: `transformer-v3`

Git hash: `9157d2ee2dc512204c46e968943e8257aa52066e`

#### Transformer V4

See Transformer V3 + disabled dropout during inference.

Git tag: `transformer-v4`

Git hash: `b28e93f289fe9f1e812e335f486c25b0f6f90228`

#### Transformer V5

Just copied MultiHeadAttention-sourcefile from tensorflow to own repository.
No further changes.

Git tag: `transformer-v5`

Git hash: `4dfdc044782307c9ea2f7ff4982cca51781c517a`

#### Transformer V6

Added option to directly add an attention-mask.

Git tag: `transformer-v6`

Git hash: `86ca01a963224c81784c6230d72f37da2cbd7400`

#### Transformer V7

Proper saving and loading of model.

Git tag: `transformer-v7`

Git hash: `5f2d12363c37bdda9baeddd9847271314d585417`

---

## Noteworthy Explanations

### Sequence

* [X-exp002](#x-exp002) (simple sequence)
* [X-exp006](#x-exp007) (possibly interesting: Attention on D -> good)
* [X-AND_then_SEQ_001](#x-andthenseq001) (attention on last event and A, X)
    * Similar: [X-XOR_then_SEQ_001](#x-xorthenseq001) (attention on last event and B)
* [X-shortened_prefix_sequence_001](#x-shortenedprefixsequence001) (shortened prefix - focus still on last column)
    * [X-shortened_prefix_sequence_005](#x-shortenedprefixsequence005) (Transformer V2 for comparison)
    * [X-shortened_prefix_sequence_006](#x-shortenedprefixsequence006) (Transformer V4 - attention only on last event)
* [X-complex_start_then_SEQ_001](#x-complexstartthenseq001) (complex start does not throw of transformer - somewhat
  interesting)

### AND-branches

* [X-even_longer_AND_branch_001](#x-evenlongerandbranch001) (anti-example, not explainable)
* Anti-examples
    * Random split + Transformer V3.
    * All look the same for whatever reason.
    * [X-even_longer_AND_branch_008](#x-evenlongerandbranch008)
    * [X-even_longer_AND_branch_009](#x-evenlongerandbranch009)
    * [X-even_longer_AND_branch_010](#x-evenlongerandbranch010) (wrong prediction!)
    * [X-even_longer_AND_branch_011](#x-evenlongerandbranch011)
* Good examples
    * Random split + Transformer V4.
    * Attention scores look different, predictions are correct now.
    * [X-even_longer_AND_branch_012](#x-evenlongerandbranch012) - similar to [010](#x-evenlongerandbranch010) but better
    * [X-even_longer_AND_branch_013](#x-evenlongerandbranch013) - similar to [011](#x-evenlongerandbranch011) but better
    * [X-even_longer_AND_branch_015](#x-evenlongerandbranch015) - very good example! Uses more balanced training data
    * [X-even_longer_AND_branch_024](#x-evenlongerandbranch024) - using AttentionRestrictedDirectlyFollowsGraph - works good.

### Long Running Dependencies

* [X-long_running_dep_001](#x-longrunningdep001) (good example)
* [X-long_running_dep2_001](#x-longrunningdep2001) (anti-example)
    * [X-long_running_dep2_005](#x-longrunningdep2005) (good example - Transformer V3)
* [X-long_running_dep3_003](#x-longrunningdep3003) (good example -
  other training data than [X-long_running_dep3_001](#x-longrunningdep3001))
* [X-long_running_dep3_005](#x-longrunningdep3005) (good example - Transformer V2)
* [X-long_running_dep3_006](#x-longrunningdep3006) (good example - Transformer V3)
    * [X-long_running_dep3_007](#x-longrunningdep3007) (anti example - Transformer V3)
* [X-long_running_dep3_006](#x-longrunningdep3008) (good example - AttentionRestrictedDirectlyFollowsGraph)

### Complex Model
* [X-complex_model_008](#x-complexmodel001008) (anti example - does not work good. NN does neither, though)
