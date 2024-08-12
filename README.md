## How to Replicate Experiments
If you want to use the explainer via gui, please run processtransformer\gui\main_windows.py

Take a look at the different configurations in the experiment folder.
- Preprocessing for event logs
- Training for training the transformer models
- Explaining for the XAI approaches

All scripts assume the working directory to be the root of this repository.
All trained models, figures, and tables can be found in the supplementary material: https://drive.google.com/drive/folders/1JtlCIQlxHo-YEmbTB01Atq2M786h3Wvg?usp=sharing 
If you use PyCharm, the different run configurations should show up automatically.
If not, use the following scripts. Conda-environment is required.

For preprocessing:
```python
python processtransformer\ml\preprocess\data_preprocessing.py --experiment_file_path experiments/preprocessing/exp_helpdesk.json
```

For training (requires preprocessing):
```python
python main.py --training_config experiments/training/exp_helpdesk_GPU.json
```

For explaining (requires training):
```python
python processtransformer\xai\main.py --explaining_config experiments/explaining/helpdesk/helpdesk-attn_expl_xai.json
```

For running conceptual-metrics on a certain dataset and XAI approach (requires training; takes forever!):
```python
python processtransformer\xai\metrics\run_through_xai_metrics.py --xai_config experiments/explaining/helpdesk/helpdesk-attn_expl_xai.json
```

For running attention-metrics on a certain dataset (requires training; depending on your GPU, this may be fast):
```python
python processtransformer\xai\metrics\run_through_nn_metrics.py --nn_config experiments/training/exp_helpdesk_GPU.json
```

For running conceptual metrics on ALL datasets and ALL XAI approaches (requires training; not recommended):
```python
python processtransformer\xai\metrics\run_exp\run_xai_exp_helper.py
```

For running attention-metrics on ALL datasets (requires training; not recommended):
```python
python processtransformer\xai\metrics\run_exp\run_nn_exp_helper.py
```

## Conda
### Initial Conda Setup
Creating and activating the Conda environment (within the root-directory):

`conda env create -f env.yaml`

`conda activate process-transformer`


### Within an IDE
Make sure to select the environment!
In PyCharm you can select the Conda environment in the run configuration.


### Installing Packages
Installing packages within an active environment:

`conda install <package-name>`

If the environment is not active, you can use:

`conda install <package-name> -n process-transformer`


### Updating Conda Environment
Updating means, that you updated the env.yaml file yourself!
Updating the conda environment (within the root-directory, not within an environment):

`conda env update --name process-transformer --file env.yaml --prune`

Afterward, execute (same as in initial step):

`conda activate process-transformer`

Additionally, in your IDE you may have to remove the old environment and add the new one.


### Exporting Conda Environment
https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment
1. Activate the environment
2. `conda env export > env.yaml`


### Listing environments
`conda env list`


### Removing Conda Environment
`conda remove -n process-transformer --all`


## Where to find what
Most important source code is in [processtransformer/xai](processtransformer/xai).
Files are linked (click on the names to get to the files/directories).

- On the top level, the different XAI approaches reside
    - [TraceModificationExplainer](processtransformer/xai/trace_modification_explainer.py)
    - [TraceBackwardExplainer](processtransformer/xai/trace_backward_explainer.py)
    - [AttentionExplorationExplainer](processtransformer/xai/attn_exploration_explainer.py)
        - Its evaluators are in [event_eval](processtransformer/xai/event_eval)
- In the [visualization](processtransformer/xai/visualization) directory, the different explanation outputs can be found:
    - For example, the rules are defined in [relations_output.py](processtransformer/xai/visualization/output_models/relations_output.py)
    - Graph outputs etc. are in the [output_models](processtransformer/xai/visualization/output_models) directory as well
    - The functions to visualize/print the output are in [viz_funcs](processtransformer/xai/visualization/viz_funcs)
- The [metrics](processtransformer/xai/metrics) are divided into:
    - Running the metrics:
        - Run all conceptual-metrics for a certain dataset and XAI approach [run_through_xai_metrics.py](processtransformer/xai/metrics/run_through_xai_metrics.py)
        - Run all attention-metrics for a certain dataset [run_through_nn_metrics.py](processtransformer/xai/metrics/run_through_nn_metrics.py)
        - Run all conceptual-metrics for ALL datasets and ALL XAI approaches [run_xai_exp_helper.py](processtransformer/xai/metrics/run_exp/run_xai_exp_helper.py)
        - Run all attention-metrics for ALL datasets [run_nn_exp_helper.py](processtransformer/xai/metrics/run_exp/run_nn_exp_helper.py)
    - Metric implementations:
        - Jaccard coefficient, JSD, and TVD are in [common.py](processtransformer/xai/metrics/common.py)
        - All environment generation scripts are in [trace_generation.py](processtransformer/xai/metrics/trace_generation.py)
        - Conceptual metrics:
            - Co01 [correctness_co01.py](processtransformer/xai/metrics/correctness_co01.py)
            - Co02 [completeness_co02.py](processtransformer/xai/metrics/completeness_co02.py)
            - Co03 [consistency_co03.py](processtransformer/xai/metrics/consistency_co03.py)
            - Co04 [continuity_co04.py](processtransformer/xai/metrics/continuity_co04.py)
            - Co05 [contrastivity_co05.py](processtransformer/xai/metrics/contrastivity_co05.py)
            - Co07 [compactness_co07.py](processtransformer/xai/metrics/compactness_co07.py)
        - Attention metrics:
            - Correlation of feature importance [attn_feature_importance.py](processtransformer/xai/metrics/attn_feature_importance.py)
            - Attention Mechanism Parameter Manipulation [attn_uniform_weights.py](processtransformer/xai/metrics/attn_uniform_weights.py)
            - Attention Score Masking [attn_masking.py](processtransformer/xai/metrics/attn_masking.py)
