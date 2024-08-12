import argparse
import json
import os
import typing

import matplotlib.pyplot as plt
import pm4py

import train_model
from processtransformer.data_models.explaining_model import ExplainingModel
from processtransformer.util import compressor
from processtransformer.util.str_util import str_to_valid_filename
from processtransformer.xai.transformer_base_check import TransformerBaseCheck
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.relations_output import RelationsOutput
from processtransformer.xai.visualization.output_models.text_output import TextOutput
from processtransformer.xai.visualization.viz_funcs.base_viz import VizOutput
from processtransformer.xai.visualization.viz_funcs.helper import BaseVizSubclasses

plt.rcParams["figure.figsize"] = (5.4 * 0.9, 4)
plt.rcParams["svg.fonttype"] = 'none'
plt.rcParams["font.size"] = 11.0

parser = argparse.ArgumentParser(description="XAI via trace variation.")

parser.add_argument("--explaining_config", type=str, required=True,
                    help='Path of the configuration file (data, parameters).')
parser.add_argument("--show_pad", action="store_true",
                    help='If set, the PAD-token is visualized in the attention-matrix.')


def explain(explaining_config: ExplainingModel, args=None) -> typing.List[VizOutput]:
    # Create result dir and dump json-config
    os.makedirs(explaining_config.result_dir, exist_ok=True)
    with open(os.path.join(explaining_config.result_dir, explaining_config.name + ".json"),
              'w', encoding="utf8") as f:
        json.dump(explaining_config.to_dict(), f, indent=2)

    model, x_dict, y_dict = load_model_and_dicts(explaining_config)

    # Create explainer instance - may fail if non-standard arguments do not fit
    show_pad = False
    if args is not None:
        show_pad = args.show_pad
    # Hint: Subclasses may use keyword-args
    # noinspection PyArgumentList
    explainer = explaining_config.explainer(model, x_dict, y_dict, explaining_config.result_dir, show_pad,
                                            **explaining_config.explainer_kwargs)

    # Read in trace(s) - usually just one
    trace_dir = explaining_config.prefix_and_y_true_log
    trace_df = pm4py.read_xes(trace_dir)
    trace_to_explain = None
    if explaining_config.trace_to_explain is not None:
        trace_to_explain = []
        if isinstance(explaining_config.trace_to_explain, str):
            trace_to_explain_df = pm4py.read_xes(explaining_config.trace_to_explain)
            for event in trace_to_explain_df['concept:name']:
                trace_to_explain.append(event)
            trace_to_explain = trace_to_explain
        else:
            trace_to_explain = explaining_config.trace_to_explain

    trace_tuples = list()
    for group_name, df_group in trace_df.groupby(['case:concept:name']):
        # Construct trace and event to be predicted (y_true)
        event_trace = []
        for event in df_group['concept:name']:
            event_trace.append(event)
        y_true = event_trace[-1]
        event_trace = event_trace[:-1]
        trace_tuples.append((event_trace, y_true))

    assert len(trace_tuples) != 0

    if trace_to_explain is not None and explainer.get_trace_support().single_trace:
        # Make actual explanation (single)
        output = explainer.explain_trace(trace_to_explain[:-1], trace_to_explain[-1], trace_df)
    elif explainer.get_trace_support().multi_trace:
        # multiple
        s_trace = None
        if trace_to_explain is not None:
            s_trace = trace_to_explain[:-1]
        output = explainer.explain_multiple_traces(trace_tuples, trace_df, s_trace)
    else:
        return []

    base_check = TransformerBaseCheck(model, x_dict, y_dict)
    base_check_outputs = _perform_base_check(base_check, explaining_config, output, trace_to_explain)

    return map_output_to_viz(output + base_check_outputs)


def _perform_base_check(base_check: TransformerBaseCheck, explaining_config: ExplainingModel,
                        output, trace_to_explain):
    base_check_outputs = []
    base_dict = base_check.get_relations_for_trace(trace_to_explain)
    base_check_outputs.append(
        RelationsOutput(base_dict,
                        FigureData(os.path.join(explaining_config.result_dir, 'ground_truth_relations.txt'),
                                   'Ground-truth relations'),
                        'Ground-truth',
                        [trace_to_explain[0]]))
    for out in output:
        if isinstance(out, RelationsOutput):
            score = base_check.compare_with_other(trace_to_explain, out.relations_dict)
            text = f'Local Faithfulness-Score: {score:.3f} for {out.name}'
            filename = str_to_valid_filename(f'score_{out.name}') + '.txt'
            filename = os.path.join(explaining_config.result_dir, filename)
            base_check_outputs.append(TextOutput(text, filename))
    return base_check_outputs


def load_model_and_dicts(explaining_config: ExplainingModel):
    # Open model and dicts
    nn_dir = explaining_config.neural_network_model_dir
    dict_dir = explaining_config.dict_dir

    return load_model_and_dicts_via_dirs(nn_dir, dict_dir)


def load_model_and_dicts_via_dirs(nn_dir, dict_dir):
    model = train_model.TrainNextActivityModel.load_model(nn_dir)
    x_dict = compressor.decompress(os.path.join(dict_dir, "x_word_dict"))
    y_dict = compressor.decompress(os.path.join(dict_dir, "y_word_dict"))

    return model, x_dict, y_dict


def map_output_to_viz(output) -> typing.List[VizOutput]:
    viz_output = []

    output_acceptors = BaseVizSubclasses.get_all_subclasses()
    for m_out in output:
        for acceptor_cls in output_acceptors:
            if m_out.__class__ not in acceptor_cls.get_accepted_formats():
                continue
            acceptor = acceptor_cls(m_out)
            viz_output += acceptor.visualize()

    return viz_output


def main():
    # Load configuration and explain
    args = parser.parse_args()
    with open(args.explaining_config, 'r', encoding="utf8") as f:
        explaining_config = ExplainingModel.from_dict(json.load(f))
        explain(explaining_config, args)


if __name__ == "__main__":
    main()
