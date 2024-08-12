# Code source: Gaël Varoquaux
# License: BSD 3 clause
# Copied from https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html

# Note:
# If you are using PyCharm and want to rotate the 3D plot then go to
# File | Settings | Tools | Python Scientific and uncheck "Show plots in tool window".
# This will now use matplotlib's interactive viewer.

import argparse
import os.path
import typing

import matplotlib.pyplot as plt
# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
import pandas as pd
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import decomposition

import train_model
from processtransformer.util import compressor

parser = argparse.ArgumentParser(description="Misc - BPMN Traversal.")
parser.add_argument("--training_dir", type=str, required=True,
                    help='Path to the training directory (with NN-model and dicts.')
parser.add_argument("--n_dims", type=int, required=True,
                    help='Number of dimension to use for PCA. Only 2 or 3 allowed.')
args = parser.parse_args()


def plot_position_embeddings(parent_dir: str, n_components=2):
    model, x_dict = load_model(n_components, parent_dir)
    length = model.max_case_length
    all_positions, position_embedding = get_position_embedding(length, model)

    split_via_pca(all_positions, position_embedding, n_components)


def plot_token_embeddings(parent_dir: str, n_components=2):
    model, x_dict = load_model(n_components, parent_dir)

    all_activities, token_embedding = get_token_embedding(list(x_dict.keys()), model, x_dict)
    split_via_pca(all_activities, token_embedding, n_components)


def plot_trace_embeddings_via_pca(parent_dir: str, trace: typing.List[str],
                                  n_components=2, use_positions=True):
    model, x_dict = load_model(n_components, parent_dir)

    all_activities, token_embedding = get_token_embedding(trace, model, x_dict)
    all_positions, position_embedding = get_position_embedding(len(trace), model)

    embedding = token_embedding
    if use_positions:
        embedding += position_embedding

    all_items = all_activities
    # all_items = all_positions
    split_via_pca(all_items, embedding, n_components)


def plot_applied_matrix_embeddings(parent_dir: str, trace: typing.List[str],
                                   n_components=2):
    model, x_dict = load_model(n_components, parent_dir)

    all_activities, token_embedding = get_token_embedding(trace, model, x_dict)
    all_positions, position_embedding = get_position_embedding(len(trace), model)

    embedding = token_embedding + position_embedding
    # embedding = np.full_like(token_embedding, fill_value=1.0)
    # np.random.seed(42)
    mean, std = stats_about_embeddings((model, x_dict))
    embedding = np.random.normal(mean, std, embedding.shape)
    # plot_vector_or_matrix(embedding)

    attention_layer = model.transformer_block.att
    functions = [attention_layer._query_dense,
                 attention_layer._key_dense,
                 attention_layer._value_dense]
    applied = [fun(embedding[tf.newaxis, :, :]) for fun in functions]
    applied = [tf.squeeze(tensor, axis=0) for tensor in applied]  # only have one "batch" -> remove this dimension

    qkv_title = ['Query', 'Key', 'Value']
    plot_key_query_value_matrices(applied, qkv_title, 'Applied Matrices to Embeddings')

    raw_matrices = [attention_layer._query_dense.kernel,
                    attention_layer._key_dense.kernel,
                    attention_layer._value_dense.kernel]
    plot_key_query_value_matrices(raw_matrices, qkv_title, 'Raw Matrices')

    # Need new axis so we have three dimensions again
    raw_biases = [attention_layer._query_dense.bias[tf.newaxis, :, :],
                  attention_layer._key_dense.bias[tf.newaxis, :, :],
                  attention_layer._value_dense.bias[tf.newaxis, :, :]]
    plot_key_query_value_matrices(raw_biases, qkv_title, 'Raw Bias-Vectors')

    query, key, value = [a[tf.newaxis, :, :, :] for a in applied]
    attention_output, attention_scores = attention_layer._compute_attention(query, key, value)
    attention_output = attention_layer._output_dense(attention_output)

    plot_vector_or_matrix(attention_output)
    attention_scores = tf.einsum('abcd->acbd', attention_scores)
    plot_key_query_value_matrices(attention_scores, ['-'], 'Attention-Scores')
    return attention_scores
    # fig, ax = plt.subplots()
    # reduced_output = tf.reduce_sum(attention_layer._output_dense.kernel, axis=[0, 1])
    # reduced_output = tf.reshape(reduced_output, (1, -1))
    # mat = ax.matshow(reduced_output, aspect='auto')
    # ax.axes.get_yaxis().set_visible(False)
    # fig.colorbar(mat, ax=ax)
    # plt.show()
    # output = tf.squeeze(attention_layer._output_dense(applied[2][tf.newaxis, :, :, :]), axis=0).numpy()
    # fig, ax = plt.subplots()
    # ax.matshow(output)
    # plt.show()


def plot_vector_or_matrix(vec_or_mat: tf.Tensor, show_color_bar=True,
                          aspect='auto'):
    if len(vec_or_mat.shape) == 1:
        vec_or_mat = tf.reshape(vec_or_mat, shape=(1, -1))
    if len(vec_or_mat.shape) > 2:
        # Try to squeeze out dimensions with 1 first
        vec_or_mat = tf.squeeze(vec_or_mat)
        if len(vec_or_mat.shape) > 2:
            raise ValueError("Shape must be 1 or 2-dimensional!")

    fig, ax = plt.subplots()
    im = ax.imshow(vec_or_mat, aspect=aspect)

    if show_color_bar:
        shrink_factor = len(vec_or_mat) / len(vec_or_mat[0])
        fig.colorbar(im, ax=ax, shrink=shrink_factor)

    ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_key_query_value_matrices(query_key_value, sub_figure_titles, title):
    num_heads = len(query_key_value[0][0])
    # For layout see: https://stackoverflow.com/a/68209152/9523044
    fig, axes = plt.subplots(nrows=len(query_key_value), ncols=1, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)  # needed in case we only have one row
    for ax in axes:
        ax.remove()
    # add subfigure per subplot
    gridspec = axes[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(sub_figure_titles[row])
        tensor = query_key_value[row]
        vmin = np.min(tensor)
        vmax = np.max(tensor)

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=num_heads)
        for col, ax in enumerate(axs):
            aspect = 'equal'  # use 'auto' to scale (or float)
            mat = ax.imshow(tensor[:, col, :], vmin=vmin, vmax=vmax,
                            aspect=aspect)
            ax.set_axis_off()
            ax.set_title(f'Head {col + 1}')

        fig.colorbar(mat, ax=ax, pad=0.05)
    fig.suptitle(title)
    plt.show()


def get_position_embedding(length, model):
    all_positions = [str(i) for i in range(1, length + 1)]
    positions = tf.range(start=0, limit=length, delta=1)
    position_embedding = model.token_and_pos_emb.pos_emb(positions)
    return all_positions, position_embedding


def get_token_embedding(activity_list, model, x_dict):
    all_activities = activity_list
    all_tokens = [x_dict[key] for key in all_activities]
    token_embedding = model.token_and_pos_emb.token_emb(np.asarray(all_tokens).reshape(1, -1))
    token_embedding = token_embedding.numpy().squeeze(axis=0)
    return all_activities, token_embedding


def load_model(n_components, parent_dir):
    # np.random.seed(5) TODO make flag and stuff
    assert n_components == 2 or n_components == 3, "Dimension must be 2 or 3!"
    dict_dir = os.path.join(parent_dir, "dicts")
    model_dir = os.path.join(parent_dir, "model")

    model = train_model.TrainNextActivityModel.load_model(model_dir)
    x_dict = compressor.decompress(os.path.join(dict_dir, "x_word_dict"))
    # y_dict = compressor.decompress(os.path.join(dict_dir, "y_word_dict"))
    return model, x_dict


def split_via_pca(all_items, embedding, n_components):
    X = embedding
    y = np.asarray(all_items)
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    else:
        ax = fig.add_subplot(111)
    ax.set_position([0, 0, 0.95, 1])
    plt.cla()
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    X = pca.transform(X)

    for name in all_items:
        bbox = dict(alpha=0.5, edgecolor="w", facecolor="w")
        ha = "center"
        x_coord = X[y == name, 0].mean()
        y_coord = X[y == name, 1].mean()
        if n_components == 3:
            ax.text3D(
                x_coord,
                y_coord,
                X[y == name, 2].mean(),
                name,
                horizontalalignment=ha,
                bbox=bbox,
            )
        else:
            ax.text(
                x_coord,
                y_coord,
                name,
                horizontalalignment=ha,
                bbox=bbox,
            )
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(y, [1, 2, 0]).astype(float)
    if n_components == 3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.nipy_spectral, edgecolor="k")
    else:
        ax.scatter(X[:, 0], X[:, 1], cmap=plt.cm.nipy_spectral, edgecolor="k")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    if n_components == 3:
        ax.zaxis.set_ticklabels([])
    plt.show()

    component_df = pd.DataFrame(pca.components_, columns=range(len(embedding[0])), index=['PC-1', 'PC-2'])
    print(component_df)

    fig, ax = plt.subplots()
    mat = ax.matshow(np.abs(component_df), cmap='Blues')
    fig.colorbar(mat, location='bottom')
    plt.show()


def stats_about_embeddings(model_source: typing.Union[str, typing.Tuple[tf.keras.Model, typing.Dict]]):
    if isinstance(model_source, str):
        model, x_dict = load_model(2, model_source)
    else:
        model, x_dict = model_source
    all_tokens = [x_dict[key] for key in x_dict.keys()]
    max_case_length = model.max_case_length.numpy()
    all_emb = np.zeros((len(all_tokens) + max_case_length,
                        model.token_and_pos_emb.token_emb.output_dim))

    # Get token embeddings
    for token, index in enumerate(all_tokens):
        emb = model.token_and_pos_emb.token_emb(token).numpy().reshape(-1)
        all_emb[index] = emb

    # Get position embeddings
    for position in range(max_case_length):
        emb = model.token_and_pos_emb.pos_emb(position).numpy().reshape(-1)
        all_emb[len(all_tokens) + position] = emb

    mean = np.mean(all_emb)
    std = np.std(all_emb)
    print(f'Embedding-Stats. Mean: {mean:0.3e}, Std-dev: {std:0.3e}.')
    # plt.hist(all_emb.flatten(), bins=30)
    # plt.show()
    return mean, std


def average_random_attention_scores(parent_dir: str, runs=10, trace_len=10):
    model, x_dict = load_model(2, parent_dir)
    mean, std = stats_about_embeddings((model, x_dict))

    # Sizes
    embedding_size = model.token_and_pos_emb.token_emb.output_dim
    max_case_len = model.max_case_length.numpy()
    pad = max_case_len - trace_len

    # Mask
    mask = np.zeros((max_case_len, max_case_len), dtype=np.int32)
    ones = np.ones((trace_len, trace_len), dtype=np.int32)
    mask[pad:, pad:] = ones
    mask = mask[np.newaxis, np.newaxis, :, :]

    all_attention_scores = np.zeros((runs, model.transformer_block.att._num_heads,
                                     trace_len, trace_len))
    all_outputs = np.zeros((runs, max_case_len, embedding_size))

    # We cannot run batches through the NN as we only keep the last attention-scores
    for run in range(runs):
        emb = np.zeros((1, max_case_len, embedding_size))
        rnd = np.random.normal(mean, std, size=(1, trace_len, embedding_size))
        emb[:, pad:, :] = rnd  # acts as a mask
        # we are not interested in the actual output, just the attention-scores
        output = model.transformer_block(emb, training=False, mask=mask)
        all_outputs[run] = tf.squeeze(output)
        attn_scores = tf.squeeze(model.transformer_block.last_attn_scores, axis=0)
        all_attention_scores[run] = attn_scores[:, pad:, pad:]

    reduced_attn = all_attention_scores.sum(axis=0, keepdims=True)
    reduced_attn = np.einsum('abcd->acbd', reduced_attn)  # head must be the third dimension
    plot_key_query_value_matrices(reduced_attn, [''], f'Attention-Scores over {runs} runs')

    reduced_output = np.abs(all_outputs.sum(axis=0))
    plot_vector_or_matrix(reduced_output, aspect='equal')


if __name__ == "__main__":
    # plot_matrices(args.training_dir)
    # stats_about_embeddings(args.training_dir)

    # average_random_attention_scores(args.training_dir, runs=1000)

    plot_applied_matrix_embeddings(args.training_dir,
                                   ['A', 'I', 'J', 'K', 'U', 'V', 'W'],
                                   args.n_dims,
                                   )

    # plot_trace_embeddings_via_pca(args.training_dir,
    #                               ['Activity A', 'Activity B', 'Activity C', 'Activity D',
    #                                'Activity E', 'Activity F', 'Activity G', 'Activity H'],
    #                               args.n_dims, use_positions=False)
