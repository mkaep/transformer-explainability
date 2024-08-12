

from scipy.special import softmax

from processtransformer.util.types import AttentionVector


def reduce_attention_to_row(attn, remove_last=False):
    # Sum up 'batch' and all heads
    pred_attn_scores = attn.sum(axis=(0, 1))

    if remove_last:
        # Remove prediction from attention-score matrix
        pred_attn_scores = pred_attn_scores[:-1, :-1]
        # Need to "re-apply" softmax per row as we removed values
        pred_attn_scores = softmax(pred_attn_scores, axis=1)

    # Sum per row
    pred_attn_scores = pred_attn_scores.sum(axis=0)
    pred_attn_scores = pred_attn_scores / sum(pred_attn_scores)
    return pred_attn_scores


def attn_transform(prefix, attention) -> AttentionVector:
    return list(zip(prefix, reduce_attention_to_row(attention)))
