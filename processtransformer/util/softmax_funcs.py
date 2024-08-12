
from processtransformer.util.types import SoftmaxVector, SimpleSoftmax


def filter_softmax_vector(softmax_vector: SoftmaxVector,
                          threshold: float = 0.1,
                          ) -> SoftmaxVector:
    """threshold should be in range [0.0, ..., 1.0] but this is not enforced"""
    return [(ev, val) for ev, val in softmax_vector if val >= threshold]


def softmax_vector_to_predictions(softmax_vector: SoftmaxVector,
                                  ) -> SimpleSoftmax:
    return [event for event, val in softmax_vector]
