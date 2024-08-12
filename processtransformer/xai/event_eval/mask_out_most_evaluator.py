
from processtransformer.util.generators import binary_generator
from processtransformer.util.types import LocalScoreDict
from processtransformer.xai.event_eval.evaluator import Evaluator, BasePredictionInfo


class MaskOutMostEvaluator(Evaluator):
    @staticmethod
    def get_name() -> str:
        return __class__.__name__

    def eval(self, info: BasePredictionInfo, local_score_dict: LocalScoreDict):
        for bool_list in binary_generator(info.relevant_attn_indices, start=1):
            masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
                self.build_masked_indices(bool_list, info.trace, info.relevant_attn_indices,
                                          mask_true_values=False)

            # Mask out most of the events, only keep relevant events
            masked_trace = [event if index in non_masked_local_indices else f'M-{event}'
                            for index, event in enumerate(info.trace)]

            self.evaluate_event_influence(info.trace, info.attn, local_score_dict,
                                          masked_events,
                                          masked_trace, non_masked_events, info.relevant_predictions)
