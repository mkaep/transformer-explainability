

from processtransformer.xai.event_eval.evaluator import Evaluator, BasePredictionInfo, LocalScoreDict


class PositionEvaluator(Evaluator):
    @staticmethod
    def get_name():
        return __class__.__name__

    def eval(self, info: BasePredictionInfo, local_score_dict: LocalScoreDict):
        # Vary position of events
        for i, attn_index in enumerate(info.relevant_attn_indices):
            len_pre = len(info.trace)
            event = info.trace[attn_index]
            bool_list = [True if j == i else False for j in range(len(info.relevant_attn_indices))]

            masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
                self.build_masked_indices(bool_list, info.trace, info.relevant_attn_indices,
                                          mask_true_values=False)
            for pos in range(len_pre):
                # Mask everything except event
                masked_trace = ['M-'] * pos + [event] + ['M-'] * (len_pre - pos)
                self.evaluate_event_influence(info.trace, info.attn, local_score_dict,
                                              masked_events,
                                              masked_trace, non_masked_events, info.relevant_predictions)
