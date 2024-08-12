

import tkinter as tk
import typing
from tkinter import ttk

from processtransformer.data_models.explaining_model import ExplainingModel
from processtransformer.gui.custom_elements.fields import ReadOnlyField, FloatField
from processtransformer.gui.custom_elements.separator import Separator
from processtransformer.gui.main_tabs.explainer_tabs.evaluator_tabs.evaluator_tab import EvaluatorTab
from processtransformer.gui.main_tabs.explainer_tabs.explainer_tab import ExplainerTab
from processtransformer.xai.attn_exploration_explainer import AttentionExplorationExplainer
from processtransformer.xai.event_eval import Evaluator


class AttentionExplorationExplainerTab(ExplainerTab):
    def __init__(self, notebook, explainer: AttentionExplorationExplainer, resize_grid_callback, **kwargs):
        super().__init__(notebook, explainer, resize_grid_callback, **kwargs)

    def _layout_child_inputs(self):
        ReadOnlyField(self, self.num_rows, 'Explainer-specific parameters', '')
        self.num_rows += 1

        self.normal_to_abs_thr_field = FloatField(self, self.num_rows,
                                                  'Normal- to absolute-sum threshold', 0.0, 1.0, 0.6)
        self.num_rows += 1

        self.dependency_threshold_field = FloatField(self, self.num_rows,
                                                     'Dependency threshold', 0.0, 1.0, 0.1)
        self.num_rows += 1

        self.attention_score_threshold_field = FloatField(self, self.num_rows,
                                                          'Attention-score threshold', 0.0, 1.0, 0.15)
        self.num_rows += 1

        self.prediction_threshold_field = FloatField(self, self.num_rows,
                                                     'Prediction threshold', 0.0, 1.0, 0.1)
        self.num_rows += 1

        self.eval_notebook = ttk.Notebook(self)
        self.eval_notebook.grid(row=self.num_rows, column=0, columnspan=self.columns, sticky=tk.EW)
        self.tabs: typing.List[EvaluatorTab] = []
        # noinspection PyUnresolvedReferences
        import processtransformer.xai.event_eval
        evaluators = Evaluator.__subclasses__()
        for ev in evaluators:
            tab = EvaluatorTab(self.eval_notebook, ev)
            self.eval_notebook.add(tab, text=ev.get_name())
            self.tabs.append(tab)
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

    @classmethod
    def get_explainer_cls(cls):
        return AttentionExplorationExplainer

    def _core_on_explaining_button_click(self, xai_model: ExplainingModel):
        # Explainer kwargs
        kwargs = {
            'normal_to_abs_thr': self.normal_to_abs_thr_field.get(),
            'dependency_threshold': self.dependency_threshold_field.get(),
            'attention_score_threshold': self.attention_score_threshold_field.get(),
            'prediction_threshold': self.prediction_threshold_field.get(),
        }

        # Pick up evaluators
        eval_list = []
        for tab in self.tabs:
            eval_cls = tab.get_evaluator_cls()
            kw = tab.get_evaluator_kwargs()
            if type(kw) == bool and kw is False:
                # Evaluator is disabled
                continue
            eval_list.append((kw, eval_cls))

        kwargs['evaluators'] = eval_list
        xai_model.explainer_kwargs = kwargs
