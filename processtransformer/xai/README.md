

# Info on XAI

Classification into:
- Space
  - global (explain multiple traces, i.e. globally)
  - local (explain single trace, i.e. locally)
- Output
  - Visual, i.e. plots
  - Textual
  - MatrixOutput (adjacency matrix)
  - GraphOutput
  - RelationsOutput (dictionary which relates events with their predictions). E.g. A -> B, C

## AttentionExplorationExplainer
See [attn_exploration_explainer.py](attn_exploration_explainer.py).

Workings:
- In one sentence: Modify each trace to check whether predictions stay the same and if so,
  the remaining events in the trace must relate to these predictions.
- Iterate over traces
- For each trace
  - Apply evaluator(s)
  - Each evaluator modifies the trace somehow to derive relations between the trace and predictions
  - Each evaluator has its own relations
- Relations are collected over the various traces
- And filtered at the end; hopefully keeping only relevant ones
- All relations may be combined in any way and added as output. E.g. logical AND between all adjacency matrices

Classification:
- global
- MatrixOutput

Notes
- Does not consider that certain sub-sequences may result in different possible predictions.

## AttentionRestrictedDirectlyFollowsGraph
See [attn_restr_DFG.py](attn_restr_DFG.py).

Workings:
- In one sentence: Constructs a directly-follows-graphs but only draws edges if this is supported by attention-scores.
- Has several so-called "checks" that permute the trace to check whether certain events are of relevance
- Each combination of checks is used (inactive and active)
- Then all traces are iterated
- For each trace the activated checks are applied, mostly to the last event in the trace
  - If it is relevant for the prediction, an edge is drawn

Classification
- global
- GraphOutput and TextOutput

Notes
- The major drawback is that some edges that would be drawn in a process-graph cannot be inserted by this algorithm.
- E.g. long distance dependencies: an edge can never be drawn from the first to the last event

## TraceModificationExplainer
See [trace_modification_explainer.py](trace_modification_explainer.py).

Base-class for several other explainers.

Workings:
- In one sentence: Locally relates attention-scores (per column) to predictions. The sums this up to be global.
- Input is a trace.
- New traces are generated via "trace-series-gen" (see subclasses).
  - Each of these traces is then modified via "trace-mod-gen" (see subclasses). 
    - Constructs a graph for each modified trace.
- Joins together all the graphs afterward

Classification
- local
- GraphOutput

Notes
- Limited as the global construction is done by simply summing up local ones
- Subclasses mainly override the trace-series-gen and trace-mod-gen.

### TraceBackwardExplainer
See [trace_backward_explainer.py](trace_backward_explainer.py).

Inherits from [TraceModificationExplainer](#tracemodificationexplainer).

Notes
- global; can be used locally
- trace-series-gen: Basically only generates a subset of prefixes by looking at relevant events.
- trace-mod-gen: mask different combinations of events. 

### TraceVarietyExplainer
See [trace_variety_explainer.py](trace_variety_explainer.py).

Inherits from [TraceModificationExplainer](#tracemodificationexplainer).

Notes
- trace-series-gen: prefixes.
- trace-mod-gen: randomly generate new traces from each trace, using its events.

### TraceMaskExplainer
See [trace_mask_explainer.py](trace_mask_explainer.py).

Inherits from [TraceModificationExplainer](#tracemodificationexplainer).

- trace-series-gen: prefixes.
- trace-mod-gen: mask different combinations of events. 
  - Traces are modified by producing several traces with one or more masked events.
    E.g. `A, B, C, D` produces `M-A, B, M-C, D` or `A, B, C, M-D`. `M-` = masked.

### TraceMaskWalkExplainer
See [trace_mask_walk_explainer.py](trace_mask_walk_explainer.py).

Inherits from [TraceMaskExplainer](#tracemaskexplainer)

- trace-series-gen: prefixes.
- trace-mod-gen: mask one event after the other.

### TraceMaskVarietyExplainer
See [trace_mask_variety_explainer.py](trace_mask_variety_explainer.py).

Inherits from [TraceMaskExplainer](#tracemaskexplainer).

- trace-series-gen: Basically only generates a subset of prefixes by looking at relevant events.
- trace-mod-gen: mask different combinations of events.

## VisualAttentionMatrixExplainer
See [visual_attn_mtrx_xai.py](visual_attn_mtrx_xai.py).

Only outputs the attention-scores within the NN for a single trace.

Classification:
- local
- VisualOutput
