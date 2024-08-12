from processtransformer.xai.metrics.run_through_xai_metrics import main_with_args


def main():
    BPIC12_configs = [
        r'experiments\explaining\bpic12\bpic12-attn_expl_xai.json',
        r'experiments\explaining\bpic12\bpic12-attn_restr_dfg_xai.json',
        r'experiments\explaining\bpic12\bpic12-trace_backward_xai.json',
        r'experiments\explaining\bpic12\bpic12-trace_mask_variety_xai.json',
        r'experiments\explaining\bpic12\bpic12-trace_mask_walk_xai.json',
        r'experiments\explaining\bpic12\bpic12-trace_mask_xai.json',
        r'experiments\explaining\bpic12\bpic12-trace_variety_xai.json',
    ]
    BPIC12_w_configs = [
        r'experiments\explaining\bpic12_w\bpic12_w-attn_expl_xai.json',
        r'experiments\explaining\bpic12_w\bpic12_w-attn_restr_dfg_xai.json',
        r'experiments\explaining\bpic12_w\bpic12_w-trace_backward_xai.json',
        r'experiments\explaining\bpic12_w\bpic12_w-trace_mask_variety_xai.json',
        r'experiments\explaining\bpic12_w\bpic12_w-trace_mask_walk_xai.json',
        r'experiments\explaining\bpic12_w\bpic12_w-trace_mask_xai.json',
        r'experiments\explaining\bpic12_w\bpic12_w-trace_variety_xai.json',
    ]
    BPIC12_wc_configs = [
        r'experiments\explaining\bpic12_wc\bpic12_wc-attn_expl_xai.json',
        r'experiments\explaining\bpic12_wc\bpic12_wc-attn_restr_dfg_xai.json',
        r'experiments\explaining\bpic12_wc\bpic12_wc-trace_backward_xai.json',
        r'experiments\explaining\bpic12_wc\bpic12_wc-trace_mask_variety_xai.json',
        r'experiments\explaining\bpic12_wc\bpic12_wc-trace_mask_walk_xai.json',
        r'experiments\explaining\bpic12_wc\bpic12_wc-trace_mask_xai.json',
        r'experiments\explaining\bpic12_wc\bpic12_wc-trace_variety_xai.json',
    ]

    BPIC12_o_configs = [
        r'experiments\explaining\bpic12_o\bpic12_o-attn_expl_xai.json',
        r'experiments\explaining\bpic12_o\bpic12_o-attn_restr_dfg_xai.json',
        r'experiments\explaining\bpic12_o\bpic12_o-trace_backward_xai.json',
        r'experiments\explaining\bpic12_o\bpic12_o-trace_mask_variety_xai.json',
        r'experiments\explaining\bpic12_o\bpic12_o-trace_mask_walk_xai.json',
        r'experiments\explaining\bpic12_o\bpic12_o-trace_mask_xai.json',
        r'experiments\explaining\bpic12_o\bpic12_o-trace_variety_xai.json',
    ]

    BPIC13_i_configs = [
        r'experiments\explaining\bpic13_incidents\bpic13_incidents-attn_expl_xai.json',
        r'experiments\explaining\bpic13_incidents\bpic13_incidents-attn_restr_dfg_xai.json',
        r'experiments\explaining\bpic13_incidents\bpic13_incidents-trace_backward_xai.json',
        r'experiments\explaining\bpic13_incidents\bpic13_incidents-trace_mask_variety_xai.json',
        r'experiments\explaining\bpic13_incidents\bpic13_incidents-trace_mask_walk_xai.json',
        r'experiments\explaining\bpic13_incidents\bpic13_incidents-trace_mask_xai.json',
        r'experiments\explaining\bpic13_incidents\bpic13_incidents-trace_variety_xai.json',
    ]
    BPIC13_cp_configs = [
        r'experiments\explaining\bpic13_closed_problems\bpic13_closed_problems-attn_expl_xai.json',
        r'experiments\explaining\bpic13_closed_problems\bpic13_closed_problems-attn_restr_dfg_xai.json',
        r'experiments\explaining\bpic13_closed_problems\bpic13_closed_problems-trace_backward_xai.json',
        r'experiments\explaining\bpic13_closed_problems\bpic13_closed_problems-trace_mask_variety_xai.json',
        r'experiments\explaining\bpic13_closed_problems\bpic13_closed_problems-trace_mask_walk_xai.json',
        r'experiments\explaining\bpic13_closed_problems\bpic13_closed_problems-trace_mask_xai.json',
        r'experiments\explaining\bpic13_closed_problems\bpic13_closed_problems-trace_variety_xai.json',
    ]

    sepsis_configs = [
        r'experiments\explaining\sepsis\sepsis-attn_expl_xai.json',
        r'experiments\explaining\sepsis\sepsis-attn_restr_dfg_xai.json',
        r'experiments\explaining\sepsis\sepsis-trace_backward_xai.json',
        r'experiments\explaining\sepsis\sepsis-trace_mask_variety_xai.json',
        r'experiments\explaining\sepsis\sepsis-trace_mask_walk_xai.json',
        r'experiments\explaining\sepsis\sepsis-trace_mask_xai.json',
        r'experiments\explaining\sepsis\sepsis-trace_variety_xai.json',
    ]

    helpdesk_configs = [
        r'experiments\explaining\helpdesk\helpdesk-attn_expl_xai.json',
        r'experiments\explaining\helpdesk\helpdesk-attn_restr_dfg_xai.json',
        r'experiments\explaining\helpdesk\helpdesk-trace_mask_xai.json',
        r'experiments\explaining\helpdesk\helpdesk-trace_backward_xai.json',
        r'experiments\explaining\helpdesk\helpdesk-trace_mask_variety_xai.json',
        r'experiments\explaining\helpdesk\helpdesk-trace_mask_walk_xai.json',
        r'experiments\explaining\helpdesk\helpdesk-trace_variety_xai.json',
    ]

    complex_model_configs = [
        r'experiments\explaining\14_complex_models\complex_model_001\complex_model_001-attn_expl_xai.json',
        r'experiments\explaining\14_complex_models\complex_model_001\complex_model_001-attn_restr_dfg_xai.json',
        r'experiments\explaining\14_complex_models\complex_model_001\complex_model_001-trace_backward_xai.json',
        r'experiments\explaining\14_complex_models\complex_model_001\complex_model_001-trace_mask_variety_xai.json',
        r'experiments\explaining\14_complex_models\complex_model_001\complex_model_001-trace_mask_walk_xai.json',
        r'experiments\explaining\14_complex_models\complex_model_001\complex_model_001-trace_mask_xai.json',
        r'experiments\explaining\14_complex_models\complex_model_001\complex_model_001-trace_variety_xai.json',
    ]

    looped_and_configs = [
        r'experiments\explaining\15_looped_AND\looped_and-attn_expl_xai.json',
        r'experiments\explaining\15_looped_AND\looped_and-attn_restr_dfg_xai.json',
        r'experiments\explaining\15_looped_AND\looped_and-trace_backward_xai.json',
        r'experiments\explaining\15_looped_AND\looped_and-trace_mask_variety_xai.json',
        r'experiments\explaining\15_looped_AND\looped_and-trace_mask_walk_xai.json',
        r'experiments\explaining\15_looped_AND\looped_and-trace_mask_xai.json',
        r'experiments\explaining\15_looped_AND\looped_and-trace_variety_xai.json',
    ]

    sequence_configs = [
        r'experiments/explaining/sequence/sequence-attn_expl_xai.json',
        r'experiments/explaining/sequence/sequence-attn_restr_dfg_xai.json',
        r'experiments/explaining/sequence/sequence-trace_backward_xai.json',
        r'experiments/explaining/sequence/sequence-trace_mask_variety_xai.json',
        r'experiments/explaining/sequence/sequence-trace_mask_walk_xai.json',
        r'experiments/explaining/sequence/sequence-trace_mask_xai.json',
        r'experiments/explaining/sequence/sequence-trace_variety_xai.json',
    ]

    long_dist_dep_configs = [
        r'experiments/explaining/long_running_dependency2/long_dist_dep2-attn_expl_xai.json',
        r'experiments/explaining/long_running_dependency2/long_dist_dep2-attn_restr_dfg_xai.json',
        r'experiments/explaining/long_running_dependency2/long_dist_dep2-trace_backward_xai.json',
        r'experiments/explaining/long_running_dependency2/long_dist_dep2-trace_mask_variety_xai.json',
        r'experiments/explaining/long_running_dependency2/long_dist_dep2-trace_mask_walk_xai.json',
        r'experiments/explaining/long_running_dependency2/long_dist_dep2-trace_mask_xai.json',
        r'experiments/explaining/long_running_dependency2/long_dist_dep2-trace_variety_xai.json',
    ]

    for config in long_dist_dep_configs:
        main_with_args(config)


if __name__ == '__main__':
    main()
