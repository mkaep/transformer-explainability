
from processtransformer.xai.metrics.run_through_nn_metrics import main_with_args


def main():
    configs = [
        # r'experiments/training/exp_bpic12_GPU.json',
        # r'experiments/training/exp_bpic12_w_GPU.json',
        # r'experiments/training/exp_bpic12_wc_GPU.json',
        # r'experiments/training/exp_bpic12_o_GPU.json',
        # r'experiments/training/exp_bpic13_incidents_GPU.json',
        # r'experiments/training/exp_bpic13_closed_problems_GPU.json',
        # r'experiments/training/exp_helpdesk_GPU.json',
        # r'experiments/training/exp_sepsis_GPU.json',
        r'experiments/training/exp_sequence.json',
        r'experiments/training/exp_long_dist_dep.json',
        r'experiments/training/exp_looped_and.json',
        r'experiments/training/exp_complex_model.json',
    ]

    for config in configs:
        main_with_args(config)


if __name__ == '__main__':
    main()
