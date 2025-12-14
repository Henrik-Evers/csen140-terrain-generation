import torch
import gc

from train import train

experiments = [
    { # Baseline, use defaults
        'name': 'exp1_baseline',
    },
    {
        'name': 'exp2_high_annealed_noise',
        'start_noise': 0.15,
    },
    {
        'name': 'exp3_no_noise',
        'start_noise': 0.0,
        'noise': 0.0,
    },
    {
        'name': 'exp4_no_onehot_softening',
        'pull': 0.0,
        'noise': 0.0,
    },
    {
        'name': 'exp5_moderate_softening',
        'pull': 0.05,
        'noise': 0.05,
    },
    {
        'name': 'exp6_stronger_softening',
        'pull': 0.1,
        'noise': 0.1,
    },
    {
        'name': 'exp7_kernel_size_5',
        'kernel_size': 5,
    },
    {
        'name': 'exp8_faster_generator',
        'lr_g': 0.0003,
    },
    {
        'name': 'exp9_middle_generator',
        'lr_g': 0.00025,
    },
    {
        'name': 'exp10_faster_both',
        'lr_g': 0.0003,
        'lr_d': 0.000225,
        'epochs': 16,
        'batches_per_epoch': 512,
    },
    {
        'name': 'exp11_slower_both',
        'lr_g': 0.0001,
        'lr_d': 0.000075,
    },
]

if __name__ == "__main__":
    for i, config in enumerate(experiments):
        try:
            # Garbage-collect to free memory and prevent leakage
            torch.cuda.empty_cache()
            gc.collect()

            # Train with these parameters
            train(config)

        except Exception as e:
            # If one fails, I would like the others to still be executed
            print(f'Failure in {config['name']}')
            print(e)
            continue
