import os
import sys
from termcolor import colored

def create_experiment_dirs(experiment_name):
    main_dir = os.path.dirname(os.getcwd())
    experiment_dir = os.path.join(main_dir, 'experiments', experiment_name)
    raw_img_dir = os.path.join(experiment_dir, 'raw_images')
    denoised_img_dir = os.path.join(experiment_dir, 'denoised_images')
    masks_dir = os.path.join(experiment_dir, 'masks')
    models_dir = os.path.join(experiment_dir, 'models')
    output_dir = os.path.join(experiment_dir, 'output')
    manually_corrected_masks_dir = os.path.join(experiment_dir, 'manually_corrected_masks')

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(raw_img_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(denoised_img_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(manually_corrected_masks_dir, exist_ok=True)
    
    print(colored(f"Directory structure for experiment '{experiment_name}' created.", "green"))
    print("----------------------------------------------")
    
    
def create_segmentation_analysis_dirs(experiment_name):
    main_dir = os.path.dirname(os.getcwd())
    experiment_dir = os.path.join(main_dir, 'experiments', experiment_name)
    analysis_dir = os.path.join(experiment_dir, 'analysis')

    image_output_dir = os.path.join(analysis_dir, 'images')
    metrics_output_dir = os.path.join(analysis_dir, 'metrics')
    plots_output_dir = os.path.join(analysis_dir, 'plots')
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(metrics_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(colored(f"Analysis directory for experiment '{experiment_name}' created.", "green"))
    print("----------------------------------------------")
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_experiment.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    create_experiment_dirs(experiment_name)
    create_segmentation_analysis_dirs(experiment_name)