import subprocess
import numpy as np
import os
import argparse

def clean_semaphores():
    shm_path = '/dev/shm'
    for filename in os.listdir(shm_path):
        file_path = os.path.join(shm_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def run_all():
    #Clean out semaphores
    clean_semaphores()

    field_list = np.arange(245, 1876)
    algorithm = 'fpw'
    log_file = '/data/zvar/logs/field_log.txt'
    
    for field in field_list:
        field_folder = f'/data/zvar/zvar_results/{str(field).zfill(4)}'
        
        # Check if the folder exists. If not, create it
        if not os.path.exists(field_folder):
            os.makedirs(field_folder, exist_ok=True)
        
        subprocess.run(['python3', 'runField.py', str(field), algorithm])
        with open(log_file, 'a') as log:
            log.write(f'Completed field: {str(field)}\n')

def run_individual(field):
    #Clean out semaphores
    clean_semaphores()

    field_folder = f'/data/zvar/zvar_results/{str(field).zfill(4)}'
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(field_folder):
        os.makedirs(field_folder, exist_ok=True)
    
    # Delete all files in the field's folder
    for filename in os.listdir(field_folder):
        file_path = os.path.join(field_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
    # Run the field
    algorithm = 'fpw'
    subprocess.run(['python3', 'runField.py', str(field), algorithm])
    
    # Log the completion
    log_file = '/data/zvar/logs/field_log.txt'
    with open(log_file, 'a') as log:
        log.write(f'Completed field: {str(field)}\n')

def resume_computation():
    #Clean out semaphores
    clean_semaphores()
    
    field_list = np.arange(245, 1876)
    log_file = '/data/zvar/logs/field_log.txt'
    
    # Read the log file to find the last completed field
    last_completed_field = None
    if os.path.exists(log_file):
        with open(log_file, 'r') as log:
            lines = log.readlines()
            if lines:
                last_completed_field = int(lines[-1].strip().split()[-1])
    
    # Determine the starting point
    start_index = 0
    if last_completed_field is not None:
        start_index = np.where(field_list == last_completed_field)[0][0] + 1
    
    # Resume computation from the next field
    algorithm = 'fpw'
    for field in field_list[start_index:]:
        field_folder = f'/data/zvar/zvar_results/{str(field).zfill(4)}'
        
        # Check if the folder exists, if not, create it
        if not os.path.exists(field_folder):
            os.makedirs(field_folder, exist_ok=True)
        
        subprocess.run(['python3', 'runField.py', str(field), algorithm])
        with open(log_file, 'a') as log:
            log.write(f'Completed field: {str(field)}\n')


def run_queued():
    #Clean out semaphores
    clean_semaphores()
    
    field_list = np.loadtxt('/data/zvar/logs/manual_queue.txt', dtype=int)
    log_file = '/data/zvar/logs/field_log.txt'
    
    # Read the log file to find the last completed field
    last_completed_field = None
    if os.path.exists(log_file):
        with open(log_file, 'r') as log:
            lines = log.readlines()
            if lines:
                last_completed_field = int(lines[-1].strip().split()[-1])
    
    # Resume computation from the next field
    algorithm = 'fpw'
    for field in field_list:
        field_folder = f'/data/zvar/zvar_results/{str(field).zfill(4)}'
        
        # Check if the folder exists, if not, create it
        if not os.path.exists(field_folder):
            os.makedirs(field_folder, exist_ok=True)
        
        subprocess.run(['python3', 'runField.py', str(field), algorithm])
        with open(log_file, 'a') as log:
            log.write(f'Completed field: {str(field)}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run field processing functions.')
    parser.add_argument('function', choices=['run_all', 'run_individual', 'resume_computation', 'run_queued'], help='Function to run')
    parser.add_argument('--field', type=int, help='Field number (required for run_individual)')

    args = parser.parse_args()

    if args.function == 'run_all':
        run_all()
    elif args.function == 'run_individual':
        if args.field is None:
            parser.error('The --field argument is required for run_individual')
        run_individual(args.field)
    elif args.function == 'resume_computation':
        resume_computation()
    elif args.function == 'run_queued':
        run_queued()