#Delegates the running of every file in a directory.
import subprocess
import threading
import queue
import sys
import time
import os
import glob
from multiprocessing import Semaphore

n_processes = int(os.environ.get('N_PROCESSES', 1))
n_gpus = int(os.environ.get('N_GPUS', 1))
# we want to assign up to 2 files to each GPU, so we make sure that the number of threads is at most 2 * n_gpus
n_processes = min(n_processes, 2 * n_gpus)
binaries_location = os.environ.get('BINARIES_LOCATION', './build')

# remove trailing slash if it exists
if binaries_location.endswith('/'):
    binaries_location = binaries_location[:-1]

def create_semaphores():
    semaphores = [Semaphore(1) for _ in range(n_gpus)]
    semaphore_names = [f"/device_{i}_lock" for i in range(n_gpus)]
    return semaphores, semaphore_names

def cleanup_semaphores(semaphore_names):
    # Remove specified semaphore names
    for name in semaphore_names:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass

    # Remove all semaphores from /dev/shm directory
    shm_dir = '/dev/shm'
    for filename in os.listdir(shm_dir):
        filepath = os.path.join(shm_dir, filename)
        if filename.startswith('sem.'):
            try:
                os.unlink(filepath)
            except FileNotFoundError:
                pass

def signal_handler(sig, frame, semaphore_names):
    cleanup_semaphores(semaphore_names)
    sys.exit(0)

def run_cudaFPW(file_path, device, datatype, n_process, stop_event):
    command = [
        'taskset',
        '-c',
        # we take 4 threads per process
        str(n_process * 4) + "-" + str((n_process + 1) * 4 - 1),
        f"{binaries_location}/cudaFPW",
        str(file_path),
        str(device),
        str(datatype),
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in process.stdout:
            print(line, end='')
        for line in process.stderr:
            print(line, end='')

        process.wait()

        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            stop_event.set()
    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()

def worker(task_queue, device, n_process, algorithm, datatype, stop_event):
    while not task_queue.empty() and not stop_event.is_set():
        try:
            file_path = task_queue.get_nowait()
        except queue.Empty:
            break

        if algorithm == 'fpw':
            run_cudaFPW(file_path, device, datatype, n_process, stop_event)
        elif algorithm == 'schuster':
            raise NotImplementedError("Schuster algorithm is not implemented")
        task_queue.task_done()

if __name__ == "__main__":
    data_dir = os.environ.get('DATA_DIR', '/data/lightcurves')
    output_dir = os.environ.get('OUTPUT_DIR', '/data/output')
    
    algorithm = 'fpw'
    datatype = 'zvar'

    semaphores, semaphore_names = create_semaphores()
    
    # time.sleep(60)  # Give some time for the semaphores to be created
    
    task_queue = queue.Queue()
    
    # bands = [band] if band else ['g', 'r']
    files = glob.glob(f"{data_dir}/*.h5")
    # remove those that end in _results.h5
    files = [file for file in files if not file.endswith('_results.h5')]
    files_completed = glob.glob(f"{output_dir}/*_results.h5")
    # remove those that are already in the output directory
    files_completed = [file.split('/')[-1] for file in files_completed]
    files = [file for file in files if file.split('/')[-1] not in files_completed]
    files = sorted(files)
    print(f"Found {len(files)} files to process.")
    # for all the files,
    for file in files:
        task_queue.put(file)
    
    threads = []
    stop_event = threading.Event()

    for i in range(n_processes):
        device = i % n_gpus
        # subdivision = i // n_gpus  # Each GPU should process two files at a time
        thread = threading.Thread(target=worker, args=(task_queue, device, i, algorithm, datatype, stop_event))
        thread.start()
        threads.append(thread)
        # Give the thread a name for easier debugging
        thread.name = f"Thread-{i}"
        print(f"Started {thread.name} for device {device}")

    try:
        while not task_queue.empty():
            if stop_event.is_set():
                break
            time.sleep(1)  # Sleep for a short duration to avoid busy-waiting
    except KeyboardInterrupt:
        stop_event.set()
        print("Exiting due to keyboard interrupt")
        cleanup_semaphores(semaphore_names)
        sys.exit(0)
    
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        print("Exiting due to keyboard interrupt")
        cleanup_semaphores(semaphore_names)
        sys.exit(0)

    if stop_event.is_set():
        cleanup_semaphores(semaphore_names)
        print("An error occurred in one of the threads. Exiting with error code.")
        sys.exit(1)
    else:
        cleanup_semaphores(semaphore_names)
        print("All files completed successfully.")
        sys.exit(0)
