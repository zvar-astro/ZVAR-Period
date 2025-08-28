#Delegates the running of every file in a field directory.
import subprocess
import threading
import queue
import sys
import time
import os
import signal

def signal_handler(sig, frame):
    sys.exit(0)

def run_cudaFPW(filepath, device, datatype, stop_event):
    command = [
        'taskset',
        '-c',
        str(device), 
        '../build/cudaFPW',
        filepath,
        str(device),
        datatype
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

def run_schuster(field, ccd, quad, band, device, stop_event):
    command = [
        './build/schuster',
        str(field),
        str(ccd),
        str(quad),
        str(band),
        str(device)
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

def worker(task_queue, device, algorithm, stop_event):
    while not task_queue.empty() and not stop_event.is_set():
        try:
            file_path, datatype = task_queue.get_nowait()
        except queue.Empty:
            break

        if algorithm == 'fpw':
            run_cudaFPW(file_path, device, datatype, stop_event)
        elif algorithm == 'schuster':
            # run_schuster(field, ccd, quad, band, device, stop_event)
            pass
        task_queue.task_done()

if __name__ == "__main__":
    # data_dir = os.environ.get('DATA_DIR', '/data/zvar/matchfiles')
    data_dir = '/data/zvar/matchfiles'
    num_threads = os.environ.get('NUM_THREADS', 4)
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python runField.py <field> <algorithm> [<band>]")
        sys.exit(1)
    
    field = sys.argv[1]
    algorithm = sys.argv[2]
    band = sys.argv[3] if len(sys.argv) == 4 else None

    if algorithm not in ['schuster', 'fpw']:
        print("ERROR: algorithm must be one of 'schuster' or 'fpw'")
        print("Usage: python runField.py <field> <algorithm> [<band>]")
        sys.exit(1)

    if band and band not in ['g', 'r']:
        print("ERROR: band must be one of 'g' or 'r'")
        print("Usage: python runField.py <field> <algorithm> [<band>]")
        sys.exit(1)
    
    task_queue = queue.Queue()
    
    bands = [band] if band else ['g', 'r']
    for b in bands:
        for ccd in range(1, 17):
            for quad in range(1, 5):
                #Check if the file exists
                file_name = f"data_{str(field).zfill(4)}_{str(ccd).zfill(2)}_{str(quad)}_z{b}.h5"
                file_path = os.path.join(data_dir, str(field).zfill(4), file_name)
                if os.path.exists(file_path):
                    task_queue.put((file_path, "zvar"))
    
    threads = []
    stop_event = threading.Event()

    for i in range(num_threads):
        device = i #There are four GPUs 0-3
        thread = threading.Thread(target=worker, args=(task_queue, device, algorithm, stop_event))
        thread.start()
        threads.append(thread)

    try:
        while not task_queue.empty():
            if stop_event.is_set():
                break
            time.sleep(1)  # Sleep for a short duration to avoid busy-waiting
    except KeyboardInterrupt:
        stop_event.set()
        print("Exiting due to keyboard interrupt")
        sys.exit(0)
    
    for thread in threads:
        thread.join()

    if stop_event.is_set():
        print("An error occurred in one of the threads. Exiting with error code.")
        sys.exit(1)
    else:
        print("All files completed successfully.")
        sys.exit(0)
