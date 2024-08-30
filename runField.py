#Delegates the running of every file in a field directory.
import subprocess
import threading
import queue
import sys
import time
import os
import signal
from multiprocessing import Semaphore

def create_semaphores():
    semaphores = [Semaphore(1) for _ in range(4)]
    semaphore_names = [f"/device_{i}_lock" for i in range(4)]
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

def run_cudaFPW(field, ccd, quad, band, device, subdivision, stop_event):
    command = [
        'taskset',
        '-c',
        # str(9 * device) + '-' + str(9 * device + 8),
        str(8 * device + 4 * subdivision) + '-' + str(8 * device + 4 * subdivision + 3),
        './build/cudaFPW',
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

def worker(task_queue, device, subdivision, algorithm, stop_event):
    while not task_queue.empty() and not stop_event.is_set():
        try:
            field, ccd, quad, band = task_queue.get_nowait()
        except queue.Empty:
            break

        if algorithm == 'fpw':
            run_cudaFPW(field, ccd, quad, band, device, subdivision, stop_event)
        elif algorithm == 'schuster':
            run_schuster(field, ccd, quad, band, device, stop_event)
        task_queue.task_done()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python runField.py <field> <algorithm>")
        sys.exit(1)
    
    field = sys.argv[1]
    algorithm = sys.argv[2]

    if algorithm not in ['schuster', 'fpw']:
        print("ERROR: algorithm must be one of 'schuster' or 'fpw'")
        print("Usage: python runField.py <field> <algorithm>")
        sys.exit(1)

    semaphores, semaphore_names = create_semaphores()
    
    task_queue = queue.Queue()
    
    for band in ['g', 'r']:
        for ccd in range(1, 17):
            for quad in range(1, 5):
                task_queue.put((field, ccd, quad, band))
    
    num_threads = 8
    threads = []
    stop_event = threading.Event()

    for i in range(num_threads):
        device = int(i/2) #There are four GPUs 0-3
        subdivision = i % 2 #Each GPU should process two files at a time
        thread = threading.Thread(target=worker, args=(task_queue, device, subdivision, algorithm, stop_event))
        thread.start()
        threads.append(thread)
    
    while not task_queue.empty():
        if stop_event.is_set():
            break
        time.sleep(1)  # Sleep for a short duration to avoid busy-waiting
    
    for thread in threads:
        thread.join()

    if stop_event.is_set():
        cleanup_semaphores(semaphore_names)
        print("An error occurred in one of the threads. Exiting with error code.")
        sys.exit(1)
    else:
        cleanup_semaphores(semaphore_names)
        print("All files completed successfully.")
        sys.exit(0)
