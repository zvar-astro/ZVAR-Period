#Delegates the running of every file in a field directory.
import subprocess
import threading
import queue
import sys
import time

def run_cudaFPW(field, ccd, quad, band, device, stop_event):
    command = [
        'taskset',
        '-c',
        str(9 * device) + '-' + str(9 * device + 8),
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

def worker(task_queue, device, algorithm, stop_event):
    while not task_queue.empty() and not stop_event.is_set():
        try:
            field, ccd, quad, band = task_queue.get_nowait()
        except queue.Empty:
            break

        if algorithm == 'fpw':
            run_cudaFPW(field, ccd, quad, band, device, stop_event)
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
    
    task_queue = queue.Queue()
    
    for band in ['g', 'r']:
        for ccd in range(1, 17):
            for quad in range(1, 5):
                task_queue.put((field, ccd, quad, band))
    
    num_threads = 4
    threads = []
    stop_event = threading.Event()

    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(task_queue, i, algorithm, stop_event))
        thread.start()
        threads.append(thread)
    
    while not task_queue.empty():
        if stop_event.is_set():
            break
        time.sleep(1)  # Sleep for a short duration to avoid busy-waiting
    
    for thread in threads:
        thread.join()

    if stop_event.is_set():
        print("An error occurred in one of the threads. Exiting with error code.")
        sys.exit(1)
