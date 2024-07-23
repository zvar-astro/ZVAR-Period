#Delegates the running of every file in a field directory.
import subprocess
import threading
import queue
import sys

def run_cudaFPW(field, ccd, quad, band, device):
    # Construct the command with the given arguments
    command = [
        './build/cudaFPW',  # Path to the executable
        str(field),
        str(ccd),
        str(quad),
        str(band),
        str(device)
    ]

    try:
        # Run the command and capture the output in real-time
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Print stdout and stderr as they come out
        for line in process.stdout:
            print(line, end='')
        for line in process.stderr:
            print(line, end='')

        # Wait for the process to complete
        process.wait()

        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
    except Exception as e:
        print(f"An error occurred: {e}")

def run_schuster(field, ccd, quad, band, device):
    # Construct the command with the given arguments
    command = [
        './build/schuster',  # Path to the executable
        str(field),
        str(ccd),
        str(quad),
        str(band),
        str(device)
    ]

    try:
        # Run the command and capture the output in real-time
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Print stdout and stderr as they come out
        for line in process.stdout:
            print(line, end='')
        for line in process.stderr:
            print(line, end='')

        # Wait for the process to complete
        process.wait()

        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
    except Exception as e:
        print(f"An error occurred: {e}")


def worker(task_queue, device, algorithm):
    while not task_queue.empty():
        field, ccd, quad, band = task_queue.get()
        if algorithm == 'fpw':
            run_cudaFPW(field, ccd, quad, band, device)
        elif algorithm == 'schuster':
            run_schuster(field, ccd, quad, band, device)
        task_queue.task_done()

if __name__ == "__main__":
    # Check for the correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python runField.py <field> <algorithm>")
        sys.exit(1)
    
    #Take field as argument
    field = sys.argv[1]
    algorithm = sys.argv[2]

    # Check that algorithm is an allowed value
    if algorithm not in ['schuster','fpw']:
        print("ERROR: algrotihm must be one of 'schuster' or 'fpw'")
        print("Usage: python runField.py <field> <algorithm>")
        sys.exit(1)
    
    # Create a queue to hold the tasks
    task_queue = queue.Queue()
    
    # Populate the queue with tasks, without device as it's now handled by the worker
    for band in ['g', 'r']:
        for ccd in range(1, 17):
            for quad in range(1, 5):
                task_queue.put((field, ccd, quad, band))
    
    # Create and start worker threads, each with a unique device ID
    num_threads = 4
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(task_queue, i, algorithm))  # Pass device ID as argument to worker
        thread.start()
        threads.append(thread)
    
    # Wait for all tasks to be processed
    task_queue.join()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
