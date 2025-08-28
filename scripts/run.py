import subprocess
import threading
import queue
import sys
import time
import os
import signal
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,  # or logging.ERROR, logging.DEBUG, etc.
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_FPW(stop_event, device, filename, output_path, datatype):
    """
    Run the cudaFPW executable for a single file using taskset to bind to a specific CPU core.

    Args:
        stop_event (threading.Event): Event to signal if an error occurs and processing should stop.
        device (int): CPU core/device number to bind the process to.
        filename (str): Path to the input file to process.
        output_path (str): Path to the output directory.
        datatype (str): Data type argument for cudaFPW (e.g., 'zvar' or 'zforce').

    Returns:
        None

    Side Effects:
        Prints stdout and stderr from the process.
        Sets stop_event if the process fails.
    """
    # Build the absolute path to the cudaFPW executable
    cudaFPW_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/cudaFPW"))
    command = [
        'taskset',
        '-c',
        str(device),
        cudaFPW_path,
        filename,
        output_path,
        str(device),
        datatype
    ]

    try:
        # Start the process and stream output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line, end='')
        for line in process.stderr:
            print(line, end='')

        process.wait()

        # If process failed, signal to stop all threads
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            stop_event.set()
    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()


def run_file(task_queue, stop_event, device, output_path, datatype):
    """
    Worker thread function to process files from the task queue.

    Args:
        task_queue (queue.Queue): Thread-safe queue containing filenames to process.
        stop_event (threading.Event): Event to signal if an error occurs and processing should stop.
        device (int): CPU core/device number to bind the process to.
        output_path (str): Path to the output directory.
        datatype (str): Data type argument for cudaFPW.

    Returns:
        None
    """
    while not task_queue.empty() and not stop_event.is_set():
        try:
            filename = task_queue.get_nowait()
        except queue.Empty:
            break

        run_FPW(stop_event, device, filename, output_path, datatype)
        task_queue.task_done()


def run_file_pool(num_threads, input_path, output_path, datatype):
    """
    Process all files in the given input path using a thread pool.

    Args:
        num_threads (int): Number of worker threads to use.
        input_path (str): Directory containing input files.
        output_path (str): Directory to write output files.
        datatype (str): Data type argument for cudaFPW.

    Returns:
        None

    Side Effects:
        Prints status messages.
        Exits the program with code 0 on success, 1 on error.
    """
    # List all files in the input_path (absolute paths only, skip subdirectories)
    files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    
    # Create a thread-safe queue and add each filename as a task
    task_queue = queue.Queue()
    for filename in files:
        task_queue.put(filename)

    threads = []
    stop_event = threading.Event()

    # Start each worker thread, passing the shared task queue
    for i in range(num_threads):
        device = i
        t = threading.Thread(target=run_file, args=(task_queue, stop_event, device, output_path, datatype))
        t.start()
        threads.append(t)

    try:
        while not task_queue.empty():
            if stop_event.is_set():
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping all threads due to keyboard interrupt")
        stop_event.set()
        sys.exit(1)

    # Wait for all threads to finish processing their tasks
    for t in threads:
        t.join()

    if stop_event.is_set():
        print("An error occurred in one of the threads. Exiting with error code.")
        sys.exit(1)
    else:
        print("All files completed successfully.")
        sys.exit(0)

def run_field(num_threads, input_path, output_path, field, datatype, **kwargs):
    """
    Run the specified field processing function.

    Args:
        field (str): The field to process.
        output_path (str): The output path for the processed data.
        datatype (str): The data type (zvar or zforce).

    Returns:
        None
    """

    data_path = os.path.join(input_path, str(field).zfill(4))
    if not os.path.exists(data_path):
        logging.error(f"Data path {data_path} does not exist.")
        sys.exit(1)

    logging.info(f"Processing field {field} of type {datatype} from {data_path} to {output_path}")

    run_file_pool(num_threads, data_path, output_path, datatype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cudaFPW batch processing.")

    # Required positional arguments
    parser.add_argument("function", help="Function to run (e.g., file, field, etc.)")
    parser.add_argument("datatype", choices=["zvar", "zforce"], help="Data type (zvar or zforce)")

    # Optional arguments
    parser.add_argument("--num_threads", type=int, default=int(os.environ.get('NUM_THREADS', 4)),
                        help="Number of worker threads (default: 4 or $NUM_THREADS)")
    parser.add_argument("--input_path", default=os.environ.get('ZVAR_DATA_DIRECTORY'),
                        help="Path to input data directory (default: $ZVAR_DATA_DIRECTORY)")
    parser.add_argument("--output_path", default=os.environ.get('ZVAR_OUTPUT_DIRECTORY'),
                        help="Path to output data directory (default: $ZVAR_OUTPUT_DIRECTORY)")
    parser.add_argument("--log_path", default=os.environ.get('ZVAR_LOG_DIRECTORY'),
                        help="Path to log directory (default: $ZVAR_LOG_DIRECTORY)")
    parser.add_argument("--field", type=int)

    args = parser.parse_args()

    if not args.input_path:
        logging.error("No input_path provided and $ZVAR_DATA_DIRECTORY is not set. Exiting.")
        sys.exit(1)

    if not args.output_path:
        logging.error("No output_path provided and $ZVAR_OUTPUT_DIRECTORY is not set. Exiting.")
        sys.exit(1)
    
    if not args.log_path:
        logging.error("No log_path provided and $ZVAR_LOG_DIRECTORY is not set. Exiting.")
        sys.exit(1)

    if args.function == run_field and not args.field:
        logging.error("Field must be specified when using run_field function.")
        sys.exit(1)

    # Print for debugging
    logging.info(f"Input path: {args.input_path}")
    logging.info(f"Output path: {args.output_path}")
    logging.info(f"Datatype: {args.datatype}")
    logging.info(f"Num threads: {args.num_threads}")

    # Call the appropriate function
    match args.function:
        case "run_field":
            run_field(args.num_threads, args.input_path, args.output_path, args.field, args.datatype)
        case _:
            logging.error(f"Unknown function: {args.function}")
            sys.exit(1)