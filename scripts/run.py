import subprocess
import threading
import queue
import sys
import time
import os
import signal
import argparse
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,  # or logging.ERROR, logging.DEBUG, etc.
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def writeout_field_log(field_logfile, field):
    with open(field_logfile, 'a') as log:
        log.write(f'Completed field: {str(field)}\n')

def run_FPW(stop_event, device, filename, output_path, logfile, datatype, error_queue, optional_args=None):
    """
    Run the cudaFPW executable for a single file using taskset to bind to a specific CPU core.

    Args:
        stop_event (threading.Event): Event to signal if an error occurs and processing should stop.
        device (int): CPU core/device number to bind the process to.
        filename (str): Path to the input file to process.
        output_path (str): Path to the output directory.
        logfile (str): Path to the log file.
        datatype (str): Data type argument for cudaFPW (e.g., 'zvar' or 'zforce').
        error_queue (queue.Queue): Queue to report errors back to the main thread.
        optional_args (dict, optional): Additional optional arguments to pass to cudaFPW.

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
        logfile,
        str(device),
        datatype
    ]

    if optional_args is not None:
        for arg, value in optional_args.items():
            if value is not None:
                command.append(f"--{arg}={value}")
            else: # Error if value is None
                logging.error(f"Optional argument {arg} has no value.")
                stop_event.set()
                return
                

    try:
        # Start the process and stream output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line, end='')
        stderr_output = ""
        for line in process.stderr:
            print(line, end='')
            stderr_output += line

        process.wait()

        # If process failed, signal to stop all threads
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            stop_event.set()
            if stderr_output.strip():
                error_queue.put(f"File: {filename}\nReturn code: {process.returncode}\nStderr:\n{stderr_output}")
            else:
                error_queue.put(f"File: {filename}\nReturn code: {process.returncode}\nNo stderr output captured.")
    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()
        error_queue.put(str(e))


def run_file(task_queue, stop_event, device, output_path, logfile, datatype, error_queue, optional_args=None):
    """
    Worker thread function to process files from the task queue.

    Args:
        task_queue (queue.Queue): Thread-safe queue containing filenames to process.
        stop_event (threading.Event): Event to signal if an error occurs and processing should stop.
        device (int): CPU core/device number to bind the process to.
        output_path (str): Path to the output directory.
        logfile (str): Path to the log file.
        datatype (str): Data type argument for cudaFPW.
        error_queue (queue.Queue): Queue to report errors back to the main thread.
        optional_args (dict, optional): Additional optional arguments to pass to cudaFPW.

    Returns:
        None
    """
    while not task_queue.empty() and not stop_event.is_set():
        try:
            filename = task_queue.get_nowait()
        except queue.Empty:
            break

        run_FPW(stop_event, device, filename, output_path, logfile, datatype, error_queue, optional_args)
        task_queue.task_done()


def run_file_pool(num_threads, input_path, output_path, log_path, datatype, process_i=False, optional_args=None):
    """
    Process all files in the given input path using a thread pool.

    Args:
        num_threads (int): Number of worker threads to use.
        input_path (str): Directory containing input files.
        output_path (str): Directory to write output files.
        log_path (str): Directory to write log files.
        datatype (str): Data type argument for cudaFPW.

    Returns:
        None

    Side Effects:
        Prints status messages.
        Exits the program with code 0 on success, 1 on error.
    """
    # List all files in the input path excluding those ending with "i.h5"
    if process_i:
        files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f))
        ]
    else:
        files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f)) and not f.endswith("i.h5") # Dont include i band
        ]
    
    # Create a thread-safe queue and add each filename as a task
    task_queue = queue.Queue()
    for filename in files:
        task_queue.put(filename)

    threads = []
    stop_event = threading.Event()

    error_queue = queue.Queue()

    # Start each worker thread, passing the shared task queue
    for i in range(num_threads):
        device = i
        logfile = os.path.join(log_path, f"device_{device}.log")
        t = threading.Thread(target=run_file, args=(task_queue, stop_event, device, output_path, logfile, datatype, error_queue, optional_args))
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
        while not error_queue.empty():
            print("Error details:", error_queue.get())
        sys.exit(1)
    else:
        print("All files completed successfully.")
        return


def run_field(num_threads, input_path, output_path, log_path, field, datatype, optional_args=None):
    """
    Run the specified field processing function.

    Args:
        num_threads (int): Number of worker threads to use.
        field (str): The field to process.
        input_path (str): The input path containing the field data.
        output_path (str): The output path for the processed data.
        log_path (str): The path for log files.
        field (int): The field number to process.
        datatype (str): The data type (zvar or zforce).

    Returns:
        None
    """

    data_path = os.path.join(input_path, str(field).zfill(4))
    if not os.path.exists(data_path):
        logging.error(f"Data path {data_path} does not exist.")
        sys.exit(1)

    logging.info(f"Processing field {field} of type {datatype} from {data_path} to {output_path}")

    run_file_pool(num_threads, data_path, output_path, log_path, datatype, optional_args)
    writeout_field_log(os.path.join(log_path, f"completed_fields.log"), field)


def run_all(num_threads, input_path, output_path, log_path, datatype, optional_args=None):
    """
    Run processing for all fields in the input path.

    Args:
        num_threads (int): Number of worker threads to use.
        input_path (str): The input path containing all fields (directories).
        output_path (str): The output path for the processed data.
        log_path (str): The path for log files.
        datatype (str): The data type (zvar or zforce).

    Returns:
        None
    """
    if not os.path.exists(input_path):
        logging.error(f"Input path {input_path} does not exist.")
        sys.exit(1)

    logging.info(f"Processing all fields of type {datatype} from {input_path} to {output_path}")

    # List all subdirectories (fields) in input_path
    field_dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    if not field_dirs:
        logging.warning(f"No field directories found in {input_path}")
        return

    for field in field_dirs:
        field_input_dir = os.path.join(input_path, field)
        field_output_dir = os.path.join(output_path, field)
        if os.path.exists(field_output_dir):
            logging.info(f"Skipping field {field}: output directory {field_output_dir} already exists.")
            continue
        os.makedirs(field_output_dir, exist_ok=True)
        logging.info(f"Processing field {field}: {field_input_dir} -> {field_output_dir}")
        run_file_pool(num_threads, field_input_dir, field_output_dir, log_path, datatype, optional_args)
        writeout_field_log(os.path.join(log_path, f"completed_fields.log"), field)


def run_queued(num_threads, input_path, output_path, log_path, datatype, optional_args=None):
    """
    Run processing for fields listed in the manual queue file.

    Args:
    num_threads (int): Number of worker threads to use.
    input_path (str): The input path containing all fields (directories).
    output_path (str): The output path for the processed data.
    log_path (str): The path for log files.
    datatype (str): The data type (zvar or zforce).

    Returns:
        None
    """
    if not os.path.exists(input_path):
        logging.error(f"Input path {input_path} does not exist.")
        sys.exit(1)

    queue_file = os.path.join(log_path, 'manual_queue.txt')
    if not os.path.exists(queue_file):
        logging.error(f"Queue file {queue_file} does not exist.")
        sys.exit(1)

    # List all fields in the queue file
    field_list = np.loadtxt(queue_file, dtype=int)

    for field in field_list:
        field_input_dir = os.path.join(input_path, str(field).zfill(4))
        field_output_dir = os.path.join(output_path, str(field).zfill(4))
        if os.path.exists(field_output_dir):
            logging.info(f"Skipping field {field}: output directory {field_output_dir} already exists.")
            continue
        os.makedirs(field_output_dir, exist_ok=True)
        logging.info(f"Processing field {field}: {field_input_dir} -> {field_output_dir}")
        run_file_pool(num_threads, field_input_dir, field_output_dir, log_path, datatype, optional_args)
        writeout_field_log(os.path.join(log_path, f"completed_fields.log"), field)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cudaFPW batch processing.")

    # Required positional arguments
    parser.add_argument("function", help="Function to run [run_field, run_field_long_period, run_all, run_all_long_period, run_queued, run_queued_long_period]")")
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
            run_field(args.num_threads, args.input_path, args.output_path, args.log_path, args.field, args.datatype)
        case "run_field_long_period":
            optional_args = {"remove_baseline": 0, "f_min": 1.0 / 1000.0, "f_max": 1.0 / 10.0, "clip_pgram": 1}
            run_field(args.num_threads, args.input_path, args.output_path, args.log_path, args.field, args.datatype, optional_args)
        case "run_all":
            run_all(args.num_threads, args.input_path, args.output_path, args.log_path, args.datatype)
        case "run_all_long_period":
            optional_args = {"remove_baseline": 0, "f_min": 1.0 / 1000.0, "f_max": 1.0 / 10.0, "clip_pgram": 1}
            run_all(args.num_threads, args.input_path, args.output_path, args.log_path, args.datatype, optional_args)
        case "run_queued":
            run_queued(args.num_threads, args.input_path, args.output_path, args.log_path, args.datatype)
        case "run_queued_long_period":
            optional_args = {"remove_baseline": 0, "f_min": 1.0 / 1000.0, "f_max": 1.0 / 10.0, "clip_pgram": 1}
            run_queued(args.num_threads, args.input_path, args.output_path, args.log_path, args.datatype, optional_args)
        case _:
            logging.error(f"Unknown function: {args.function}")
            # Print the possible functions
            logging.info("Possible functions: run_field, run_field_long_period, run_all, run_all_long_period")
            sys.exit(1)