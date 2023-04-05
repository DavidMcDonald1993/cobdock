import os


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    os.pardir))

if __name__ == "__main__":

    import sys

    sys.path.insert(1, PROJECT_ROOT)

import subprocess

import signal

from timeit import default_timer

SUDO_PASSWORD = os.environ.get("SUDO_PASSWORD", "") 

PROCESS_DIR = os.path.join(
    PROJECT_ROOT, 
    "data", 
    "processes",
    )

running_procs = set()

def save_process_id(
    job_id: int, 
    subpid: int,
    ):
    """Write a subprocesss ID to file

    Parameters
    ----------
    job_id : int
        Main job process ID
    subpid : int
        Sub process ID_
    """
    os.makedirs(PROCESS_DIR, exist_ok=True)
    with open(os.path.join(PROCESS_DIR, str(job_id)), "w") as f:
        f.write(str(subpid))

def get_process_id(
    job_id: int,
    ):
    """Get the current subprocesss ID for a job

    Parameters
    ----------
    job_id : int
        The ID of the job

    Returns
    -------
    int
        Current subprocesss ID
    """

    os.makedirs(PROCESS_DIR, exist_ok=True)
    if os.path.isfile(os.path.join(PROCESS_DIR, str(job_id))):
        with open(os.path.join(PROCESS_DIR, str(job_id)), "r") as f:
            subpid = f.read()
            if subpid == "None":
                return None
            return int(subpid)
    return None

def del_process_file(
    job_id: int,
    ):
    """Delete process file for a job ID

    Parameters
    ----------
    job_id : int
        Job ID
    """
    os.makedirs(PROCESS_DIR, exist_ok=True)
    filename = os.path.join(PROCESS_DIR, str(job_id))
    if os.path.isfile(filename):
        os.remove(filename)
    # delete_file(filename)

def start_main_process(job_id): #these dont get run
    print("====== ADDING " + str(job_id) + " TO RUNNING PROCS IDS")
    save_process_id(job_id, None)

def finish_main_process(job_id):
    print("====== REMOVING " + str(job_id))
    del_process_file(job_id)

def terminate_subprocess(job_id):
    print("========================")
    print(get_process_id(job_id))
    print(job_id)
    subpid = get_process_id(job_id)
    if subpid is not None:
        os.kill(subpid, signal.SIGTERM)

def execute_system_command(
    cmd: str, 
    main_job_id = None,
    timeout = None,
    allow_non_zero_return: bool = False,
    as_subprocess: bool = False,
    verbose: bool = False,
    ):
    """Execute a system command as a subprocess.

    Parameters
    ----------
    cmd : str
        The system command to execute
    main_job_id : int, optional
        ID of main job, by default None
    timeout : int, optional
        Timeout for comman before a SIGKILL signal is sent, by default None
    allow_non_zero_return : bool, optional
        Flag to allow a non-zero return code, by default False
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    int
        System return code

    Raises
    ------
    Exception
        System command exited with non-zero return code
    """

    if timeout:
        cmd = f"timeout -s SIGKILL {timeout} {cmd}"

    if verbose:
        print ("Executing system command:", cmd, )
        start_time = default_timer()


    if as_subprocess:

        p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", )
        if verbose:
            print("Adding process ID", p.pid , "to running procs")
        running_procs.add(p)
        if main_job_id is not None:
            save_process_id(main_job_id, p.pid)

        return_code = p.wait() # wait on completion of subprocess+
        if verbose:
            print("Removing process ID", p.pid , "from running procs")
        running_procs.remove(p)    
        if main_job_id is not None:
            save_process_id(main_job_id, None)

    else:

        return_code = os.system(cmd)

    if return_code != 0 and not allow_non_zero_return:
        # raise Warning("BAD RETURN CODE")
        raise Exception("Bad return code:", return_code)
        
    if verbose:
        print ("Obtained return code", return_code)
        time_taken = default_timer() - start_time
        print ("Completed in", round(time_taken, 3), "seconds")

    return return_code

if __name__ == "__main__":

    cmd = "echo hello world"

    execute_system_command(cmd, )