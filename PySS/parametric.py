from itertools import product
from shutil import rmtree
from time import sleep
from random import random
import os
import FileLock as flck


def testfunc(*args, **kargs):
    return("")

def parametric_run(
    prj_name,
    exec_func,
    func_args=None,
    func_kargs=None,
    p_args=None,
    p_kargs=None,
    mk_subdirs=None,
    del_subdirs=None,
    delay_jobs=None,
):
    """
    Run a parametric job.

    Creates a full factorial matrix for a given list of parameters and executes a job for each combination. A list of
    values is needed for each parameter.
    Each job is executed in a different automatically created subdirectory and a summary of the results is written in
    parent directory.

    Parameters
    ----------
    prj_name : str
        Name of the parametric project. Used for directory name and filenames.
    exec_func : function name
        Function to be executed parametrically.
    func_args : list, optional
        A list containing the arguments to be passed to the function, both parametric and static.
    func_kargs : dict, optional
        A dictionaty of all the keyword arguments to be passed to the function, both parametric and static.
    p_args : list of integers, optional
        A list of integers, indicating the positions on the func_args list of the arguments for which the parametric
        matrix is composed. On the positions indicated by p_args, the func_args must contain list items.
    p_kargs : list of strings, optional
        A list of the keys of all the keyword arguments that need to be executed parametrically, similarly to p_args.
    mk_subdirs : bool, optional
        Perform each execution in a separate subdirectory.
        Useful when the executed job generates files on the working directory. If subdirectory with the same name
        exists, the job will run in the directory ignoring pre-existing files.
        Default is False (run all jobs in the current directory)
    del_subdirs : bool, optional
        Remove the subdirectories for the individual jobs and keep only the results summary.
        This argument is used in combination with the mk_subdirs.
        Default is False (do not remove).
    delay_jobs : bool (optional)
        Add a small random delay before each job execution. Helps resolving conflicts when executed multiple parametric
        projects in a slurm environment (2 jobs might initiate simultaneously and try entering the same directory)
        Default is False (do not delay).

    Returns
    -------
    list
        List item containing the results of all the executed runs

    """
    # Defaults
    if p_args is None:
        p_args = []

    if mk_subdirs is None:
        mk_subdirs = False

    if del_subdirs is None:
        del_subdirs = False

    if delay_jobs is None:
        delay_jobs = False

    # Pick up the parametric variables from the list of arguments
    if func_args:
        param_args = [func_args[x] for x in p_args]
    else:
        param_args = []

    if func_kargs:
        param_kargs = [func_kargs[x] for x in p_kargs]
    else:
        param_kargs = []

    print("The parametric argument values are: \n", param_args)
    print("The parametric keyword argument values are: \n", param_kargs)

    # Cartesian product of parameters
    all_params = param_args + param_kargs
    combinations = list(product(*all_params))

    # Check for existing "last_job.log" and if not, initialise a new one
    queue = "last_job.log"
    if not os.path.isfile(queue):
        with open(queue, "w+") as f:
            curr_job_nr = -1
            f.write(str(curr_job_nr))
    else:
        with open(queue, "r") as f:
            curr_job_nr = int(f.read())

    # Initiate a list for the results
    prj_results = []

    # Loop through the combinations of the given input values
    while curr_job_nr < len(combinations)-1:
        with flck.FileLock(queue):
            with open(queue, "r+") as f:
                curr_job_nr = int(f.read()) + 1
                #TODO: check the break condition
                if curr_job_nr == len(combinations):
                    break
                f.seek(0)
                f.truncate()
                f.write(str(curr_job_nr))

        comb_case = combinations[curr_job_nr]
        # Construct an id string for the current job based on the combination. The string is formatted so that it does
        # contain any illegal characters for filename and abaqus job name usage.
        job_id = str(comb_case).translate(None, ",.&*~!()[]{}|;:\'\"`<>?/\\")
        job_id = job_id.replace(" ", "-")
        job_id = job_id + "-" + prj_name

        # Check if the directory exists. If so, continue to the next job.
        if os.path.isdir("./" + job_id):
            print("Job already exists: A directory with the same name'" + job_id + "' exists in the cwd")
            continue

        # Wait some seconds to avoid multiple initiation of jobs
        if delay_jobs is True:
            sleep(round(10 * random(), 2))

        if mk_subdirs is True:
            if os.path.isdir(job_id):
                os.chdir(job_id)
            else:
                # Make a new subdirectory for the current session
                os.mkdir(job_id)

                # Change working directory
                os.chdir(job_id)

        # Assemble current job's input arguments
        if func_args:
            current_func_args = func_args
            for (index, new_parameter) in zip(p_args, comb_case):
                current_func_args[index] = new_parameter
        else:
            current_func_args = []

        # Assemble current job's input keyword arguments
        if func_kargs:
            current_func_kargs = func_kargs
            for i, x in enumerate(p_kargs):
                current_func_kargs[x] = comb_case[len(p_args)+i]
            current_func_kargs["IDstring"] = job_id
        else:
            current_func_kargs = {}

        # The function to be run for the full factorial parametric is called here
        print("Running job nr: " + str(curr_job_nr) + " with name: " + job_id)

        # Execute the current job
        job_return = exec_func(*current_func_args, **current_func_kargs)

        # Create an output string
        return_string = str(job_return)

        # Return to parent directory
        if mk_subdirs is True:
            os.chdir('../')

        # Open a file to collect the results
        with flck.FileLock('./' + prj_name + '_info.dat'):
            with open('./' + prj_name + '_info.dat', 'a') as out_file:
                out_file.write(job_id + ", " + return_string + "\n")

        # Add the result of the current job to the return list
        prj_results = prj_results + [job_return]

        # Remove job's folder (only the output information is kept)
        if mk_subdirs and del_subdirs is True:
            rmtree(job_id)

    # Return a list of results
    return prj_results

def subdir_crawler(
        exec_func,
        func_args=None,
        func_kargs=None,
        prj_name=None,):
    """
    Execute function in all subdirs.

    Parameters
    ----------
    :param exec_func:
    :param out_file:
    :return:
    """
    if func_args is None:
        func_args = ()

    if func_kargs is None:
        func_kargs = {}

    if prj_name is None:
        prj_name = ""

    #TODO: list the directories with an if condition, enter them and executue the function
    all_dirs = [i for i in os.listdir('.') if os.path.isdir(i)]

    for directory in all_dirs:
        os.chdir(directory)
        func_return = exec_func(*func_args, **func_kargs)
        os.chdir("..")
        with open('./' + prj_name + '_info.dat', 'a') as out_file:
            out_file.write(directory + ", " + str(func_return) + "\n")
