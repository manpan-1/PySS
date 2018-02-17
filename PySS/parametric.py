from itertools import product
from shutil import rmtree
from time import sleep
from random import random
from math import sqrt
import os


# Return a list with all the divisors of a numbers
def divisors(n):
    """
    Divisors of an integer.

    Return all the possible divisors for a given integer.

    Parameters
    ----------
    n: int

    Returns
    -------
    int

    Notes
    -----

    """
    large_divisors = []
    for i in range(1, int(sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


# range(6, 31, 2),
# range(27, 55, 3),

def parametric_run(
    prj_name,
    exec_func,
    func_args,
    param_indices=None,
    mk_subdirs=None,
    del_subdirs=None,
    delay_jobs=None,
    **kwargs
):
    """
    Run a parametric job.

    Creates a full factorial matrix for a given list of parameters and executes a job for each combination. A list of
    values is needed for each parameter.
    Each job is executed in a different automatically created subdirectory and a summary of the results is written in
    parent directory.
    REFRESH-EXTEND DOCSTRING

    Parameters
    ----------
    prj_name : str
        Name of the parametric project. Used for directory name and filenames.
    exec_func : function name
        Function to be executed parametrically.
    func_args : list
        A list containing the arguments to be passed to the function, both parametric and static.
    param_indices : list, optional
        A list of integers, indicating the positions on the func_args list of the arguments for which the parametric
        matrix is composed. The func_args list must contain a list type on the positions described by param_indices.
        Default behaviour assumes that all of the arguments contained in func_args are participating in the construction
        of the parametric matrix and thus, it expects that func_args contains only lists (func_args = list of lists).
    mk_subdirs : bool, optional
        Perform each execution in a separate subdirectory.
        Useful when the executed job generates files on the working directory.
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

    Notes
    -----

    """
    # Defaults
    if param_indices is None:
        param_indices = range(0, len(func_args))

    if mk_subdirs is None:
        mk_subdirs = False

    if del_subdirs is None:
        del_subdirs = False

    if delay_jobs is None:
        delay_jobs = False

    # Pick up the parametric variables from the list of arguments
    param_args = [func_args[x] for x in param_indices]

    # Cartesian product of parameters
    combinations = list(product(*param_args))

    # Open a file to collect the results
    out_file = open('./' + prj_name + '_info.dat', 'a')

    # Initiate a list for the results
    prj_results = []

    # Loop through the combinations of the given input values
    for parameters in combinations:
        job_id = (''.join("%03d-" % e for e in parameters) + prj_name)
        job_id = job_id.translate(None, '.')

        # Wait some seconds to avoid multiple initiation of jobs
        if delay_jobs is True:
            sleep(round(10 * random(), 2))
        #TODO: does the following line need to be under the if mk_subdirs condition??
        # Check if the directory exists
        if os.path.isdir("./" + job_id):
            print("Job already exists: A directory with the same name'" + job_id + "' exists in the cwd")
            continue

        if mk_subdirs is True:
            # Make a new subdirectory for the current session
            os.mkdir(job_id)

            # Change working directory
            os.chdir('./' + job_id)

        # Assemble current job's input arguments
        current_func_args = func_args
        for (index, new_parameter) in zip(param_indices, parameters):
            current_func_args[index] = new_parameter

        # The function to be run for the full factorial parametric is called here
        print('Running job: ' + job_id)

        try:
            # Execute the current job
            job_return = exec_func(*current_func_args, **kwargs)

            # Create an output string
            return_string = str(job_return)

            # Return to parent directory
            if mk_subdirs is True:
                os.chdir('../')

            # Write each returned string to the file separated by newlines
            # job_return = str(job_return)
            # TODO: create a standardised output that can be imported by the FEM results class in steel_tools.Column titles, input and output.
            out_file.write(job_id + ", " + return_string + "\n")

            # Add the result of the current job to the return list
            prj_results = prj_results + [job_return]
        #TODO: Write log file
        except:
            print('Problem while executing job: ' + job_id)
            print('Job is canceled. See log file (no log file yet)')
            if mk_subdirs is True:
                os.chdir('../')

        # Remove job's folder (only the output information is kept)
        if mk_subdirs and del_subdirs is True:
            rmtree(job_id)

    # Close the project results file
    out_file.close()

    # Return a list of results
    return prj_results
