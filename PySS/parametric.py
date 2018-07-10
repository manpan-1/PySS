import sys
from itertools import product
from shutil import rmtree
from time import sleep
from random import random
import os
import PySS.FileLock as flck
import csv


def testfunc(*args, **kargs):
    return("")


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
    for i in range(1, int(n**.5 + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(int(n / i))
    for divisor in reversed(large_divisors):
        yield divisor

def get_queued(filename):
    """Read a batch info file and return the indexes of the queued jobs"""
    queue = []
    with flck.FileLock(filename):
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if sys.version[0] == "2":
                    curr_line = line.split(",")
                else:
                    curr_line = line.split(sep=",")
                if curr_line[-1] == "QUEUED\n":
                    queue.append(i)
    return queue


def goto_next_queued(filename):
    """Read a batch info file and return the indexes of the queued jobs"""
    with flck.FileLock(filename):
        lines = open(filename, "r").readlines()
        for i, line in enumerate(lines):
            if sys.version[0] == "2":
                curr_line = line.split(",")
            else:
                curr_line = line.split(sep=",")
            if curr_line[-1] == "QUEUED\n":
                lines[i] = ",".join(curr_line[:-1]) + ",RUNNING\n"
                with open(filename, "w") as f:
                    f.writelines(lines)
                return i
        return False


def update_job_status(filename, job_nr, new_status):
    """Change status of a job number on a batch status file"""
    with flck.FileLock(filename):
        lines = open(filename, "r").readlines()
        if sys.version[0] == "2":
            curr_line = lines[job_nr].split(",")
        else:
            curr_line = lines[job_nr].split(sep=",")
        lines[job_nr] = ",".join(curr_line[:-1]) + "," + new_status + "\n"
        with open(filename, "w") as f:
            f.writelines(lines)


def run_factorial(
    prj_name,
    exec_func,
    func_args=None,
    func_kargs=None,
    p_args=None,
    p_kargs=None,
    mk_subdirs=None,
    del_subdirs=None,
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
        
    if p_kargs is None:
        p_kargs = []

    if mk_subdirs is None:
        mk_subdirs = False

    if del_subdirs is None:
        del_subdirs = False

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

    isnumeric = lambda x: isinstance(x, int) or isinstance(x, float)
    # numeric_dimensions = [(param_args.index(x), len(str(max(x)))) for x in param_args if isnumeric(max(x))]
    numeric_dimensions = [param_args.index(x) for x in param_args if isnumeric(max(x))]
    leading_zeros = [len(str(max(x))) for x in param_args if isnumeric(max(x))]
    
    # Write the combinations in a file
    batch_status_file = prj_name + "_batch_status.dat"
    if not os.path.isfile(batch_status_file):
        with open(batch_status_file, "w") as f:
            for dimensions in combinations:
                dim_string = ""
                for idx, dimension in enumerate(dimensions):
                    if idx in numeric_dimensions:
                        dim_string = dim_string + str(dimension).zfill(leading_zeros[idx]) + ","
                    else:
                        dim_string = dim_string + str(dimension) + ","
                dim_string = dim_string[:-1] + ",QUEUED" + "\n"
                f.write(dim_string)
    
    # Initiate (if doesn't exist) a file for the results
    results_file = prj_name + "_results.dat"
    if not os.path.isfile(results_file):
        with open(results_file, "w") as f:
            for dimensions in combinations:
                dim_string = ""
                for idx, dimension in enumerate(dimensions):
                    if idx in numeric_dimensions:
                        dim_string = dim_string + str(dimension).zfill(leading_zeros[idx]) + ","
                    else:
                        dim_string = dim_string + str(dimension) + ","
                dim_string = dim_string[:-1] + ",Wait for it...!" + "\n"
                f.write(dim_string)
    # Initiate a list for the results
    prj_results = []

    while True:
        curr_job_nr = goto_next_queued(batch_status_file)
        if curr_job_nr or curr_job_nr is 0:
            comb_case = combinations[curr_job_nr]
            # Construct an id string for the current job based on the combination. The string is formatted so that it does
            # contain any illegal characters for filename and abaqus job name usage.
            if sys.version[0] == "2":
                job_id = str(comb_case).translate(None, ",.&*~!()[]{}|;:\'\"`<>?/\\")
            else:
                job_id = str(comb_case).translate(str.maketrans("", "", ",.&*~!()[]{}|;:\'\"`<>?/\\"))
            job_id = job_id.replace(" ", "-")
            job_id = job_id + "-" + prj_name
    
            # # Check if the directory exists. If so, continue to the next job.
            # if os.path.isdir("./" + job_id):
            #     print("Job already exists: A directory with the same name'" + job_id + "' exists in the cwd")
            #     continue
    
            if mk_subdirs:
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
                    current_func_kargs[x] = comb_case[len(p_args) + i]
                current_func_kargs["IDstring"] = job_id
            else:
                current_func_kargs = {}
    
            # The function to be run for the full factorial parametric is called here
            print("Running job nr: " + str(curr_job_nr) + " with name: " + job_id)
    
            # Execute the current job
            try:
                result = exec_func(*current_func_args, **current_func_kargs)
                new_status = "COMPLETED"
            except:
                result = []
                new_status = "FAILED"
    
            # Create an output string
            result = str(result)
    
            # Return to parent directory
            if mk_subdirs is True:
                os.chdir('../')
    
            # with flck.FileLock('./' + prj_name + '_results.dat'):
            #     with open('./' + prj_name + '_results.dat', 'a') as out_file:
            #         out_file.write(job_id + "," + str(curr_job_nr) + "," + result + "\n")
            
            # Add the result of the current job to the return list
            prj_results = prj_results + [result]
    
            # Remove job's folder (only the output information is kept)
            if mk_subdirs and del_subdirs is True:
                rmtree(job_id)

            # Write the result of the current job to the file
            update_job_status(results_file, curr_job_nr, result)
            
            # Update the job status on the common batch status file
            update_job_status(batch_status_file, curr_job_nr, new_status)
        else:
            break
        
    return(prj_results)
    # 
    # # Check for existing "last_job.tmp" and if not, initialise a new one
    # queue = "last_job_of_"+prj_name+".tmp"
    # if not os.path.isfile(queue):
    #     with open(queue, "w+") as f:
    #         curr_job_nr = -1
    #         f.write(str(curr_job_nr) + ",00")
    # else:
    #     with open(queue, "r") as f:
    #         reader = csv.reader(f)
    #         info = list(reader)[0]
    #         curr_job_nr = int(info[1])
    #         if curr_job_nr == len(combinations):
    #             print("A previous batch job exists in the directory. Check \"" + queue + "\"")
    #             return
    # 
    # # Loop through the combinations of the given input values
    # # while curr_job_nr < len(combinations)-1:
    # while True:
    #     with flck.FileLock(queue):
    #         with open(queue, "r+") as f:
    #             reader = csv.reader(f)
    #             info = list(reader)[0]
    #             curr_job_nr = int(info[0]) + 1
    #             curr_parallel_jobs = int(info[1]) + 1
    #             if curr_job_nr >= len(combinations):
    #                 if curr_parallel_jobs == 1:
    #                     removelog = True
    #                     
    #                 f.seek(0)
    #                 f.truncate()
    #                 f.write(str(curr_job_nr) + "," + str(curr_parallel_jobs-1))
    #                 break
    #             
    #             f.seek(0)
    #             f.truncate()
    #             f.write(str(curr_job_nr) + "," + str(curr_parallel_jobs))
    #     
        # 
        # comb_case = combinations[curr_job_nr]
        # # Construct an id string for the current job based on the combination. The string is formatted so that it does
        # # contain any illegal characters for filename and abaqus job name usage.
        # if sys.version[0] == "2":
        #     job_id = str(comb_case).translate(None, ",.&*~!()[]{}|;:\'\"`<>?/\\")
        # else:
        #     job_id = str(comb_case).translate(str.maketrans("", "", ",.&*~!()[]{}|;:\'\"`<>?/\\"))
        # job_id = job_id.replace(" ", "-")
        # job_id = job_id + "-" + prj_name
        # 
        # # Check if the directory exists. If so, continue to the next job.
        # if os.path.isdir("./" + job_id):
        #     print("Job already exists: A directory with the same name'" + job_id + "' exists in the cwd")
        #     continue
        # 
        # if mk_subdirs:
        #     if os.path.isdir(job_id):
        #         os.chdir(job_id)
        #     else:
        #         # Make a new subdirectory for the current session
        #         os.mkdir(job_id)
        # 
        #         # Change working directory
        #         os.chdir(job_id)
        # 
        # # Assemble current job's input arguments
        # if func_args:
        #     current_func_args = func_args
        #     for (index, new_parameter) in zip(p_args, comb_case):
        #         current_func_args[index] = new_parameter
        # else:
        #     current_func_args = []
        # 
        # # Assemble current job's input keyword arguments
        # if func_kargs:
        #     current_func_kargs = func_kargs
        #     for i, x in enumerate(p_kargs):
        #         current_func_kargs[x] = comb_case[len(p_args)+i]
        #     current_func_kargs["IDstring"] = job_id
        # else:
        #     current_func_kargs = {}
        # 
        # # The function to be run for the full factorial parametric is called here
        # print("Running job nr: " + str(curr_job_nr) + " with name: " + job_id)
        # 
        # # Execute the current job
        # try:
        #     job_return = exec_func(*current_func_args, **current_func_kargs)
        # except:
        #     job_return = "FAILED"
        # 
        # # Create an output string
        # return_string = str(job_return)
        # 
        # # Return to parent directory
        # if mk_subdirs is True:
        #     os.chdir('../')
        # 
        # # Open a file to collect the results
        # with flck.FileLock('./' + prj_name + '_info.dat'):
        #     with open('./' + prj_name + '_info.dat', 'a') as out_file:
        #         out_file.write(job_id + "," + str(curr_job_nr) + "," + return_string + "\n")
        # 
        # # Add the result of the current job to the return list
        # prj_results = prj_results + [job_return]
        # 
        # # Remove job's folder (only the output information is kept)
        # if mk_subdirs and del_subdirs is True:
        #     rmtree(job_id)
        #     
        # # Denote that the job is finished on the common .tmp file
        # with flck.FileLock(queue):
        #     with open(queue, "r+") as f:
        #         reader = csv.reader(f)
        #         info = list(reader)[0]
        #         curr_job_nr = int(info[0])
        #         curr_parallel_jobs = int(info[1])
        # 
        #         f.seek(0)
        #         f.truncate()
        #         f.write(str(curr_job_nr) + "," + str(curr_parallel_jobs - 1))
    # 
    # #Return a list of results
    # if removelog:
    #     os.remove(queue)
    # return prj_results


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
    all_dirs.sort()

    for directory in all_dirs:
        os.chdir(directory)
        # Execute the current job
        try:
            func_return = exec_func(*func_args, **func_kargs)
        except:
            func_return = "FAILED"

        os.chdir("..")
        with open('./' + prj_name + '_info.dat', 'a') as out_file:
            out_file.write(directory + "," + str(func_return) + "\n")
