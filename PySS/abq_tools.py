# code to be evaluated for max field output value
# https://gist.github.com/crmccreary/1074551

"""
Module containing tools for Abaqus cae.

"""

import os
import odbAccess
from abaqusConstants import *
import string
import pickle


# definition of a method to search for a keyword position
def get_block_position(model, block_prefix):
    """
    Find a string and return the block number.

    Method to find a given string on the keywords file of a model and return an integer for the position of the first
    occurrence.

    Parameters
    ----------
    model : class
        Abaqus model to search for keyword
    block_prefix : string
        String to look for

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """
    pos = 0
    for block in model.keywordBlock.sieBlocks:
        if string.lower(block[0:len(block_prefix)]) == string.lower(block_prefix):
            return pos
        pos = pos + 1
    return -1


def open_odb(odb_path):
    """
    A more sophisticated open odb function.

    Parameters
    ----------
    odb_path : string
        Path and filename of the database (without the '.odb' extension)

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """

    base, ext = os.path.splitext(odb_path)
    odb_path = base + '.odb'
    if odbAccess.isUpgradeRequiredForOdb(upgradeRequiredOdbPath=odb_path):
        print('odb %s needs upgrading' % (odb_path,))
        path, file_name = os.path.split(odb_path)
        file_name = base + "_upgraded.odb"
        new_odb_path = os.path.join(path, file_name)
        odbAccess.upgradeOdb(existingOdbPath=odb_path, upgradedOdbPath=new_odb_path)
        odb_path = new_odb_path
    odb = odbAccess.openOdb(path=odb_path, readOnly=True)
    return odb


def field_max(odb, result):
    """
    Look for the max value in a field output.

    Scan a field output on an abaqus result database and return the maximum value

    Parameters
    ----------
    odb : class
        Abaqus model containing the field results
    result : class
        Field output result to search for max

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """

    result_field, result_invariant = result
    _max = -1.0e20
    for step in odb.steps.values():
        print('Processing Step:', step.name)
        for frame in step.frames:
            if frame.frameValue > 0.0:
                all_fields = frame.fieldOutputs
                if all_fields.has_key(result_field):
                    stress_set = all_fields[result_field]
                    for stressValue in stress_set.values:
                        if result_invariant:
                            if hasattr(stressValue, result_invariant.lower()):
                                val = getattr(stressValue, result_invariant.lower())
                            else:
                                raise ValueError('Field value does not have invariant %s' % (result_invariant,))
                        else:
                            val = stressValue.data
                        if val > _max:
                            _max = val
                else:
                    raise ValueError('Field output does not have field %s' % (result_field,))
    return _max


def fetch_hist(odb, step_name, node_name, hist_out_name):
    """
    Return a history output from an odb

    Parameters
    ----------
    odb :  str
        odb filename (without the extension)
    step_name :  str
        Name of the step containing the history output
    node_name : str
        Name of the regarded node
    hist_out_name : str
        Name of the history output

    Return
    ------
    tuple

    """
    my_odb = odbAccess.openOdb(path=odb + '.odb')
    step = my_odb.steps[step_name]
    node = step.historyRegions[node_name]
    hist_out = node.historyOutputs[hist_out_name]
    data = hist_out.data
    odbAccess.closeOdb(my_odb)
    return data

# Look for the max value in a history output
# TO BE FIXED. THE REFERENCE POINTS (rp1key, ho1key etc.) ARE NOT GENERIC.
# Fetch maximum load, displacement and LPF for a riks analysis.
# The method assumes that a) the the odb is located in the current directory
def history_max(odb_obj, step_name):
    """
    Look for the max value in a history output.

    Scan a history output on a step on an abaqus result database and return the maximum value.
    Currently, it returns LPF, load and disp history outputs. To be generalised.

    Parameters
    ----------
    odb_obj : Abaqus database object
        Abaqus model filename.
    step_name : str
        Name of the step

    """

    # my_odb = odbAccess.openOdb(path=odb_name)
    riks_step = odb_obj.steps[step_name]
    rp1key = riks_step.historyRegions.keys()[1]
    ho1key = riks_step.historyRegions[rp1key].historyOutputs.keys()[0]
    rp2key = riks_step.historyRegions.keys()[2]
    ho2key = riks_step.historyRegions[rp2key].historyOutputs.keys()[0]
    asskey = riks_step.historyRegions.keys()[0]
    hoasse = riks_step.historyRegions[asskey].historyOutputs.keys()[-1]
    load_hist = riks_step.historyRegions[rp1key].historyOutputs[ho1key].data
    disp_hist = riks_step.historyRegions[rp2key].historyOutputs[ho2key].data
    lpf_hist = riks_step.historyRegions[asskey].historyOutputs[hoasse].data
    maxpos = load_hist.index(max(load_hist, key=lambda x: x[1]))
    load = load_hist[maxpos][1]
    disp = -disp_hist[maxpos][1]
    lpf = lpf_hist[maxpos][1]
    return lpf, load, disp


def find_odb_in_cwd(all=False):

    all_odbs = [i for i in os.listdir(os.getcwd()) if i.split(".")[-1] == "odb"]
    if all:
        return all_odbs
    else:
        return all_odbs[0]


def fetch_eigenv(odb_name, step_name, n_eigen):
    """
    Get eigenvalues.

    Return the eigenvalues of a perturbation buckling analysis from an abaqus database.

    Parameters
    ----------
    odb_name : class
        Abaqus model containing the eigenvalues
    step_name : string
        Name of the step
    n_eigen : int
        Number of eigenvalues to return

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """

    bckl_odb = odbAccess.openOdb(path=odb_name + '.odb')
    bckl_step = bckl_odb.steps[step_name]

    # Gather the eigenvalues
    eigenvalues = ()
    eigen_string = ""
    for J_eigenvalues in range(1, n_eigen + 1):
        current_eigen = float(bckl_step.frames[J_eigenvalues].description.split()[-1])
        eigenvalues = eigenvalues + (current_eigen,)
        eigen_string = eigen_string + "%.3E " % current_eigen

    # Close the odb
    odbAccess.closeOdb(bckl_odb)

    # Return variables
    return eigenvalues, eigen_string


def exporter(data, filename):
    """
    Eaport data to pickle

    Parameters
    ----------
    data
        Variable containing the data to be exported
    filename : str
        Filename of the export file
    """
    with open(filename, "wb") as fh:
        pickle.dump(data, fh)


# def main():
#     """
#     Load FEM data for the 9 tested specimens.
#     """
#     odb = "./016-030-specimens/RIKS-016-030-specimens"
#     step_name = "RIKS"
#     node_name = "Node ASSEMBLY.1"
#     hist_out_name = "RF3"
#
#     sp1 = fetch_hist(odb, step_name, node_name, hist_out_name)