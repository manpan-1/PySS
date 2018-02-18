import PySS
import os

# Ask the user for the path where the assumed file structure with the data is found.
directory = input('Where are the data files? :')

# If the user does not provide one, the default is used

def execute():
    PySS.polygonal.main(directory=directory, make_plots=False, export=False)

if not directory:
    directory = None
    execute()

# If the given directory does not exist, terminate.
if directory:
    if os.path.isdir(directory):
        execute()
    else:
        print('The given directory does not exist.')
