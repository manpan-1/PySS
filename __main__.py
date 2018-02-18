import PySS
import os


def execute(directory):
    PySS.polygonal.main(directory=directory, make_plots=False, export=False)


def main():
    # Ask the user for the path where the assumed file structure with the data is found.
    directory = input('Where are the data files? :')

    # If the user does not provide one, the default is used
    if not directory:
        directory = None
        execute(directory)

    # If the given directory does not exist, print message and terminate.
    if directory:
        if os.path.isdir(directory):
            execute(directory)
        else:
            print('The given directory does not exist.')


if __name__ == "__main__":
    main()
