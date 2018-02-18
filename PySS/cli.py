# -*- coding: utf-8 -*-

"""Console script for PySS."""

import click


@click.command()
def main(args=None):
    """Console script for PySS."""
    click.echo("Replace this message by putting your code into "
               "PySS.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
