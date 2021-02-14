"""CLI for the IP model.

This script optionally supports uploading solution files to Dropbox, which is
helpful when running on AWS. Access tokens can be generated via the developer
console: https://www.dropbox.com/developers/apps
"""
import os
import sys
import click
from time import time
from model import optimize_makespan_k_ways
from cgshop2021_pyutils import InstanceDatabase, SolutionZipWriter, Solution
try:
    import dropbox  # optional
except ImportError:
    dropbox = None


@click.command()
@click.option('--db',
              required=True,
              help='Path of the SoCG 2021 instances ZIP.')
@click.option('--name', required=True, help='Name of the instance to solve.')
@click.option('--time-limit',
              type=float,
              default=300,
              help='Time limit for an IP subproblem.')
@click.option('--buffer',
              type=int,
              default=2,
              help='Margin around the instance core.')
@click.option('--verbose', is_flag=True)
@click.option('--no-polish', is_flag=True, help='Skip final binary search.')
@click.option('-k',
              multiple=True,
              type=int,
              help='Number of ways to split the problem.')
@click.option('--n-threads',
              type=int,
              default=0,
              help='Number of threads to use when solving IPs (default: all).')
@click.option('--out-file',
              default='.',
              help='Path of the local output ZIP file.')
@click.option('--dropbox-access-token')
@click.option('--dropbox-out-file')
def main(db, name, time_limit, buffer, verbose, no_polish, k, n_threads,
         out_file, dropbox_access_token, dropbox_out_file):
    if not k:
        k = [1]  # don't split by default
    k = sorted(k)
    idb = InstanceDatabase(db)
    instance = idb[name]

    dbx = None
    if dropbox_access_token and dropbox_out_file:
        if dropbox is None:
            print('Warning: Dropbox client not installed.', file=sys.stderr)
        else:
            # see https://stackoverflow.com/a/36851978
            dbx = dropbox.Dropbox(dropbox_access_token)

    for way in k:
        tic = time()
        print('Solving', name, f'with k={way}...')
        sol = optimize_makespan_k_ways(instance=instance,
                                       buffer=buffer,
                                       k=way,
                                       time_limit=time_limit,
                                       polish=not no_polish,
                                       verbose=verbose,
                                       n_threads=n_threads)
        toc = time()
        if isinstance(sol, Solution):
            print('Solved {} in {:.2f} seconds (k={:d}).'.format(
                name, toc - tic, way))
            with SolutionZipWriter(out_file) as szw:
                szw.add_solution(sol)
            if dbx and dropbox_out_file:
                with open(out_file, 'rb') as f:
                    dbx.files_upload(f.read(), dropbox_out_file)
            break  # no need to try a larger number of splits
        else:
            print('Failed to solve {} in {:.2f} seconds (k={:d}).'.format(
                name, toc - tic, way))


if __name__ == '__main__':
    main()
