#!/usr/bin/env python

import argparse

import fishact

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate data files.')
    parser.add_argument('activity_fname', metavar='activity_file', type=str,
                        help='Name of activity file.')
    parser.add_argument('gtype_fname', metavar='genotype_file', type=str,
                        help='Name of genotype file.')

    args = parser.parse_args()

    print('------------------------------------------------')
    print('Checking genotype file...')
    fishact.validate.test_genotype_file(args.gtype_fname)
    print('------------------------------------------------\n\n\n')
    print('------------------------------------------------')
    print('Checking activity file...')
    fishact.validate.test_activity_file(args.activity_fname, args.gtype_fname)
    print('------------------------------------------------')
