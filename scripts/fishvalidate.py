#!/usr/bin/env python

import argparse

import fishact

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate data files.')
    parser.add_argument('fname', metavar='filename', type=str,
                        help='Name of file to be validated.')
    parser.add_argument('--gtype', '-g', action='store_true', dest='gtype',
        help="Flag if file being validated is a genotype file; otherwise activity file.")
    args = parser.parse_args()

    if args.gtype:
        fishact.validate.test_genotype_file(args.fname)
    else:
        fishact.validate.test_activity_file(args.fname, args.gtype)
