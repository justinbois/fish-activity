#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    # TODO: put package requirements here
]

setup_requirements = [
    'pytest-runner',
    # TODO(justinbois): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='fish_activity',
    version='0.1.0',
    description="Parsing and plotting tools for fish activity assays.",
    long_description=readme + '\n\n' + history,
    author="Justin Bois",
    author_email='bois@caltech.edu',
    url='https://github.com/justinbois/fish-activity',
    packages=find_packages(include=['fishact',
                                    'fishact.parse',
                                    'fishact.summarize',
                                    'fishact.validate',
                                    'fishact.visualize']),
    entry_points={
        'console_scripts': [
            'fish_activity=fish_activity.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='fish_activity',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
