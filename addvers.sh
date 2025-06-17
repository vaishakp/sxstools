#!/bin/bash

#echo $PWD
CWD=$(basename $(pwd))
#echo ${CWD}
/usr/bin/env python .generate_version.py
git add ${CWD}/__init__.py
git add ${PWD}/setup.py
git add ${PWD}/public/version
git add ${PWD}/docs/vers_badge.svg
