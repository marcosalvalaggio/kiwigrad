#!/bin/bash

if [ -d "build" ]; then
  rm -rf build
fi

if [ -d "dist" ]; then
  rm -rf dist
fi

if [ -d "lanablas.egg-info" ]; then
  rm -rf lanablas.egg-info
fi

python3 setup.py sdist bdist_wheel --universal