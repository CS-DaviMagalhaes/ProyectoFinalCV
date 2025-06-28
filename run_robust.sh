#!/bin/bash
# Wrapper script para ejecutar el demo robusto con PYTHONPATH correcto

export PYTHONPATH="src:$PYTHONPATH"
python run_robust_demo.py "$@" 