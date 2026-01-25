.. Local Inference Calculator documentation master file

Local Inference Calculator
===========================

Welcome to the **Local Inference Calculator** documentation - a capacity planning
tool for local Large Language Model (LLM) inference.

This tool allows you to quickly estimate which language models can run on specific
GPUs, considering context size and model precision/quantization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   glossary
   api
   examples

Overview
========

The *Local Inference Calculator* was developed to answer a simple question:

**"With this GPU and this context size, which LLMs can I run?"**

The tool considers:

* **Model parameters**: Base memory required to store weights
* **Overhead**: Additional memory for runtime, activations, etc.
* **KV Cache**: Memory for attention cache during inference
* **Precision/Quantization**: FP32, FP16, INT8, or INT4

Features
--------

* Support for models from 7B to 180B parameters
* Database with 38 GPUs (consumer + datacenter)
* Conservative calculations to ensure real-world viability
* Export results to JSON and CSV
* Command-line interface (CLI)

Other Languages
===============

* **Português (Brazil)**: Run ``make html LANG=pt_BR`` and open
  ``docs/_build/pt_BR/html/index.html``

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
