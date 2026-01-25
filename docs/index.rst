.. Local Inference Calculator documentation master file

Local Inference Calculator
===========================

Welcome to the **Local Inference Calculator** documentation - a capacity planning
tool for local Large Language Model (LLM) inference.

Bem-vindo à documentação do **Local Inference Calculator** - uma ferramenta de
capacity planning para inferência local de Large Language Models (LLMs).

This tool allows you to quickly estimate which language models can run on specific
GPUs, considering context size and model precision/quantization.

Esta ferramenta permite estimar rapidamente quais modelos de linguagem podem rodar
em determinadas GPUs, considerando o tamanho do contexto e a precisão/quantização
dos modelos.

.. toctree::
   :maxdepth: 2
   :caption: Contents / Conteúdo:

   installation
   usage
   api
   examples

Overview
========

The *Local Inference Calculator* was developed to answer a simple question:

O *Local Inference Calculator* foi desenvolvido para responder a uma pergunta simples:

**"With this GPU and this context size, which LLMs can I run?"**

**"Com essa GPU e esse tamanho de contexto, quais modelos LLM consigo rodar?"**

The tool considers:

A ferramenta considera:

* **Model parameters**: Base memory required to store weights
* **Parâmetros do modelo**: Memória base necessária para armazenar os pesos

* **Overhead**: Additional memory for runtime, activations, etc.
* **Overhead**: Memória adicional para runtime, activations, etc.

* **KV Cache**: Memory for attention cache during inference
* **KV Cache**: Memória para cache de attention durante inferência

* **Precision/Quantization**: FP32, FP16, INT8, or INT4
* **Precisão/Quantização**: FP32, FP16, INT8 ou INT4

Features
--------

* Support for models from 7B to 180B parameters
* Suporte a modelos de 7B a 180B parâmetros

* Database with 38 GPUs (consumer + datacenter)
* Base de dados com 38 GPUs (consumer + datacenter)

* Conservative calculations to ensure real-world viability
* Cálculos conservadores para garantir viabilidade real

* Export results to JSON and CSV
* Exportar resultados em JSON e CSV

* Command-line interface (CLI)
* Interface de linha de comando (CLI)

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
