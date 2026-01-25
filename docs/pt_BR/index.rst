.. Arquivo mestre da documentação do Local Inference Calculator

Local Inference Calculator
===========================

Bem-vindo à documentação do **Local Inference Calculator** - uma ferramenta de
capacity planning para inferência local de Large Language Models (LLMs).

Esta ferramenta permite estimar rapidamente quais modelos de linguagem podem
rodar em determinadas GPUs, considerando o tamanho do contexto e a
precisão/quantização dos modelos.

.. toctree::
   :maxdepth: 2
   :caption: Conteúdo:

   instalacao
   uso
   glossario
   api
   exemplos

Visão Geral
===========

O *Local Inference Calculator* foi desenvolvido para responder a uma pergunta simples:

**"Com essa GPU e esse tamanho de contexto, quais modelos LLM consigo rodar?"**

A ferramenta considera:

* **Parâmetros do modelo**: Memória base necessária para armazenar os pesos
* **Overhead**: Memória adicional para runtime, activations, etc.
* **KV Cache**: Memória para cache de attention durante inferência
* **Precisão/Quantização**: FP32, FP16, INT8 ou INT4

Funcionalidades
---------------

* Suporte a modelos de 7B a 180B parâmetros
* Base de dados com 38 GPUs (consumer + datacenter)
* Cálculos conservadores para garantir viabilidade real
* Exportar resultados em JSON e CSV
* Interface de linha de comando (CLI)

Outros Idiomas
===============

* **English**: Execute ``make html LANG=en`` e abra
  ``docs/_build/en/html/index.html``

Índices e Tabelas
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
