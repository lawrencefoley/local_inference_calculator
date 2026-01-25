Glossário
=========

Esta seção explica termos técnicos utilizados ao longo da documentação.

This section explains technical terms used throughout the documentation.

Alocador de Memória (Memory Allocator)
--------------------------------------

**PyTorch-style (HF Transformers, vLLM)**

O padrão de alocação de memória usado por frameworks baseados em PyTorch. A memória
é alocada de forma contígua para os pesos do modelo e ativações, com overhead adicional
para operações CUDA. Este alocador tende a ser conservador no gerenciamento de memória,
reservando mais VRAM do que estritamente necessário para evitar fragmentação e garantir
operação suave.

Características dos frameworks:

* **Hugging Face Transformers**: Usa o alocador CUDA padrão do PyTorch com overhead
  adicional para carregamento do modelo e passos de otimização
* **vLLM**: Implementa PagedAttention com gerenciamento de memória mais eficiente,
  mas ainda segue padrões de alocação PyTorch-style para pesos do modelo

**Nota: TensorRT-LLM, llama.cpp, EXL2**

Estes frameworks usam estratégias de alocação de memória diferentes que podem resultar
em menor uso de VRAM:

* **TensorRT-LLM**: Usa alocadores de memória customizados otimizados para engines
  TensorRT, com pooling de memória mais agressivo e menos overhead
* **llama.cpp (GGUF)**: Usa alocação aware de quantização com arquivos memory-mapped,
  frequentemente requerendo 20-30% menos VRAM que equivalentes PyTorch
* **EXL2 (ExLlamaV2)**: Implementa alocação de memória extremamente otimizada com
  overhead mínimo, suportando quantização de taxa variável por camada

Ao calcular requisitos de VRAM, esta ferramenta assume alocação PyTorch-style para
consistência. Se usando esses frameworks alternativos, o uso real de VRAM pode ser
10-30% menor que o estimado.

Adaptadores LoRA
----------------

**Low-Rank Adaptation** é uma técnica de fine-tuning eficiente em parâmetros que adiciona
matrizes de decomposição de posto treináveis aos pesos existentes do modelo em vez de
fazer fine-tuning de todos os parâmetros.

Impacto na memória:

* **Por adaptador**: +0.5 a 2 GB de VRAM dependendo do rank e tamanho do modelo
* **Rank 8-16**: Configuração típica, ~0.5-1 GB de overhead
* **Rank 32-64**: Adaptadores de maior capacidade, ~1-2 GB de overhead
* **Múltiplos adaptadores**: A memória escala linearmente com adaptadores ativos

Impacto no cálculo de VRAM:

.. code-block:: python

   # Com LoRA, adicione memória do adaptador ao total
   total_vram = modelo_base + kv_cache + overhead_lora_adapter

A ferramenta assume nenhum adaptador LoRA ativo por padrão. Se usando LoRA,
adicione 0.5-2 GB por adaptador aos requisitos de VRAM estimados.

Decodificação Especulativa (Speculative Decoding)
-------------------------------------------------

Também conhecida como **draft decoding** ou **assistant decoding**, esta técnica usa um
modelo draft menor para prever tokens que são então verificados pelo modelo alvo.

Impacto na memória:

* **Overhead do modelo draft**: +30-50% de VRAM adicional para o modelo draft
* **Exemplo**: Usar um modelo draft de 7B com um alvo de 70B requer memória para
  ambos os modelos
* **Tamanho típico de draft**: 10-20% dos parâmetros do modelo alvo

Trade-offs:

* **Vantagens**: Geração de tokens 2-3x mais rápida
* **Desvantagens**: Uso de memória significativamente maior, requer dois modelos na memória

A ferramenta assume nenhuma decodificação especulativa. Se habilitada, calcule a memória
para ambos os modelos e adicione 10-15% para overhead de verificação.

KV Cache Paginado Agressivo (Aggressive Paged KV)
-------------------------------------------------

**Paged KV Cache** é uma técnica de gerenciamento de memória inspirada em paging de
sistema operacional, onde o cache KV é dividido em blocos de tamanho fixo (páginas)
que podem ser alocados e desalocados dinamicamente.

Variantes de implementação:

* **Paged KV padrão**: Implementação padrão do vLLM, uso de memória eficiente
* **Paged KV agressivo**: Estratégias mais agressivas de pré-alocação e cache

Impacto na memória:

* **Padrão**: Redução de 10-20% na memória de KV cache vs alocação contígua
* **Agressivo**: Economias adicionais de 5-10% mas maior overhead de CPU para
  gerenciamento de páginas

A ferramenta usa estimativas conservadoras de KV cache. Implementações de paged KV
agressivo podem usar 10-30% menos memória que o calculado para o componente de KV cache.

KV Cache (Cache Key-Value)
--------------------------

O **KV cache** armazena chaves e valores de atenção para todos os tokens gerados
anteriormente em uma sequência, evitando recomputação durante a geração autoregressiva.

Cálculo de memória:

.. code-block:: python

   kv_cache_memory = 2 × num_layers × hidden_dim × context_length × bytes_per_element

Componentes:

* **Keys (Chaves)**: Cache para chaves de atenção (projeções de tokens de consulta)
* **Values (Valores)**: Cache para valores de atenção (conteúdo para atender)
* **Context length**: Escalonamento linear - 2x contexto = 2x memória de KV cache

Impacto da precisão:

* **KV FP32**: 4 bytes por elemento (raramente usado)
* **KV FP16/BF16**: 2 bytes por elemento (padrão)
* **KV INT8**: 1 byte por elemento (experimental, suporte limitado)
* **KV FP8**: 1 byte por elemento (GPUs mais recentes apenas, H100+)

Escalonamento de Contexto (Context Scaling)
--------------------------------------------

A memória de KV cache escala **linearmente** com o tamanho do contexto, tornando
cenários de contexto longo extremamente intensivos em memória.

Exemplos de escalonamento (modelo 7B, FP16):

+------------+--------------+------------------+
| Contexto   | KV Cache     | VRAM Total       |
+============+==============+==================+
| 4K tokens  | ~2.1 GB      | ~18 GB           |
+------------+--------------+------------------+
| 8K tokens  | ~4.2 GB      | ~20 GB           |
+------------+--------------+------------------+
| 16K tokens | ~8.4 GB      | ~24 GB           |
+------------+--------------+------------------+
| 32K tokens | ~16.8 GB     | ~33 GB           |
+------------+--------------+------------------+
| 128K tokens| ~67.2 GB     | ~83 GB           |
+------------+--------------+------------------+

Implicações práticas:

* **4K → 8K**: +2.1 GB (gerenciável)
* **8K → 32K**: +12.6 GB (pode exceder limites de GPUs consumer)
* **Modelos de contexto longo**: Tipicamente requer multi-GPU ou quantização

A ferramenta exibe KV cache projetado para contextos estendidos (16K, 32K, 128K)
para ajudar no planejamento de cenários que requerem janelas de contexto maiores.

Termos Adicionais
-----------------

**Batch Size**

Número de sequências processadas simultaneamente. A memória escala linearmente com
o batch size. Esta ferramenta assume batch_size=1. Serving em produção com
batch_size>8 requer 2-10x mais VRAM.

**Tensor Parallelismo**

Divisão de pesos do modelo através de múltiplas GPUs. Cada GPU armazena uma porção
de cada camada. Permite rodar modelos muito grandes para uma única GPU, mas requer
interconexões NVLink de alta velocidade.

**Pipeline Parallelismo**

Divisão de camadas do modelo através de múltiplas GPUs. Cada GPU armazena camadas
completas consecutivas. Menor overhead de comunicação que tensor parallelismo mas
introduz pipeline bubbles.

**Flash Attention**

Um algoritmo de atenção que reduz o uso de memória materializando a matriz de atenção
em blocos em vez de tudo de uma vez. Proporciona redução de 2-4x na memória de
ativações durante treinamento, mas impacto limitado no VRAM de inferência (que é
dominado por pesos e KV cache).

**Quantização (AWQ, GPTQ, NF4)**

Métodos alternativos de quantização com diferentes trade-offs:

* **AWQ (Activation-aware Weight Quantization)**: Otimiza para distribuição de
  ativações, melhor desempenho em INT4
* **GPTQ**: Quantização pós-treinamento, boa compressão mas pode requerer calibração
* **NF4 (4-bit NormalFloat)**: Quantização desenhada para pesos normalmente distribuídos,
  usado em QLoRA

**ACT (Activation Checkpointing)**

Trocando computação por memória recomputando ativações durante o backward pass.
Reduz memória durante treinamento mas impacto mínimo no VRAM de inferência.

**Mixture of Experts (MoE)**

Modelos onde apenas um subconjunto de parâmetros (experts) são ativados por token.
Modelos MoE como Mixtral 8x7B têm 47B parâmetros totais mas usam apenas ~13B por
forward pass, reduzindo requisitos de memória ativa.

**RoPE Scaling (Rotary Position Embedding)**

Técnicas para estender o tamanho do contexto além dos limites de treinamento:
* **Interpolação linear**: Extrapola posições suavemente
* **YaRN (Yet another RoPE extensioN)**: Interpolação NTK-aware
* **NTK Dinâmico**: Ajusta interpolação baseado na posição

Estes afetam a qualidade do modelo mas não o cálculo de VRAM diretamente.
