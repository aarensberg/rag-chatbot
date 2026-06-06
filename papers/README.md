# PDF Corpus — History of AI, LLMs, RAG, and Agents

This folder contains the document corpus used to build a RAG chatbot focused on the history of ideas, architectures, and methods that progressively led to modern large language models and agentic systems. The corpus is intentionally composed only of **primary-source PDF research papers**, so that indexing, retrieval, reranking, and answer generation stay grounded in a clear and auditable source base.

## Corpus purpose

This corpus was assembled for a specific goal: building a chatbot that can answer questions about:

- the theoretical foundations of artificial intelligence;
- the evolution of neural networks and deep learning;
- the emergence of distributed language representations;
- the rise of attention, transformers, and large language models;
- the research lineage behind RAG, reasoning-oriented prompting, and tool-using agents.

The objective is not to cover the entire history of AI, but to create a **compact, technically coherent timeline** that works well in a RAG setting.

## Scope

The corpus mainly covers five broad families of papers:

1. **AI and neural network foundations** — Turing, perceptron, backpropagation, sequential memory.
2. **Language representation** — word embeddings and distributed vector representations.
3. **Sequence modeling and attention** — seq2seq, attention, and the transition away from pure RNN/LSTM pipelines.
4. **LLMs and scaling** — Transformer, BERT, GPT-3, GPT-4.
5. **RAG, alignment, and agents** — RAG, RLHF, Chain-of-Thought, ReAct.

## Naming convention

Each file follows a stable naming pattern:

`YYYY-Title.pdf`

This naming convention makes it easier to:

- sort papers chronologically;
- reconstruct a timeline;
- display readable sources in the chatbot UI;
- attach metadata such as `year`, `title`, `category`, and `tags` during indexing.

## Document list

| Year | File | Main topic | Role in the corpus |
|---|---|---|---|
| 1950 | `1950-Computing_Machinery_and_Intelligence.pdf` | AI foundations | Foundational text on whether machines can think. |
| 1958 | `1958-The_Perceptron.pdf` | Neural networks | Early formal basis for trainable artificial neurons. |
| 1986 | `1986-Learning_representations_by_back‑propagating_errors.pdf` | Deep learning | Makes multilayer learning practically usable. |
| 1997 | `1997-Long_Short-Term_Memory.pdf` | Sequence memory | Key step toward learning long-range dependencies. |
| 2013 | `2013-Efficient_Estimation_of_Word_Representations_in_Vector_Space.pdf` | Embeddings | Core paper for distributed word vectors. |
| 2014 | `2014-Sequence_to_Sequence_Learning_with_Neural_Networks.pdf` | Seq2Seq | Establishes the encoder-decoder paradigm. |
| 2014 | `2014-Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate.pdf` | Attention | Introduces modern attention in neural translation. |
| 2017 | `2017-Attention_Is_All_You_Need.pdf` | Transformer | Direct foundation of modern LLM architecture. |
| 2018 | `2018-BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding.pdf` | Pretraining | Shows the power of bidirectional Transformer pretraining. |
| 2020 | `2020-Language_Models_are_Few-Shot_Learners.pdf` | LLM scaling | Establishes the large-scale few-shot paradigm. |
| 2020 | `2020-Retrieval‑Augmented_Generation_for_Knowledge‑Intensive_NLP_Tasks.pdf` | RAG | Core paper for retrieval + generation workflows. |
| 2022 | `2022-Training_language_models_to_follow_instructions_with_human_feedback.pdf` | Alignment / RLHF | Foundation of modern instruction-following behavior. |
| 2022 | `2022-Chain‑of‑Thought_Prompting_Elicits_Reasoning_in_Large_Language_Models.pdf` | Reasoning | Important for step-by-step reasoning behavior. |
| 2022 | `2022-ReAct_Synergizing_Reasoning_and_Acting_in_Language_Models.pdf` | Agents | Connects reasoning, acting, and tool use. |
| 2023 | `2023-GPT‑4_Technical_Report.pdf` | Advanced LLMs | Current endpoint of the selected historical timeline. |

## Recommended metadata for indexing

For a clean RAG pipeline, each chunk should ideally keep at least the following metadata:

- `source`: PDF filename;
- `title`: paper title;
- `year`: publication year;
- `category`: high-level topic family;
- `tags`: short list of keywords;
- `authors`: main authors;
- `chunk_id`: unique chunk identifier.

These metadata fields make it easier to:

- display provenance in answers;
- filter by period or topic family (`foundations`, `embeddings`, `transformers`, `rag`, `agents`);
- evaluate retrieval quality;
- later add a Streamlit filter by source or category.

## Suggested chatbot behavior

The chatbot should answer **from the indexed corpus**, not from uncontrolled general knowledge. In practice, this means:

- retrieve the most relevant chunks from Chroma first;
- rerank the candidates with a cross-encoder;
- pass only the best chunks to the LLM;
- display the most relevant paper sources in the UI.

This corpus is particularly suitable for questions such as:

- “Which paper introduced modern attention?”
- “What is the difference between Word2Vec and BERT?”
- “Which papers matter most to understand RAG?”
- “How do we get from seq2seq to the Transformer?”
- “Which papers prepared the ground for coding agents like Claude Code?”

## Corpus limitations

The corpus is intentionally **small, curated, and historical**. It is well suited for an academic assignment, for RAG experiments, and for an educational chatbot, but it does not cover:

- the full history of machine learning;
- the complete LLM literature;
- systems and inference engineering papers for production deployments;
- the most recent benchmark-heavy literature beyond the selected set.

If the project grows, natural extensions would be:

- adding papers on modern embeddings and bi-encoders;
- adding papers on long-context methods, tool use, and function calling;
- adding a few recent surveys on agents and RAG evaluation.

## Associated files

- `manifest.json`: structured metadata for programmatic ingestion and filtering.

## Submission note

For the final assignment submission, the instructions explicitly say **not to include the PDFs, the Chroma database, or any API keys** in the final repository or zip file. Keeping this documentation is still useful because it describes the corpus even when the PDF files themselves are excluded from the final submission.
