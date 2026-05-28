# CAPP-130: A Corpus of Chinese Application Privacy Policy Summarization and Interpretation

<div align="center">

[![Paper - NeurIPS 2023](https://img.shields.io/badge/Paper-NeurIPS_2023-blue)](https://proceedings.neurips.cc/paper_files/paper/2023/file/92225ec7e87b97a9e007ca6ab7944b14-Paper-Datasets_and_Benchmarks.pdf)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

## 📖 Introduction

A privacy policy serves as an online agreement crafted by service providers, detailing how they collect, process, store, manage, and use personal information when users engage with applications. However, these policies are often filled with technobabble and legalese, making them virtually incomprehensible to the general public. As a result, users frequently consent to terms unknowingly—some of which may even conflict with legal regulations—posing a considerable risk to personal privacy.

To tackle these challenges, we introduce **CAPP-130**, a fine-grained corpus, and **TCSI-pp**, a novel summarization framework:
* **CAPP-130** contains **130** Chinese privacy policies from popular applications that have been carefully annotated and interpreted by legal experts, resulting in **52,489** annotations and **20,555** rewritten sentences.
* **TCSI-pp** first extracts sentences related to the topics specified by users, and then utilizes a generative model to rewrite them into a comprehensible summary. 
* Built upon this framework, we constructed **TCSI-pp-zh**, a summarization tool utilizing RoBERTa (selected from six classification models) for sentence extraction and mT5 (selected from five generative models) for sentence rewriting.

---

## ⚙️ Environment & Requirements

Install the project dependencies via pip:

```bash
pip install -r requirements.txt

```

> **Hardware Recommendation:** 2 × NVIDIA A100 GPUs

---

## 📚 Chinese Application Privacy Policy Corpus (CAPP-130)

CAPP-130 contains 130 carefully annotated Chinese privacy policies, producing 52,489 annotations and 19,570 rewritten sentences.

### Basic Statistics

For detailed annotation guidelines and tags, please refer to our [NeurIPS Paper](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3DOyTIV57Prb) and the Annotation Guidelines ([Chinese Version](https://www.google.com/search?q=Documents/Annotation_Guidelines_Chinese_Version.pdf), [English Version](https://www.google.com/search?q=Documents/Annotation_Guidelines_English_Version.pdf)). *Note: The full English translation of the guidelines is actively being worked on.*

**Table 1: Basic Statistics of Corpus CAPP-130**

| Data Practice Categories | Quantity | Percentage (%) | Median | Mean |
| --- | --- | --- | --- | --- |
| Information Collection | 6,967 | 17.9 | 58 | 70 |
| Permission Acquisition | 1,852 | 4.8 | 54 | 62 |
| Sharing and Disclosure | 4,740 | 12.2 | 52 | 63 |
| Usage | 3,589 | 9.2 | 64 | 75 |
| Storage | 1,360 | 3.5 | 41 | 46 |
| Security Measures | 3,000 | 7.7 | 53 | 60 |
| Special Audiences | 1,416 | 3.6 | 54 | 60 |
| Management | 5,324 | 13.7 | 43 | 49 |
| Contact Information | 712 | 1.8 | 41 | 54 |
| Authorization and Revisions | 1,049 | 2.7 | 35 | 43 |
| Cessation of Operations | 110 | 0.3 | 64 | 68 |
| Important | 20,555 | 52.8 | 52 | 61 |
| Risks | 1,815 | 4.7 | 40 | 46 |

**Table 2: Pre-sliced Data Used to Train TCSI-pp**

| Sub-dataset | Train Samples | Validation Samples | Test Samples |
| --- | --- | --- | --- |
| `important_identification_dataset` | 27,222 | 5,833 | 5,834 |
| `risk_identification_dataset` | 14,338 | 3,083 | 3,084 |
| `topic_identification_dataset` | 14,190 | 3,043 | 3,035 |
| `rewritten_sentences` | 15,656 | 1,957 | 1,957 |

---

## 🛠️ Topic-Controlled Framework (TCSI-pp)

Unlike previous methods that merely extract specific sentences, **TCSI-pp** retrieves relevant sentences based on user-selected data practice topics using a classification model. Subsequently, a generative model rewrites these sentences clearly and concisely for the general public, specifically emphasizing potentially risky sentences.

### 1. Information Extraction

These modules are utilized for binary classification ("Important Identification", "Risk Identification") and multi-class classification ("Topic Identification").

**Model Preparation:**
Place the pre-trained model downloaded from [Hugging Face](https://huggingface.co/) into the `[MODEL_NAME]_pretrain` directory. Ensure it contains the following files:

* `pytorch_model.bin`
* `bert_config.json`
* `vocab.txt`

**Training & Inference:**

```bash
# Train and test binary classification model:
python run.py --model [MODEL_NAME] --data [DATA_NAME]

# Train and test multi-classification model:
python run_multi.py --model [MODEL_NAME] --data [DATA_NAME]

```

**Classification Baselines:**
We acquired benchmarks from six different models (RoBERTa, BERT, mBERT, SBERT, PERT, ERNIE).

*Table 3: Evaluation Metrics (F1-score) for Classification Models*

| Methods | Topic (Micro) | Topic (Macro) | Important (Micro) | Important (Macro) | Risk (Micro) | Risk (Macro) |
| --- | --- | --- | --- | --- | --- | --- |
| RoBERTa | **0.819** | **0.841** | **0.897** | **0.899** | 0.920 | 0.711 |
| BERT | 0.802 | 0.820 | 0.895 | 0.896 | 0.921 | 0.719 |
| mBERT | 0.809 | 0.821 | 0.889 | 0.889 | 0.918 | 0.709 |
| SBERT | 0.781 | 0.794 | 0.875 | 0.874 | 0.917 | 0.689 |
| PERT | 0.801 | 0.812 | 0.895 | 0.897 | **0.922** | **0.716** |
| ERNIE | 0.807 | 0.821 | 0.895 | 0.896 | 0.921 | 0.702 |

> ✨ **New:** All model parameters will be hosted [here](https://www.google.com/search?q=https://huggingface.co/EnlightenedAI/TCSI_pp_zh/tree/main).

### 2. Sentence Rewriting

We fine-tuned mT5, BERT2BERT, BERT2GPT, RoBERTa2GPT, and ERNIE2GPT to rewrite the extracted text into user-friendly summaries.

**Training & Inference:**

```bash
python [MODEL_NAME].py

```

*(Replace `[MODEL_NAME]` with `mT5`, `Bert2Bert`, `Bert2gpt`, `RoBerta2gpt`, or `ERNIE2gpt`)*

**Rewriting Baselines:**

*Table 4: Evaluation metrics for the rewriting models*

| Methods | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | BARTScore | Carburacy |
| --- | --- | --- | --- | --- | --- | --- |
| mT5 | **0.753** | **0.609** | **0.733** | **0.888** | **-4.577** | **0.833** |
| RoBERTa2GPT | 0.749 | 0.577 | 0.719 | 0.872 | -4.975 | 0.755 |
| BERT2BERT | 0.718 | 0.535 | 0.689 | 0.864 | -5.020 | 0.747 |
| BERT2GPT | 0.751 | 0.574 | 0.720 | 0.872 | -4.964 | 0.764 |
| ERNIE2GPT | 0.623 | 0.406 | 0.581 | 0.809 | -5.716 | 0.715 |

---

## 🚀 Summarization Tool (TCSI-pp-zh)

By combining the most effective models (RoBERTa and mT5), we implemented **TCSI-pp-zh**. Experiments on real-world policies demonstrate that TCSI-pp-zh outperforms GPT-4 and other general models, exhibiting higher readability and reliability.

**Run the Tool:**

```bash
python ./TCSI_pp_zh/TCSI_pp_zh.py \
  --binary_model [BINARY_MODEL_NAME] \
  --multi_model [MULTI_MODEL_NAME] \
  --rewrite_model [REWRITE_MODEL_NAME] \
  --topic_list [TOPIC_LIST] \
  --data [PRIVACY_POLICY_PATH]

```

**Demonstration:**
The figure below compares the summarizations generated by GPT-4 and TCSI-pp-zh. Matching background colors represent descriptions of the same policy clauses, while **red text** emphasizes inaccurate content produced in the summary.

---

## 📜 Citation

If you use the data or code from this project, or find our work helpful, please cite our NeurIPS 2023 paper:

```bibtex
@inproceedings{zhu2023capp,
  title={CAPP-130: a corpus of chinese application privacy policy summarization and interpretation},
  author={Zhu, Pengyun and Wen, Long and Liu, Jinfei and Xue, Feng and Lou, Jian and Wang, Zhibo and Ren, Kui},
  booktitle={Proceedings of the 37th International Conference on Neural Information Processing Systems},
  pages={46773--46785},
  year={2023}
}

```
---

## 📌 Update

We will continue to actively maintain and update this repository on GitHub.

```
