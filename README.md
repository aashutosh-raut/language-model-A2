# LSTM Language Model Demo (Flask)

This project demonstrates a **word-level LSTM language model** trained on *Lord of the Rings movie dialogues* and deployed as a **Flask web application**.  
Users can enter a text prompt through a web interface and receive a generated continuation produced by the trained model.

The project integrates **PyTorch**, **Flask**, and a saved vocabulary (`vocab.pkl`) to ensure inference behavior is consistent with training.


## Overview


The model is trained separately in a Jupyter notebook and loaded once at application startup for efficient inference.

## Dataset Description

The dataset used in this project consists of **Lord of the Rings movie dialogues**, sourced from Kaggle.  
It contains conversational text spoken by characters across the trilogy, making it suitable for training
a **word-level language model** due to its narrative structure, recurring entities, and varied sentence lengths.

Before training, the text was cleaned and normalized by lowercasing, filtering to alphanumeric characters
and apostrophes, and tokenizing at the word level. Rare words were mapped to an `<unk>` token, and an
`<eos>` token was appended to mark sentence boundaries.

---


## Features



## Project Structure
```
A2/
├── app/
│ ├── app.py # Flask application
│ ├── model_utils.py # Model, vocab loading, tokenizer, generation
│ ├── lotr-lstm_lm.pt # Trained model checkpoint
│ ├── vocab.pkl # Saved vocabulary (itos list)
│ ├── templates/
│ │ └── index.html # Web UI
│
├── LSTM LM.ipynb # Training notebook
├── lotr_scripts.csv # Dataset
├── requirements.txt # Python dependencies
└── README.md
```


## Requirements


### Core dependencies

### Install dependencies (recommended: virtual environment)

#### Windows (PowerShell)

```powershell
python -m venv env
./env/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
If you have a CUDA-capable GPU, install the appropriate PyTorch build from:
https://pytorch.org/get-started/locally/

Model & Vocabulary
Training
The model is trained in LSTM LM.ipynb

Tokens are:

Lowercased

Filtered to [a-z0-9']

Words occurring fewer than 3 times are mapped to <unk>

<eos> is appended to mark sentence boundaries

Vocabulary size during training: 884 tokens

Saved Files
Model checkpoint: lotr-lstm_lm.pt

Vocabulary: vocab.pkl (pickled itos list)

These two files must match.
The app does not retrain the model.

Running the Application
From the app/ directory:

python app.py
The app runs on:

http://localhost:5000/
or, when deployed behind JupyterHub / Puffer:

/user/<username>/proxy/5000/
Web UI Usage
Open the application in a browser

Enter a text prompt (e.g., bilbo baggins is)

Click Generate

The generated continuation appears below the form

Generation Settings
In app.py:

generate_text(
    prompt,
    model,
    vocab,
    device,
    max_len=40,
    temperature=0.8
)
max_len: Maximum number of generated tokens

temperature:

Lower → more deterministic

Higher → more diverse output

Implementation Details
Model Architecture
Defined in model_utils.py:

Embedding size: 512

Hidden size: 512

LSTM layers: 2

Dropout: 0.65

This configuration must exactly match training, otherwise loading the checkpoint will fail.

Vocabulary Handling (Important)
vocab.pkl is loaded at startup

Vocabulary size is inferred dynamically using:

len(itos)
No hardcoded vocab sizes (e.g., 884 or 906) are used

This guarantees index alignment between the model and tokens

Troubleshooting
Model file not found
Ensure this file exists in app/:

lotr-lstm_lm.pt

Vocabulary mismatch
Ensure vocab.pkl was generated during training

Do not recreate a dummy vocabulary

CUDA issues
The app automatically falls back to CPU if CUDA is unavailable

Port already in use
Change the port in app.py:

app.run(host="0.0.0.0", port=5001)
Limitations
Word-level modeling leads to <unk> usage for unseen words

LSTM struggles with long-range coherence

Small dataset risks memorization

No attention mechanism

Despite these limitations, the project effectively demonstrates sequence modeling, truncated BPTT, and deployment of neural language models.

### Conclusion
This assignment implements a complete pipeline:





## Flask-based deployment

The project highlights both the strengths and limitations of classical RNN-based language models and provides a solid foundation for understanding modern NLP systems

# LSTM Language Model Demo (Flask)

A concise, production-friendly README for a Flask app that serves a PyTorch LSTM language model trained on Lord of the Rings movie dialogues. Users enter a prompt and receive a generated continuation via the web UI.

---

## Overview

- Model: Word-level LSTM Language Model (PyTorch)
- Backend: Flask (+ `ProxyFix` for reverse-proxy compatibility)
- Frontend: Jinja2 HTML form
- Checkpoints: `lotr_lstm_lm.pt` (preferred) or `best-val-lstm_lm.pt` (fallback)
- Optional vocab: `vocab.pkl` (pickled `itos` list)

The model is trained outside this app and loaded once at startup for efficient inference.

---

## Features

- Loads a pretrained LSTM only once at startup
- Simple tokenizer aligned with training: lowercase, keep `[a-z0-9']`, whitespace split
- Temperature-controlled sampling; early stop on `<eos>`
- Reverse-proxy friendly via `ProxyFix`

---

## Project Structure (app folder)

- [app.py](app.py): Flask entry point and route (`/`) for GET/POST
- [model_utils.py](model_utils.py): Model class, tokenizer, checkpoint/vocab loading, generation
- [templates/index.html](templates/index.html): Web UI template
- [lotr_lstm_lm.pt](lotr_lstm_lm.pt): Trained model checkpoint (preferred name)
- [best-val-lstm_lm.pt](best-val-lstm_lm.pt): Alternate checkpoint (fallback)
- [requirements.txt](requirements.txt): Pinned Python dependencies
- `vocab.pkl`: Optional saved vocabulary (`itos` list)

If your broader repo includes training assets (e.g., a notebook and dataset), document them in the top-level README of that project.

---

## Installation

Requirements: Python 3.9+

Windows (PowerShell), using a virtual environment and requirements file:

```powershell
python -m venv env
./env/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

CUDA note: If you have a CUDA-capable GPU, you may need a CUDA-specific PyTorch build from https://pytorch.org/get-started/locally/. Adjust installation accordingly (e.g., uninstall/reinstall `torch` with the index URL recommended by PyTorch).

---

## Quickstart

From the `app` directory, start the server:

```powershell
python app.py
```

Open in your browser:

- http://localhost:5000/

When deployed behind a reverse proxy (e.g., JupyterHub/Puffer), `ProxyFix` helps the app respect upstream headers.

---

## Usage

1. Open the app in a browser.
2. Enter a text prompt (e.g., "bilbo baggins is").
3. Click Generate.
4. The continuation appears below the form.

Generation parameters used by the app:

```python
generate_text(
        prompt,
        model,
        vocab,
        device,
        max_len=40,
        temperature=0.8,
)
```

- `max_len`: Maximum number of generated tokens
- `temperature`: Lower → more deterministic; higher → more diverse

---

## Model & Vocabulary

- Startup search order for checkpoints: `lotr_lstm_lm.pt`, then fallback to `best-val-lstm_lm.pt` if present.
- If `vocab.pkl` exists, it is loaded and reconciled to the checkpoint’s embedding size. If absent, a placeholder vocabulary is created to match the checkpoint dimensions so inference can proceed. For best quality, provide the exact training vocabulary.
- Unknown tokens map to the default index (commonly `<unk>` in the training vocab).

---

## Implementation Details

- Architecture in `LSTMLanguageModel` (see [model_utils.py](model_utils.py)):
    - Embedding size: 512
    - Hidden size: 512
    - LSTM layers: 2
    - Dropout: 0.65
- This must match the training configuration for the checkpoint to load correctly.
- Tokenizer: lowercase, keep `[a-z0-9']`, whitespace split (see `tokenizer()` in [model_utils.py](model_utils.py)).

---

## Troubleshooting

- Model file not found: Ensure one of [lotr_lstm_lm.pt](lotr_lstm_lm.pt) or [best-val-lstm_lm.pt](best-val-lstm_lm.pt) exists alongside the code.
- Vocab issues: Provide `vocab.pkl` from training to maximize quality; otherwise a placeholder vocab is used to match checkpoint size.
- CUDA/CPU: The app auto-selects CUDA if available, else CPU. Install a matching PyTorch build.
- Port in use: Change the port in [app.py](app.py) via `app.run(host="0.0.0.0", port=5001)`.

---

## Limitations

- Word-level modeling yields `<unk>` for unseen words
- LSTM may struggle with long-range coherence
- Small datasets risk memorization
- No attention mechanism

Despite these limitations, the app demonstrates sequence modeling and the deployment of neural language models in a minimal, reproducible setup.

---

## Next Steps

- Add an API endpoint for JSON generation requests
- Log prompts and generations for analysis
- Parameterize `max_len` and `temperature` via the UI

## Training Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Model Type     | LSTM Language Model |
| Embedding Size | 512 |
| Hidden Size    | 512 |
| LSTM Layers    | 2 |
| Dropout        | 0.65 |
| Optimizer      | Adam |
| Learning Rate  | 0.001 |
| Batch Size     | 64 |
| Epochs         | 20 |
| Loss Function  | Cross-Entropy Loss |

## Conclusion

This project demonstrates an end-to-end pipeline for building and deploying a classical
neural language model. A word-level LSTM was trained on movie dialogue data, serialized, and
successfully integrated into a Flask web application for interactive text generation.

While LSTM-based language models are limited in handling long-range dependencies compared to
modern transformer architectures, this assignment effectively illustrates core NLP concepts
such as sequence modeling, vocabulary construction, truncated backpropagation through time,
and model deployment. The resulting system provides a clear and practical foundation for
understanding how neural language models are trained and served in real-world applications.