\# HTR Project — Handwritten Text Recognition



Deep Learning-Based Handwritten Text Recognition System (Hangzhou Normal University AI Lab, 2026).



\## Project Structure



htr-project/

├── week1\_mnist/      # Week 1: MNIST digit recognition (CNN)

├── week2\_emnist/     # Week 2: EMNIST character recognition

├── week3\_crnn/       # Week 3: CRNN for word-level HTR

├── week4\_survey/     # Week 4: Transformer OCR survey

├── data/             # Datasets (gitignored)

└── runs/             # Model outputs (gitignored)



\## Setup



\### 1. Python 3.11

Download: https://www.python.org/downloads/release/python-3119/



\### 2. Virtual environment

python -m venv venv

venv\\Scripts\\activate       # Windows

source venv/bin/activate    # Linux/macOS



\### 3. Dependencies

pip install torch torchvision torchaudio

pip install numpy matplotlib scikit-learn tqdm seaborn



\## Week 1: MNIST Digit Recognition



\*\*Goal:\*\* Test accuracy ≥ 98%



\*\*Files:\*\*

\- `dataset.py` — MNIST data loaders with normalization

\- `model.py` — SimpleCNN (2 conv layers + 2 FC layers)

\- `train.py` — Training loop with loss/accuracy curves

\- `evaluate.py` — Confusion matrix + error analysis



\*\*Run:\*\*

cd week1\_mnist

python dataset.py    # Verify data loading

python model.py      # Verify model forward pass

python train.py      # Train (10 epochs)

python evaluate.py   # Confusion matrix + error analysis



\*\*Results:\*\*

\- Test accuracy: \*\*99.37%\*\*

\- Loss curves: `runs/curves.png`

\- Confusion matrix: `runs/confusion\_matrix.png`

\- Error analysis: `runs/error\_analysis.png`



\*\*Architecture:\*\*

Conv(1→32, 3x3) → ReLU → MaxPool(2x2) →

Conv(32→64, 3x3) → ReLU → MaxPool(2x2) →

Flatten → FC(3136→128) → ReLU → Dropout(0.5) → FC(128→10)



\*\*Parameters:\*\* 421,642

