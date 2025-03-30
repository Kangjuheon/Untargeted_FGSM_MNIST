## MNIST FGSM Attack

This project implements a simple CNN on the MNIST dataset and evaluates its robustness using FGSM (Fast Gradient Sign Method) adversarial attacks.

## Assignment Instructions

This was done as part of "Trustworthy AI" Assignment.
We implemented the following tasks:

- Training a CNN model on MNIST
- Implementing untargeted FGSM attack
- Measuring clean and adversarial accuracy
- Visualizing FGSM attack examples
- Plotting accuracy vs. epsilon (attack strength)

## File Descriptions

- `test.py` — Main script that trains the model and evaluates FGSM attacks
- `requirements.txt` — List of required Python packages

## How to Run
```bash
#colab 실행시
from google.colab import files
uploaded = files.upload()
!pip install -r requirements.txt
!python test.py
```
```bash
pip install -r requirements.txt
python test.py
