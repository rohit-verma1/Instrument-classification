# Audio Classification Project README

## Overview
This project involves the classification of musical instruments in audio files using a Convolutional Neural Network (CNN) trained on mel spectrograms. The project is organized into several components:

1. **Data Loader:** [`datav2.py`](datav2.py) contains the data loader class responsible for loading audio files.

2. **CNN Architecture:** [`modelv2.py`](modelv2.py) includes the CNN architecture used for instrument classification.

3. **Training:** [`trainv2.py`](trainv2.py) is where the model was trained using actual labels as targets and mel spectrograms as inputs. The trained model is saved as `feedforwardnet.pth`.

4. **Inference:** [`inference.py`](inference.py) evaluates the trained model and generates inferences. The expected and predicted values are mapped in a file called `inference_results.csv`. An updated annotations file (`updated_metadata.csv`) containing pseudo labels is created based on these inferences.

5. **Training with Pseudo Labels:** [`train_pseudo.py`](train_pseudo.py) involves training the model again, but this time using the pseudo labels obtained from the previous step. The trained model is saved as `pseudo_model.pth`.

6. **Inference with Pseudo Labels:** [`inference_pseudo.py`](inference_pseudo.py) evaluates the model trained with pseudo labels, and the inferences are compared against both actual labels and pseudo labels. The accuracy is recorded for both scenarios.

## Project Structure
The project directory structure is as follows:

```plaintext
- datav2.py
- modelv2.py
- trainv2.py
- inference.py
- updated_metadata.csv
- train_pseudo.py
- pseudo_model.pth
- inference_pseudo.py
- README.md
