# DeepGlobe-Road-Extraction-Challenge
Code for the 1st place solution in [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467).

# Requirements

- Cuda 8.0
- Python 3.7
- Pytorch 0.2.0
- cv2

# Usage

### Data
Place '*train*', '*valid*' and '*test*' data folders in the '*dataset*' folder.

Data is from [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467#participate-get_starting_kit). You should sign in first to get the data.

### Train
- Run `python train.py` to train the default D-LinkNet34.
- Run `python train.py location_of_pretrained_model_file_to_start_training_from_the_given_point`
### Example
 `python train.py /content/gdrive/My Drive/model.pt`
 or
 `python train.py`

### Predict
Predicting two possible image sets

- Run `python test.py valid` to predict on the default D-LinkNet34 from the validation dataset.
- Run `python test.py test`  to predict on the default D-LinkNet34 from the testing dataset.

### Download trained D-LinkNet34
- [Dropbox](https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma?dl=0)
