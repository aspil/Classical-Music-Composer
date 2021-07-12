# Classical-Music-Composer

This project was developed and presented for the Computer Music course as a semester project. The goal is to generate classical pieces that follow the musical patterns of the training data.  
The results were not disappointing nor rewarding as the generated pieces didn't really have those musical patterns in them but they have a lot of potential.  
That alone sets the challenge.

## Prerequisites
Install required modules using
```pip install -r requirements.txt```

## Usage
First download the project using `git clone`.

To train the model and generate a new note sequence into a midi file, run:  
```python src/main.py --path dataset/classical_music_midi --composers <composer names>```

Composer names should be space separated. For the available composers check the folders in *dataset/classical_music_midi* directory.

## Future work
Some ideas worth trying:
- better preprocessing, tweaks in input format
- encode more musical information into the input data
- remodeling of the network architecture, maybe move to Transformers as well

## Licence
This project is licensed under the MIT License.

##
PS: trained model files exceeded GitHub's file size limit of 100.00 MB, so they weren't uploaded
