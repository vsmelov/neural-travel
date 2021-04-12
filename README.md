# neural-travel
Why do you need to travel if you have neural networks?

## Run

See ./generate-frames.ipynb

### Docker

TODO

### Bash

```bash
git clone https://github.com/vsmelov/neural-travel.git
cd neural-travel
git submodule init
git submodule update
virtualenv -p python3.7 venv
. venv/bin/activate
pip install -r requirements.txt

cd swapping-autoencoder-pytorch
wget http://efrosgans.eecs.berkeley.edu/SwappingAutoencoder/swapping_autoencoder_models_and_test_images.zip
unzip swapping_autoencoder_models_and_test_images.zip
rm -r swapping_autoencoder_models_and_test_images.zip
cd ..

PYTHONPATH=./src:./Mask_RCNN:./Mask_RCNN/samples/coco python src/neural_travel/mycoco.py --img=
```


```bash
python frames2video.py
```

```bash
ffmpeg -i video.avi video.mp4
```