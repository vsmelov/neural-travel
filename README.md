# neural-travel
Why do you need to travel if you have neural networks?

## Run

### Docker

### Bash

#### Requirements

Python3.8

```bash
git clone https://github.com/vsmelov/neural-travel.git
cd neural-travel
git submodule init
git submodule update
virtualenv -p python3.7 venv
. venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=./src:./Mask_RCNN python src/neural_travel/mycoco.py --img=
```
