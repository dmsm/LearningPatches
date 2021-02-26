## LearningPatches | [Webpage](https://people.csail.mit.edu/smirnov/learning-patches/) | [Paper](https://openreview.net/forum?id=Gu5WqN9J3Fn) | [Video](https://youtu.be/mxpkK9hHfFE)

<img src="https://people.csail.mit.edu/smirnov/learning-patches/im.png" width="75%" alt="Learning Manifold Patch-Based Representations of Man-Made Shapes" />

**Learning Manifold Patch-Based Representations of Man-Made Shapes**<br>
Dmitriy Smirnov, Mikhail Bessmeltsev, Justin Solomon<br>
[International Conference on Learning Representations (ICLR) 2021](https://iclr.cc/Conferences/2021/)

### Set-up
To install the code, run:
```
conda create -n learningpatches python=3.6 -y
conda activate learningpatches
conda install pytorch=1.3.1 torchvision cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt
```

Also, be sure to execute `export PYTHONPATH=:$PYTHONPATH` prior to running any of the scripts.

### Demo
First, download the pretrained models for each shape category:
```
wget -O models.zip https://www.dropbox.com/s/ntt1ytpjwx2385i/learningpatches_models.zip?dl=0
unzip models.zip
```

Then, run the following to generate an OBJ file with the 3D model for a given input sketch PNG image:
```
python scripts/run.py demo/airplane.png airplanes out.obj --no-turbines
```
Make sure to specify `airplanes`, `bathtubs`, `bottles`, `cars`, `guitars`,
`guns`, `knives`, or `guns` as the shape category. Optionally, for the airplanes
category, the `--no-turbines` flag does not output the turbine patches in the 
3D model.

The `demo` directory contains PNGs of some sample input sketches.

Note that the meshes output by the demo script may have non-manifold discontinuities between patches due to
discretization artifacts. This can be avoided by choosing the number of subdivisions based on patch boundary
arc lengths. The results shown in the paper are all computed in this way.

### BibTeX
```
@inproceedings{smirnov2021patches,
  title={Learning Manifold Patch-Based Representations of Man-Made Shapes},
  author={Smirnov, Dmitriy and Bessmeltsev, Mikhail and Solomon, Justin},
  year={2021},
  booktitle={International Conference on Learning Representations (ICLR)}
}
```
