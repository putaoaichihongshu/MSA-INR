#!/bin/bash

# sin200x
python curve_fitting.py --config config/curvefit_sin200x.yaml --model msa --seed 46

# sin500x
python curve_fitting.py --config config/curvefit_sin500x.yaml --model msa --seed 46

# sin1000x
python curve_fitting.py --config config/curvefit_sin1000x.yaml --model msa --seed 44

# multi_scale
python curve_fitting.py --config config/curvefit_multiscale.yaml --model msa --seed 42  -scale


# 2D image_fitting
python image_fitting.py --config config/imagefit.yaml --model msa

# 2D color image_fitting
python image_fitting.py --config config/imagefitcolor.yaml --model msa

# video
python video_fitting.py --config config/videofit_bbb.yaml --model msa

# 3D shape
python reconstruction3D.py --config config/reconstruction.yaml --model msa