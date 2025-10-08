# PARC

Project page: https://michaelx.io/parc

# Getting Started
Tested with Ubuntu 22.04

Install IsaacGym: https://developer.nvidia.com/isaac-gym

Make sure to install it within a conda environment with python 3.8.19 (Other versions may also work, but not tested).

Install requirements:
```
conda activate parc
pip install -r requirements.txt
```
and it should be good to go. If pytorch is not being able to detect CUDA, try reinstalling:
```
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset and Models
Download the datasets from the initial iteration and each stage of PARC, as well as the models trained in the 3rd PARC iteration.
These files are loaded with anim/motion_lib.py and anim/kin_char_model.py.
You can view them with motion_forge.py, by editing the "motion_filepath" param in motionscope/motion_forge_config.yaml
https://1sfu-my.sharepoint.com/:f:/g/personal/mxa23_sfu_ca/Et16uLMFxoRKouibvBa7LbwBEmX5_iI5a8dZyiMc0wmSTA?e=ihma1b
The password is "PARC".

## Codebase Guide
The PARC training loop consists of 4 main stages, which are individually run by:
```
python parc_1_train_gen.py --config path/to/config
python parc_2_kin_gen.py --config path/to/config
python parc_3_tracker.py --config path/to/config
python parc_4_phys_record.py --config path/to/config
```

These modules are independent and flexibly configured using configuration files. However, to aid in setting up all the configuration files for a PARC iteration, we provide the following script:
```
parc_0_setup_iter.py
```
* [PARC Guide](doc/parc_guide.md)  (Coming Soon)
* [Motionscope Guide](doc/motionscope_guide.md)  (Coming Soon)

## Planned Features
* Iteration 4 and 5 models, datasets
* Motion file format refactoring
* Faster inference code
* Full codebase cleanup and refactoring

## Citation
If you find PARC helpful, please consider citing:
```bibtex
@inproceedings{xu2025parc,
    author = {Xu, Michael and Shi, Yi and Yin, KangKang and Peng, Xue Bin},
    title = {PARC: Physics-based Augmentation with Reinforcement Learning for Character Controllers},
    year = {2025},
    booktitle={SIGGRAPH 2025 Conference Papers (SIGGRAPH '25 Conference Papers)}
}
```

Please also consider citing [MimicKit](https://github.com/xbpeng/MimicKit), the codebase that PARC was built on:
```bibtex
@misc{MimicKit,
	title = {MimicKit},
	author = {Peng, Xue Bin},
	year = {2025},
	publisher = {GitHub},
	journal = {GitHub repository},
	howpublished = {\url{https://github.com/xbpeng/MimicKit}},
}
```

If you find Motionscope (previous name: Motion Forge, update coming soon) helpful, please consider citing:
```bibtex
@software{Xu_Michael_Motionscope,
  author = {Xu, Michael and Peng, Xue Bin},
  title = {{motionscope}},
  year = {2025},
  month = aug,
  version = {0.0.1},
  url = {https://github.com/mshoe/PARC},
  license = {MIT}
}
```

Motionscope is built on top of a wonderful tool called [Polyscope](https://polyscope.run/py/). Please consider citing Polyscope here:
```bibtext
@misc{polyscope,
  title = {Polyscope},
  author = {Nicholas Sharp and others},
  note = {www.polyscope.run},
  year = {2019}
}
```

Also, consider citing these important papers which PARC's motion tracker and motion generator builds upon:

DeepMimic
```bibtex
@article{
	2018-TOG-deepMimic,
	author = {Peng, Xue Bin and Abbeel, Pieter and Levine, Sergey and van de Panne, Michiel},
	title = {DeepMimic: Example-guided Deep Reinforcement Learning of Physics-based Character Skills},
	journal = {ACM Trans. Graph.},
	issue_date = {August 2018},
	volume = {37},
	number = {4},
	month = jul,
	year = {2018},
	issn = {0730-0301},
	pages = {143:1--143:14},
	articleno = {143},
	numpages = {14},
	url = {http://doi.acm.org/10.1145/3197517.3201311},
	doi = {10.1145/3197517.3201311},
	acmid = {3201311},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {motion control, physics-based character animation, reinforcement learning},
} 
```

Motion Diffusion Model
```bibtex
@inproceedings{
tevet2023human,
title={Human Motion Diffusion Model},
author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}
```