# Representational Alignment Security
This repository contains the official code for our paper: 
> **Alignment and Adversarial Robustness: Are More Human-Like Models More
> Secure?** [[Paper Link]](https://arxiv.org/abs/2502.12377) \
> *Blaine Hoak\*, Kunyang Li\*, Patrick McDaniel (\*Equal contribution)* \
> 📍*The European Conference on Artificial Intelligence (ECAI) - Workshop on
> Security and Privacy-Preserving AI/ML (SPAIML), October 25-26, 2025, Bologna, Italy*


## 📌 Overview
Can models that see more like humans also resist attacks more like humans? We
present the first large-scale empirical analysis of the relationship between
representational alignment---how closely a model resembles human visual
perception---and adversarial robustness. 

We evaluate 144 vision models spanning diverse architectures and training
schemes, and analyze their performance on: 
- 105 neural, behavioral, and engineering alignment benchmarks (via
  [Brain-Score](https://www.brain-score.org/))
- Robustness to adversarial attacks using the AutoAttack benchmark

Our key findings are: 
- Average alignment is a weak predictor of robustness, especially behavioral
  alignment (~6% variance explained).
- Specific alignment benchmarks (e.g., on texture information processing) are
  highly predictive of robust accuracy. 
- Some forms of alignment improve robustness, others hurt it, highlighting the
  importance of what kind of human-likeness a model achieves. 

## 📁 Project Structure (dummy)
<pre>
alignment-security/
├── scripts/ 
│   ├── main.py 
│   ├── attack.py 
│   └── utilities.py 
├── Dockerfile # Requirements and dependencies
└── README.md 
</pre>

## 🧪 Experiments
### 1. Setup
First, clone the repository: 
```
git clone git@github.com:kyangl/alignment-security.git
cd alignment-security
```
We use Docker to manage dependencies and ensure reproducibility. Now, you can build
and run the container as follows: 
```
# Build the Docker image 
docker build -t alignment-security . 

# Run the container with GPU support 
docker run --gpus all -it alignment-security
``` 

Note: `--gpu all` is required for GPU support. Make sure [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
is installed. 

### 2. Running Experiments

## 📎 Citation
If you find this work useful, please cite the following paper: 
```

@inproceedings{hoak_alignment_2025,
	title = {Alignment and {Adversarial} {Robustness}: {Are} {More} {Human}-{Like} {Models} {More} {Secure}?},
    booktitle = {Workshop on Security and Privacy-Preserving AI/ML (SPAIML)}
	url = {http://arxiv.org/abs/2502.12377},
	author = {Hoak, Blaine and Li, Kunyang and McDaniel, Patrick},
	month = feb,
	year = {2025},
}

```


## 📬 Contact
For questions or collaborations, you are welcome to contact us at [email]().