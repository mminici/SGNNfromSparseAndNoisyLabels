**Use this [colab notebook](https://colab.research.google.com/drive/1dxc5n-Xf0tHfO8ys3t6ubRN3jJhRL66y?usp=sharing) to reproduce our results**

If you use this code or part of it, please cite the original reference:

> Minici, M., Cinus, F., Bonchi, F., & Manco, G. (2024, October). Link Polarity Prediction from Sparse and Noisy Labels. In Proceedings of the 33rd ACM International Conference on Information & Knowledge Management. doi: [https://doi.org/10.1145/3511808.3557253](https://doi.org/10.1145/3627673.3679786)

---

# Data Preprocessing
* Run ```python pre-processing-unsupervised-experiment.py``` for each dataset (```bitcoin_alpha, bitcoin_otc, wiki, slashdot```) and noise percentage (```0.0, 0.1, 0.2```). Alternatively, you can use the preprocessed files we will soon update on Zenodo.

---

# Running scripts
Each dataset has a different set of hyperparameters, change their values accordingly in the ```run_SDGNN_lrw_MicroMesoSB.py```.

* ```bitcoin_alpha```:
  * ```unlabeled_perc = [None, ]```
  * ```init_eps_one = True```
* ```bitcoin_otc```:
  * ```unlabeled_perc = [0.8, ]```
  * ```init_eps_one = True```
* ```wiki```:
  * ```unlabeled_perc = [None, ]```
  * ```init_eps_one = True```
* ```slashdot```:
  * ```unlabeled_perc = [0.5, ]```
  * ```init_eps_one = False```

We ensure other researchers can reproduce our results using this ready-to-use [colab notebook](https://colab.research.google.com/drive/1dxc5n-Xf0tHfO8ys3t6ubRN3jJhRL66y?usp=sharing).
For the sake of experimentation velocity, we will soon update our preprocessed files on a Zenodo node. However, you can preprocess the files by yourself using the ```pre-processing-unsupervised-experiment.py``` script (present in the Colab environment).
