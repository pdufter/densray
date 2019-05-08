### Introduction

This repository compares different methods of obtaining interpretable dimension in word embedding spaces. 

More specifically it compares: 
* [Densifier](https://arxiv.org/pdf/1602.07572.pdf)
* [DensRay](https://arxiv.org/pdf/1904.08654.pdf): A method closely related to Densifier, but computable in closed form. 
* Support Vector Machines / Regression
* Linear / Logistic Regression.

The evaluation tasks are lexicon induction and set-based word analogy. 

For more details see the [Paper](https://arxiv.org/pdf/1904.08654.pdf).

Note that this repo does not include an implementation of the Densifier, but relies on the original Matlab implementation by the authors of Densifier.


### Usage

For an example how to use the code see `example.sh`.


### References

If you use the code, please cite 
```
@article{dufter2019analytical,
  title={Analytical Methods for Interpretable Ultradense Word Embeddings},
  author={Dufter, Philipp and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:1904.08654},
  year={2019}
}
```
