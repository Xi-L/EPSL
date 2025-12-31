# EPSL

Code for TEVC2025 Paper: Dealing with Structure Constraints in Evolutionary Pareto Set Learning

The code is mainly designed to be simple and readable, it contains:
- <code>run_[epsl].py</code> is a ~300-line main file to run the EPSL method, calculate the hypervolume, and plot the Pareto front;
- <code>model.py or model_[shared_component/shared_component_syn/variable_relation/keypoint].py</code> contains the standard EPSL model or EPSL model with structure constraint of [shared_component/shared_component_syn/variable_relation/keypoint];
- <code>problem.py</code> contains all test problems used in this paper;
- The folder <code>data</code> contains the problem information for the RE problems, which is obtained from the [reproblems repository](https://github.com/ryojitanabe/reproblems).


**Reference**

If you find our work helpful for your research, please cite our paper:
```
@article{lin2025dealing,
  title={Dealing With Structure Constraints in Evolutionary Pareto Set Learning},
  author={Lin, Xi and Zhang, Xiaoyuan and Yang, Zhiyuan and Zhang, Qingfu},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2025},
  publisher={IEEE}
}
```

If you find the RE problems useful for your research, please also cite the RE paper:
```
@article{tanabe2020easy,
  title={An easy-to-use real-world multi-objective optimization problem suite},
  author={Tanabe, Ryoji and Ishibuchi, Hisao},
  journal={Applied Soft Computing},
  volume={89},
  pages={106078},
  year={2020},
  publisher={Elsevier}
}
```
