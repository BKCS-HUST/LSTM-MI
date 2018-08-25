# A LSTM based framework for handling multiclass imbalance in DGA botnet detection.

This repo contains the code for classifying domains in order to detect DGA bots. This code demonstrates the results in the paper: https://www.sciencedirect.com/science/article/pii/S0925231217317320, which presents a novel LSTM.MI algorithm to combine both binary and multiclass classification models, where the original LSTM is adapted to be cost-sensitive. The cost items are introduced into backpropagation learning procedure to take into account the identification importance among classes. We also include a sample dataset, called trainingdga5.cvs for those, who would like to assess the LSTM.MI algorithm. This dataset contains 100,000 domains from Alexa and over 60,000 DGA domains. The ID related to Alexa class is 20.

**If you reuse this code, please cite the following paper**

Tran D., Mac H., Tong V., Tran H.A. and Nguyen L.G., "A LSTM based framework for handling multiclass imbalance in DGA botnet detection." Neurocomputing, vol. 275, pp. 2401-2413, 2018.

or

@article{tran2018lstm,
 title={A LSTM based framework for handling multiclass imbalance in DGA botnet detection},
 author={Tran, Duc and Mac, Hieu and Tong, Van and Tran, Hai Anh and Nguyen, Linh Giang},
 journal={Neurocomputing},
 volume={275},
 pages={2401--2413},
 year={2018},
 publisher={Elsevier}
}

Note that the lstm.py will train the classifier and generate the report, related to the evaluation of the LSTM.MI. This code was run on CPU, but can be implemented with GPU to reduce the computational cost.

