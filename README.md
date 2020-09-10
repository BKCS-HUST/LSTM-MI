# A LSTM based framework for handling multiclass imbalance in DGA botnet detection. 

This repo contains very simple code for classifying domains as DGA or benign. This code demonstrates our results in our paper here: https://www.sciencedirect.com/science/article/pii/S0925231217317320. This paper presents a novel LSTM.MI algorithm to combine both binary and multiclass classification models, where the original LSTM is adapted to be cost-sensitive. The cost items are introduced into backpropagation learning procedure to take into account the identification importance among classes.

Follow the instructions, you can reimplement our method. Besides, we also provide some datasets:

- The real-world collected datasets of some DGA botnet including Shifu, Tinba, Necurs, Locky, etc.
- trainingdata.csv: This dataset contains 1 million normal domains of Alexa, over 600,000 DGA domains, and 8 statistic features corresponding to these domains. The DGA domains are collected at http://osint.bambenekconsulting.com/feeds/.
- traindga4.csv: 10% of trainingdata dataset. However, 8 statistic features are not normalized into the values ranging from 0 to 1.
- traindga5.csv: 10% of trainingdata dataset. However, 8 statistic features are normalized into the values ranging from 0 to 1.

**If you reuse our dataset, please cite out paper as follows.**

Tran, Duc, et al. "A LSTM based framework for handling multiclass imbalance in DGA botnet detection." Neurocomputing 275 (2018): 2401-2413.

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

## Implementing the code

'lstm.py' will trainthe classifier and generate the report to evaluate the method. This code will run on your local machine or on a machine with a GPU (GPU will of course be much faster).
  
## Backup link
If there is any problems related to the download on Github, please use the backup link on Google Driver:
https://drive.google.com/drive/folders/1EI67tTK2NnwW9MZa3Qr2anPbJ_Dlix91?fbclid=IwAR0A_Ltrkrn9Zt4VvF0IFhE_v3GWz-YDqaq-SsHaU2mHNG5861KXylvZlBY
