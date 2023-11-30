# FeBT-A-Feature-Balancing-Transformer-for-Corporate-ESG-Forecasting

# Contents
- [Abstract](#abstract)
- [Key Components](#keycomponents)
	- [Feature Balancing Model](#featurebalancingmodel)
	- [Forecasting Model](#forecastingmodel)
 - [Performance](#performance)

# Abstract

  Environmental, social, and governance (ESG) serves as a crucial indicator for evaluating firms in terms of sustainable development. 
However, the existing ESG evaluation systems suffer from limitations, such as narrow coverage, subjective bias, and lack of timeliness. 
Therefore, there is a pressing need to leverage machine learning methods to predict the ESG performance of firms using their publicly available data. 
However, traditional machine learning models encounter the feature imbalance problem due to the heterogeneity in ESG-related features, which results in the neglect of low-dimensional features. 
Common approaches typically involve unfolding all features, thereby granting high-dimensional features greater exposure and accessibility to downstream models.
Consequently, a research gap exists regarding fully using the heterogeneous features of enterprises to enhance ESG prediction performance. 
In this paper, we propose FeBT, which is an end-to-end model based on a masking autoencoder and Transformer. 
FeBT incorporates a novel feature balancing technique that compresses and enhances high-dimensional features from imbalanced data into low-dimensional representations, thereby aligning them with other features. 
We also use an appropriate sliding window approach to expand the sample size and effectively capture temporal information, which leads to improved prediction performance. 
Through a series of experiments, we determined the optimal model structure and verified the superior performance of FeBT compared with state-of-the-art methods in predicting ESG performance.


# Key Components
![the workflow of FeBT]((https://github.com/leeyan0612/FeBT-A-Feature-Balancing-Transformer-for-Corporate-ESG-Forecasting/blob/master/figure/Figure_workflow.jpg) "the workflow of FeBT")



# Performance
**ESG score and ESG rating prediction for FeBT and the baselines.**
<div align="center">

| Model | Scoring <br> 5% Margin Acc ⬆ | Scoring <br> 10% Margin Acc ⬆ | Scoring <br> Mape ⬇ | Rating <br> Acc ⬆ | Rating <br> Rec ⬆ | Rating <br> Pre ⬆ |
| :----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| XGBoost | 0.624 | 0.899 | 0.053 | 0.857 | 0.552 | 0.774 |
| LightGBM | 0.646 | **0.920** | 0.048 | 0.857 | 0.558 | 0.767 |
| LSTM | 0.590 | 0.865 | 0.059 | 0.814 | 0.568 | 0.699 |
| GRU | 0.581 | 0.884 | 0.057 | 0.787 | 0.581 | 0.732 |
| Transformer | 0.597 | 0.894 | 0.057 | 0.787 | 0.581 | 0.732 |
| FeBT-padding | 0.686 | 0.873 | 0.053 | 0.851 | 0.553 | 0.747 |
| **FeBT** | **0.735** | 0.916 | **0.042** | **0.883** | **0.660** | **0.795** |
  
</div>
