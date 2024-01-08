Code for **_"FeBT : A Feature Balancing Transformer for Corporate ESG-Forecasting"_**

# Contents
- [Abstract](#abstract)
- [Key Components](#key-components)
	- [Workflow](#workflow)
	- [Masking](#masking)
 - [Datasets and Evaluation Metrics](#datasets-and-evaluation-metrics)
   	- [Datasets](#datasets)
   	- [Evaluation Metrics](#evaluation-metrics)
 - [Performance](#performance)
 - [Conclusion](#conclusion)	
 	- [Theoretical Implications](#theoretical-implications)
   	- [Practical Implications](#practical-implications)
 - [Data Availability Statement](#data-availability-statement)	

# Abstract
We propose FeBT, which is an end-to-end model based on a masking autoencoder and Transformer. FeBT incorporates a novel feature balancing technique that compresses and enhances high-dimensional features from imbalanced data into low-dimensional representations, thereby aligning them with other features. We also use an appropriate sliding window approach to expand the sample size and effectively capture temporal information, which leads to improved prediction performance. Through a series of experiments, we determined the optimal model structure and verified the superior performance of FeBT compared with state-of-the-art methods in predicting ESG performance.


# Key Components

## Workflow
The operational schema of FeBT is depicted in the figure below. The workflow of FeBT is divided into two sequential phases: an initial pre-training phase confined to the feature balancing module, followed by a forecasting phase that engages the entire model. The FBM is located on the lower left in the figure and takes the folding features as input. It undertakes the pretraining of $A=|\mathbf{X}^{\rm fld}|$ distinct autoencoders for $A$ folding attributes, thereby deducing an optimal compression strategy that corresponds to each specified $d_i$ through a detached operation. Enhanced by data masking techniques, this module seeks to amplify the autoencoders’ proficiency in pattern distillation. After pre-training, the downstream Transformer harnesses the balanced dataset $\chi^{\rm bal}$ to extrapolate the final ESG predictions.

![the workflow of FeBT](./figure/Figure_workflow.jpg "the workflow of FeBT")

###### **Workflow of FeBT. Using the feature balancing technique, we allow the folding features $\mathbf{F}\_{i}$ within the imbalanced dataset $\chi^{\rm orig}$ to obtain latent representations through the upstream autoencoders. Subsequently, we construct a balanced feature set $\chi^{\rm bal}$ using these latent representations ${\mathbf{F}_{i}'|i=1,2,\cdots}$ combined with the original palin features $\mathbf{X}^{pln}$. Then we use the balanced feature set as the input for the downstream Transformer blocks to predict ESG scores.**

## Masking
To further enhance the quality of the compressed features, we use a masking technique. This technique involves deliberately erasing parts of the input features during the training of the autoencoders.  The erased parts, or masks, compel the autoencoders to focus on less obvious characteristics of the data, thereby improving their ability to reconstruct the original signal, even in the presence of incomplete or distorted input. This added complexity not only refines the feature representations but also leads to a degree of robustness in the model.  Specifically, we divide the folding feature vectors into $k$ regular non-overlapping patches and make one patch of them invisible to the encoders. We adopt one-hot positional embeddings to identify the location of patches. Figure below shows these procedures.

<div align="center">
  <img src="./figure/Figure_datamasking.jpg" alt="Datamasking" width="600"/>
</div>

###### **Data masking.  Each patch is assigned with a positional embedding. The masked patch is invisible to the encoders.**

# Datasets and Evaluation Metrics

## Datasets

We acquired four datasets through direct releases from companies or by gathering data via textual analysis.  These datasets are the Chinese Major Cities Air Quality dataset (CMCAQ), the Stock Bar Information of Chinese Listed Companies (SBICIC), the Green Word Vectors of Chinese Listed Companies (GWVCIC), and the Fundamental dataset of Chinese Listed Companies (FCIC).  In table below, we list some key statistics of features in all datasets.

<div align="center">
	
| Dataset | Feature Type | Fold | #Feature | #Dimension <br> Per Annual |
| :----------: | :-----------: | :-----------: | :-----------: | :-----------: |	
| CMCAQ | _Time Series_ | √ | 7 | 2,555 |
| SBICIC | _Time Series_ | √ | 3 | 1,095 |
| GWVCIC | _Word Vector_ | √ | 1 | 160 |
| FCIC | _Discrete_ | × | 31 | 31 |

</div>

## Evaluation Metrics
For the ESG scoring regression task, we adopted Mean Squared Error (MSE) as the loss function, recognizing its effectiveness in capturing the variance of the prediction errors. We evaluated the regression performance using a set of carefully chosen metrics.

**Scoring 5% Margin Acc**: This metric quantifies the proportion of predictions where the predicted ESG scores deviate by no more than 5% from their groundtruth. It offers a fine-grained assessment of the model's performance with a narrow error margin.

**Scoring 10% Margin Acc**: Similar to the 5\% margin metric, this measures the proportion of predictions within a 10% error margin. It provides a broader view of the model's accuracy, accommodating a slightly more relaxed error tolerance.

**Scoring MAPE**: This represents the mean of the absolute percentage errors across all predictions, offering a comprehensive overview of the model's overall percentage accuracy.

In parallel, we extended our investigation to ESG rating classification to gain additional insights into our model's versatility. Here, we employed cross-entropy as the loss function, acknowledging its suitability for classification problems. The classification performance was evaluated based on the accuracy (**Rating Acc**), macro average recall (**Rating Rec**), and macro average precision (**Rating Pre**).


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

# Conclusion

## Theoretical Implications

The evaluation system for corporate ESG scores is characterized by its complexity and significant dimensional variability among its features. Overlooking the high-dimensional folding features could lead to missing critical aspects. Directly merging these folding features with a firm’s plain features during training may impede the model’s predictive efficiency. Additionally, conventional machine learning models often overlook the temporal nature of ESG scores. To address these gaps, we developed FeBT, a two-stage, data-adaptive ESG prediction model leveraging Transformer technology. Our approach introduces a novel Feature Balancing Module (FBM), enhancing the compression of high-dimensional features and effectively addressing imbalances across any type of folding feature, thereby amplifying the model’s expressiveness. To our knowledge, FeBT represents the first universal feature balancing pipeline specifically designed to tackle the challenges posed by imbalanced ESG data.

## Practical Implications

FeBT empowers each feature to realize its full potential, unencumbered by data imbalance issues. In the realm of ESG evaluation, challenges such as limited score coverage, high subjectivity, and poor comparability among different systems are prevalent. Leveraging publicly available data, our proposed ESG forecasting model substantially mitigates these concerns. Experimental outcomes on our dataset suggest that FeBT has the capability to enhance the uniformity and transparency of ESG evaluation systems. This advancement contributes to the ongoing enhancement of the systems’ reliability and overall impact in the field.

# Data Availability Statement

The model was evaluated using publicly available benchmarks that can be accessed via their references.


