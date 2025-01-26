Comparing Kolmogorov-Arnold Networks (KAN) and Traditional Neural Networks (TNN) for Air Quality Index (AQI) Classification

Abstract

This research investigates the comparative performance of Kolmogorov-Arnold Networks (KAN) and Traditional Neural Networks (TNN) on a classification task using an Air Quality Index (AQI) dataset from India. KAN, inspired by the Kolmogorov-Arnold representation theorem, offers a unique approach by employing learnable activation functions on edges, whereas TNN relies on fixed activations at nodes. The study examines the theoretical claims surrounding KAN's architecture and evaluates its practical performance against TNN. Experimental results show that TNN slightly outperforms KAN on this dataset, with higher accuracy and reduced complexity. These findings suggest that while KAN holds promise for highly nonlinear datasets, its benefits may not generalize across all tasks.

1. Introduction

The Air Quality Index (AQI) is a critical measure for assessing air pollution levels and their impact on human health. Classifying AQI into meaningful categories, such as "Good," "Moderate," and "Unhealthy," is essential for timely intervention. Machine learning models, particularly neural networks, have shown great potential in solving such classification tasks.

Traditional Neural Networks (TNN) are widely used for classification tasks due to their simplicity and effectiveness. However, recent advances propose alternative architectures, such as Kolmogorov-Arnold Networks (KAN). KAN, inspired by the Kolmogorov-Arnold representation theorem, approximates multivariate functions using a combination of univariate transformations and an additive layer. According to [1], KAN introduces learnable activation functions on edges, potentially improving its capacity to model nonlinear relationships.

This study aims to empirically evaluate KAN's performance against TNN in the context of AQI classification. Specifically, it explores whether KAN's theoretical advantages translate to practical improvements in performance metrics.

2. Related Work

Neural networks have been extensively applied to environmental datasets for predictive modeling and classification. Studies like [2] and [3] have demonstrated the efficacy of TNN in predicting AQI levels. However, the application of advanced architectures, such as KAN, remains underexplored in this domain.

The Kolmogorov-Arnold representation theorem has been foundational in theoretical studies of neural networks. Research in [1] claims that KAN's learnable edge activations offer greater flexibility than the fixed-node activations in TNN. Other works, such as [4], discuss the role of parameterized activations (e.g., PReLU) in enhancing TNN's performance, challenging the unique advantages of KAN.

3. Methodology

3.1 Dataset

The AQI dataset, sourced from an open repository, contains pollutant concentrations (SO2, NO2, PM10, PM2.5, O3) and corresponding AQI values. The target variable is categorized into six classes: Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, and Hazardous. Preprocessing steps included:

Replacing missing values with zeros.

Standardizing features using a scaler.

Binning AQI values into categorical labels.

3.2 Model Architectures

Traditional Neural Network (TNN):

Two hidden layers with 64 neurons each.

ReLU activation functions.

Softmax output layer for multi-class classification.

Optimized using the Adam optimizer.

Kolmogorov-Arnold Network (KAN):

Learnable univariate transformations (inspired by the Kolmogorov-Arnold theorem).

Aggregation of transformations using an additive layer.

L2 regularization and dropout to mitigate overfitting.

Hyperparameters (e.g., units, L2 strength, dropout rate, learning rate) optimized via grid search.

3.3 Evaluation Metrics

Accuracy on the test dataset.

Training and validation loss curves.

Class-wise prediction distributions.

4. Experiments and Results

4.1 Accuracy Comparison

TNN Accuracy: 73.75%

KAN Accuracy: 70.69%

Although KAN came close, TNN slightly outperformed it in terms of classification accuracy. This suggests that the simplicity of TNN's architecture made it better suited for this dataset.

4.2 Loss Curve Analysis

Both models demonstrated convergence during training. However, KAN exhibited slightly higher validation loss, suggesting minor overfitting despite regularization.

4.3 Class Distribution Analysis

Both models effectively captured the overall class distribution but struggled with underrepresented classes, indicating the need for data balancing or augmentation.

4.4 Dataset Nonlinearity

Preliminary analysis revealed moderate nonlinearity in the dataset, but not enough to justify the added complexity of KAN over TNN.

5. Discussion

KAN's Strengths:

Theoretical flexibility for modeling complex nonlinear relationships.

Learnable univariate transformations offer adaptability.

Challenges with KAN:

Computationally intensive and requires extensive tuning.

On moderately nonlinear datasets like AQI, the added complexity does not yield significant performance gains.

Additionally, the claim that "KANs employ learnable activation functions on edges, instead of the fixed activation functions on nodes" [1] appears less impactful in practice. Modern TNN architectures with parameterized activations (e.g., PReLU) or attention mechanisms can achieve similar flexibility, making KAN's distinction more theoretical than practical.

6. Conclusion

This study highlights the importance of selecting the appropriate architecture for a given task. While KAN offers theoretical advantages, its practical utility depends on the dataset's complexity. For the AQI dataset, the simplicity and efficiency of TNN proved superior.

Future work should focus on datasets with pronounced nonlinear patterns, where KAN might demonstrate its full potential. Integrating advanced preprocessing techniques, such as feature engineering and class balancing, could further enhance the performance of both models.

The complete code and details of this research can be found on my GitHub repository: GitHub Link.

References

LLMs Will Always Hallucinate, and We Need to Live
With This : https://arxiv.org/pdf/2409.05746

Study on Neural Networks for AQI Prediction: https://www.jsoftcivil.com/article_163709.html

Advances in Parameterized Activations for Neural Networks: https://keras.io/keras_tuner/

