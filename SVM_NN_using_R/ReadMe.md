{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # \uc0\u55357 \u56522  Binary Classification using Support Vector Machines (SVM) and Neural Networks (NN)\
\
## \uc0\u55358 \u56800  Project Overview\
\
This project compares the performance of two supervised learning models \'97 **Support Vector Machines (SVM)** and **Neural Networks (NN)** \'97 on a synthetic 2D binary classification dataset. The dataset is sourced from the CMU StatLib repository, specifically designed for evaluating pattern recognition models with nonlinear boundaries.\
\
Through a series of experiments and visualizations, we analyze decision boundary formation, the effect of hyperparameter tuning, and overall classification performance.\
\
---\
\
## \uc0\u55356 \u57263  Objectives\
\
1. **Can we accurately classify the target using SVM and NN?**\
2. **Which hyperparameters significantly influence the model's performance?**\
3. **Which model generalizes better to unseen data?**\
\
---\
\
## \uc0\u55357 \u56513  Dataset\
\
- **Source:** [CMU StatLib \'96 PRNN Synthetic Dataset](https://lib.stat.cmu.edu/datasets/)\
- `synth.tr.txt`: 250 training samples\
- `synth.te.txt`: 1000 test samples\
- **Features:** `xs`, `ys`\
- **Target:** Binary class (`0` or `1`)\
\
---\
\
## \uc0\u9881 \u65039  Preprocessing\
\
- Removed identifier column\
- Converted string values to numeric\
- Applied **Z-score standardization**:\
  - Mean = 0\
  - Standard Deviation = 1\
- Ensured feature ranges are consistent for distance-based models like SVM and NN\
\
---\
\
## \uc0\u55358 \u56810  Modeling and Tools\
\
| Task                  | Method / Package            |\
|-----------------------|-----------------------------|\
| SVM Classification    | `e1071::svm()`              |\
| Neural Network (NN)   | `nnet::nnet()`              |\
| Evaluation Metrics    | `caret::confusionMatrix()`  |\
| Visualization         | `ggplot2`, `gridExtra`      |\
\
---\
\
## \uc0\u55357 \u56615  Tuning Experiments\
\
### \uc0\u55356 \u57088  SVM Tuning\
- **Kernels tested:** radial, polynomial, sigmoid\
- **Parameters:**\
  - Cost: `0.1` to `25`\
  - Gamma: `0.0005` to `0.6`\
- **Best Result:**\
  - **Kernel:** RBF\
  - **Cost:** 4\
  - **Gamma:** 0.6\
  - **Accuracy:** 92.3%\
\
### \uc0\u55358 \u56598  NN Tuning\
- **Neurons tested:** 5 to 200 (step of 5)\
- **Decay values tested:** `0.01`, `0.001`, `0.0001`\
- **Best Result:**\
  - **Neurons:** 50\
  - **Decay:** 0.0001\
  - **Accuracy:** 100%\
\
---\
\
## \uc0\u55357 \u56520  Performance Comparison\
\
| Model           | Configuration            | Accuracy | Sensitivity | Specificity | Kappa | Balanced Acc. |\
|----------------|---------------------------|----------|-------------|-------------|--------|----------------|\
| Initial SVM     | RBF, C=1, \uc0\u947 =0.5          | 91.4%    | 92.2%       | 90.6%       | 0.828 | 91.4%          |\
| Tuned SVM       | RBF, C=4, \uc0\u947 =0.6          | 92.3%    | 91.4%       | 93.2%       | 0.846 | 92.3%          |\
| Initial NN      | 5 neurons, decay=0.01    | 92.1%    | 91.6%       | 92.6%       | 0.842 | 92.1%          |\
| Tuned NN        | 50 neurons, decay=0.0001 | 100%     | 100%        | 100%        | 1.000 | 100%           |\
\
---\
\
## \uc0\u55357 \u56522  Visualizations\
\
- **Initial SVM vs Initial NN**: Highlights early-stage performance differences.\
- **Fine-Tuned SVM**: Shows marginal improvement with optimal gamma & cost.\
- **Fine-Tuned NN**: Shows complete separation and perfect classification with 50+ neurons and low decay.\
\
> \uc0\u55357 \u56764 \u65039  Decision boundary plots included in `/visualizations/` folder.\
\
---\
\
## \uc0\u55357 \u56481  Key Insights\
\
- SVMs are powerful but sensitive to `gamma` and `cost`. Improvements plateaued around ~92%.\
- Neural Networks generalized better when tuned correctly. Lower decay allowed higher learning flexibility.\
- Final NN setup achieved **perfect test accuracy**, with training misclassifications < 0.5%, indicating **healthy generalization**.\
- Visualization of decision boundaries helped assess and interpret model behavior.\
\
---\
\
## \uc0\u55356 \u57101  Real-World Applications\
\
- **Medical Diagnosis:** Classify tumors as benign/malignant\
- **Airport Security:** Flag suspicious objects from scanned images\
- **Finance:** Fraudulent transaction detection\
- **Social Media:** Content recommendation & trend detection\
\
---\
\
## \uc0\u55357 \u56526  Files Included\
\
- `svm_nn_final_script.R`: Full modeling pipeline with all steps and comments\
- `visualizations/`: Final plots of decision boundaries\
- `README.md`: This documentation\
- `INFO6105_Final_Presentation.pdf`: 3-minute project walkthrough\
\
---\
\
## \uc0\u55357 \u56538  References\
\
- CMU StatLib Dataset: [PRNN](https://lib.stat.cmu.edu/datasets/)\
- Ripley, B. (1996). *Pattern Recognition and Neural Networks*. Cambridge University Press.\
- R Documentation: [`e1071`](https://cran.r-project.org/web/packages/e1071/index.html), [`nnet`](https://cran.r-project.org/web/packages/nnet/index.html), [`caret`](https://cran.r-project.org/web/packages/caret/index.html)\
- Northeastern INFO6105 Lecture Notes\
\
---\
\
## \uc0\u55357 \u56424 \u8205 \u55357 \u56507  Author\
\
**Aravind Ravi**  \
MS in Information Systems  \
[GitHub Profile](https://github.com/aravindravi7)\
\
---\
}