# 📊 Binary Classification using Support Vector Machines (SVM) and Neural Networks (NN)

![2](https://github.com/user-attachments/assets/053e2679-c034-4909-9c56-3a343a1eac26)
![3](https://github.com/user-attachments/assets/6ca00a74-5d73-4153-86f7-adffc0cad637)
![4](https://github.com/user-attachments/assets/e2496ad2-0c0f-4226-8d66-ef7f41cda580)
![5](https://github.com/user-attachments/assets/0f912547-0a88-4e1e-8eaf-afb7974fd6e1)
![6](https://github.com/user-attachments/assets/45308198-b364-4ce4-b743-d2ad1f2cf7b2)
![7](https://github.com/user-attachments/assets/db0a4f70-4fb5-4cf6-94ed-479b81871f0f)
![8](https://github.com/user-attachments/assets/da97ef1f-0c64-4a0f-8d13-6d745c8cac5c)
![9](https://github.com/user-attachments/assets/6d24d99b-ae86-4f0f-8458-59f35e1c28f8)
![10](https://github.com/user-attachments/assets/46c590d7-7b42-420d-af95-ed36bc0681c1)


## 🧠 Project Overview

This project compares the performance of two supervised learning models — **Support Vector Machines (SVM)** and **Neural Networks (NN)** — on a synthetic 2D binary classification dataset. The dataset is sourced from the CMU StatLib repository, specifically designed for evaluating pattern recognition models with nonlinear boundaries.

Through a series of experiments and visualizations, we analyze decision boundary formation, the effect of hyperparameter tuning, and overall classification performance.

---

## 🎯 Objectives

1. **Can we accurately classify the target using SVM and NN?**
2. **Which hyperparameters significantly influence the model's performance?**
3. **Which model generalizes better to unseen data?**

---

## 📁 Dataset

- **Source:** [CMU StatLib – PRNN Synthetic Dataset](https://lib.stat.cmu.edu/datasets/)
- `synth.tr.txt`: 250 training samples
- `synth.te.txt`: 1000 test samples
- **Features:** `xs`, `ys`
- **Target:** Binary class (`0` or `1`)

---

## ⚙️ Preprocessing

- Removed identifier column
- Converted string values to numeric
- Applied **Z-score standardization**:
  - Mean = 0
  - Standard Deviation = 1
- Ensured feature ranges are consistent for distance-based models like SVM and NN

---

## 🧪 Modeling and Tools

| Task                  | Method / Package            |
|-----------------------|-----------------------------|
| SVM Classification    | `e1071::svm()`              |
| Neural Network (NN)   | `nnet::nnet()`              |
| Evaluation Metrics    | `caret::confusionMatrix()`  |
| Visualization         | `ggplot2`, `gridExtra`      |

---

## 🔧 Tuning Experiments

### 🌀 SVM Tuning
- **Kernels tested:** radial, polynomial, sigmoid
- **Parameters:**
  - Cost: `0.1` to `25`
  - Gamma: `0.0005` to `0.6`
- **Best Result:**
  - **Kernel:** RBF
  - **Cost:** 4
  - **Gamma:** 0.6
  - **Accuracy:** 92.3%

### 🤖 NN Tuning
- **Neurons tested:** 5 to 200 (step of 5)
- **Decay values tested:** `0.01`, `0.001`, `0.0001`
- **Best Result:**
  - **Neurons:** 50
  - **Decay:** 0.0001
  - **Accuracy:** 100%

---

## 📈 Performance Comparison

| Model           | Configuration            | Accuracy | Sensitivity | Specificity | Kappa | Balanced Acc. |
|----------------|---------------------------|----------|-------------|-------------|--------|----------------|
| Initial SVM     | RBF, C=1, γ=0.5          | 91.4%    | 92.2%       | 90.6%       | 0.828 | 91.4%          |
| Tuned SVM       | RBF, C=4, γ=0.6          | 92.3%    | 91.4%       | 93.2%       | 0.846 | 92.3%          |
| Initial NN      | 5 neurons, decay=0.01    | 92.1%    | 91.6%       | 92.6%       | 0.842 | 92.1%          |
| Tuned NN        | 50 neurons, decay=0.0001 | 100%     | 100%        | 100%        | 1.000 | 100%           |

---

## 📊 Visualizations

- **Initial SVM vs Initial NN**: Highlights early-stage performance differences.
![InitialSVMvsNN](https://github.com/user-attachments/assets/ce20de8b-423c-44b7-8c4b-f880162c996b)


- **Fine-Tuned SVM**: Shows marginal improvement with optimal gamma & cost.
![InitialSVMvsFinalSVM_91 4vs92 3](https://github.com/user-attachments/assets/5c6afaeb-f890-4a3e-9d44-2c3e91364a0e)


- **Fine-Tuned NN**: Shows complete separation and perfect classification with 50+ neurons and low decay.
![UpdatedNN_92 4vs100](https://github.com/user-attachments/assets/d6d3c3e0-b87e-42b2-939d-c87ebde6003b)


> 🖼️ Decision boundary plots included in `/visualizations/` folder.

---

## 💡 Key Insights

- SVMs are powerful but sensitive to `gamma` and `cost`. Improvements plateaued around ~92%.
- Neural Networks generalized better when tuned correctly. Lower decay allowed higher learning flexibility.
- Final NN setup achieved **perfect test accuracy**, with training misclassifications < 0.5%, indicating **healthy generalization**.
- Visualization of decision boundaries helped assess and interpret model behavior.

---

## 🌍 Real-World Applications

- **Medical Diagnosis:** Classify tumors as benign/malignant
- **Airport Security:** Flag suspicious objects from scanned images
- **Finance:** Fraudulent transaction detection
- **Social Media:** Content recommendation & trend detection

---

## 📎 Files Included

- `svm_nn_final_script.R`: Full modeling pipeline with all steps and comments
- `visualizations/`: Final plots of decision boundaries
- `README.md`: This documentation
- `INFO6105_Final_Presentation.pdf`: 3-minute project walkthrough

---

## 📚 References

- CMU StatLib Dataset: [PRNN](https://lib.stat.cmu.edu/datasets/)
- Ripley, B. (1996). *Pattern Recognition and Neural Networks*. Cambridge University Press.
- R Documentation: [`e1071`](https://cran.r-project.org/web/packages/e1071/index.html), [`nnet`](https://cran.r-project.org/web/packages/nnet/index.html), [`caret`](https://cran.r-project.org/web/packages/caret/index.html)
- Northeastern INFO6105 Lecture Notes

---

## 👨‍💻 Author

**Aravind Ravi**  


---
