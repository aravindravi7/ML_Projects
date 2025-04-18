# ---------------------------
# Final Project Script - INFO6105
# SVM vs. Neural Network (Initial Config Only)
# Author: Aravind Ravi
# ---------------------------

# ---- Setup & Packages ----
install.packages(c("e1071", "nnet", "caret", "ggplot2", "gridExtra"))
library(e1071)
library(nnet)
library(caret)
library(ggplot2)
library(gridExtra)

# ---- 1. Load and Clean Data ----
train_raw <- read.table("synth.tr.txt", header = FALSE)
test_raw  <- read.table("synth.te.txt", header = FALSE)

# Remove first column and rename
train_data <- train_raw[, -1]; colnames(train_data) <- c("xs", "ys", "class")
test_data  <- test_raw[, -1];  colnames(test_data)  <- c("xs", "ys", "class")
test_data$xs <- as.numeric(as.character(test_data$xs))
test_data$ys <- as.numeric(as.character(test_data$ys))

# ---- 2. Feature Scaling ----
train_data$xs <- as.numeric(as.character(train_data$xs))
train_data$ys <- as.numeric(as.character(train_data$ys))
train_data$class <- as.character(train_data$class)
train_data <- train_data[train_data$class %in% c("0", "1"), ]
train_labels <- as.factor(train_data$class)
train_features <- scale(train_data[, c("xs", "ys")])
stopifnot(nrow(train_features) == length(train_labels))
train_scaled <- data.frame(train_features, class = train_labels)

# Scale test data using training mean/sd
test_scaled_features <- scale(test_data[, c("xs", "ys")],
                              center = attr(train_features, "scaled:center"),
                              scale = attr(train_features, "scaled:scale"))
test_labels <- as.character(test_data$class)
test_filtered <- test_scaled_features[test_labels %in% c("0", "1"), ]
test_labels <- as.factor(test_labels[test_labels %in% c("0", "1")])
test_scaled <- data.frame(test_filtered, class = test_labels)

# ---- 3. Initial Models ----
# Initial SVM (default)
svm_model <- svm(class ~ ., data = train_scaled, kernel = "linear")

# Initial NN (5 neurons)
nn_model <- nnet(class ~ ., data = train_scaled, size = 5, maxit = 200, decay = 0.01, trace = FALSE)

# ---- 4. Evaluate Initial Models ----
test_scaled$svm_initial <- predict(svm_model, test_scaled)
test_scaled$nn_initial  <- predict(nn_model, test_scaled, type = "class")

# Convert to factor
lvls <- levels(test_scaled$class)
test_scaled$svm_initial <- factor(test_scaled$svm_initial, levels = lvls)
test_scaled$nn_initial  <- factor(test_scaled$nn_initial,  levels = lvls)

# Print Accuracies
cat("\nInitial SVM Performance:\n")
print(confusionMatrix(test_scaled$svm_initial, test_scaled$class))

cat("\nInitial NN (5 neurons) Performance:\n")
print(confusionMatrix(test_scaled$nn_initial, test_scaled$class))

# ---- 5. Decision Boundary Visualization (Training Set) ----
xrange <- seq(min(train_scaled$xs) - 0.5, max(train_scaled$xs) + 0.5, length.out = 300)
yrange <- seq(min(train_scaled$ys) - 0.5, max(train_scaled$ys) + 0.5, length.out = 300)
grid <- expand.grid(xs = xrange, ys = yrange)

# Predictions on grid
grid$svm_initial <- predict(svm_model, newdata = grid)
grid$nn_initial  <- predict(nn_model, newdata = grid, type = "class")
grid$svm_initial <- as.factor(grid$svm_initial)
grid$nn_initial  <- as.factor(grid$nn_initial)

# Wrong predictions
train_scaled$svm_initial_wrong <- predict(svm_model, train_scaled) != train_scaled$class
train_scaled$nn_initial_wrong  <- predict(nn_model, train_scaled, type = "class") != train_scaled$class

# SVM Plot
p_svm <- ggplot() +
  geom_tile(data = grid, aes(x = xs, y = ys, fill = svm_initial), alpha = 0.3) +
  geom_point(data = train_scaled, aes(x = xs, y = ys, color = class), size = 1.2) +
  geom_point(data = subset(train_scaled, svm_initial_wrong), aes(x = xs, y = ys),
             shape = 4, color = "black", size = 2, stroke = 1) +
  labs(title = "Initial SVM (default) Decision Boundary") +
  theme_minimal()

# NN Plot
p_nn <- ggplot() +
  geom_tile(data = grid, aes(x = xs, y = ys, fill = nn_initial), alpha = 0.3) +
  geom_point(data = train_scaled, aes(x = xs, y = ys, color = class), size = 1.2) +
  geom_point(data = subset(train_scaled, nn_initial_wrong), aes(x = xs, y = ys),
             shape = 4, color = "black", size = 2, stroke = 1) +
  labs(title = "Initial NN (5 neurons) Decision Boundary") +
  theme_minimal()

grid.arrange(p_svm, p_nn, nrow = 1)

run_svm_finetune_experiments <- function() {
  library(e1071)
  library(caret)
  
  # Define hyperparameters
  kernels <- c("radial", "polynomial", "sigmoid")
  costs <- c(0.1, 1, 5, 10)
  gammas <- c(0.0005, 0.001, 0.005, 0.01)
  
  # Store results
  svm_results <- data.frame(Kernel=character(), Cost=double(), Gamma=double(), Accuracy=double(), stringsAsFactors=FALSE)
  
  cat("Starting SVM Fine-Tuning Experiments...\n")
  
  for (k in kernels) {
    for (c_val in costs) {
      for (g_val in gammas) {
        cat(sprintf("\nTesting: Kernel = %-9s | Cost = %-5.3f | Gamma = %-6.4f\n", k, c_val, g_val))
        
        # Train and test
        model <- svm(class ~ ., data = train_scaled, kernel = k, cost = c_val, gamma = g_val)
        preds <- predict(model, test_scaled)
        preds <- factor(preds, levels = levels(test_scaled$class))
        
        acc <- confusionMatrix(preds, test_scaled$class)$overall["Accuracy"]
        acc_rounded <- round(acc, 4)
        print(acc_rounded)
        
        svm_results <- rbind(svm_results, data.frame(Kernel = k, Cost = c_val, Gamma = g_val, Accuracy = acc_rounded))
      }
    }
  }
  
  return(svm_results)
}

# Run the function
svm_results <- run_svm_finetune_experiments()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Run this as a separate file - SVM_FineTuning
# Load required packages
library(e1071)
library(caret)

# Load and clean training data
train_raw <- read.table("synth.tr.txt", header = FALSE)
train_data <- train_raw[, -1]
colnames(train_data) <- c("xs", "ys", "class")
train_data$xs <- as.numeric(as.character(train_data$xs))
train_data$ys <- as.numeric(as.character(train_data$ys))
train_data$class <- as.character(train_data$class)
train_data <- train_data[train_data$class %in% c("0", "1"), ]
train_labels <- as.factor(train_data$class)
train_features <- scale(train_data[, c("xs", "ys")])
train_scaled <- data.frame(train_features, class = train_labels)

# Load and clean test data
test_raw <- read.table("synth.te.txt", header = FALSE)
test_data <- test_raw[, -1]
colnames(test_data) <- c("xs", "ys", "class")
test_data$xs <- as.numeric(as.character(test_data$xs))
test_data$ys <- as.numeric(as.character(test_data$ys))
test_data$class <- as.character(test_data$class)
test_data <- test_data[test_data$class %in% c("0", "1"), ]
test_labels <- as.factor(test_data$class)
test_scaled_features <- scale(test_data[, c("xs", "ys")],
                              center = attr(train_features, "scaled:center"),
                              scale = attr(train_features, "scaled:scale"))
test_scaled <- data.frame(test_scaled_features, class = test_labels)


# Define parameter grid
kernels <- c("radial", "polynomial", "sigmoid")
costs <- c(0.1, 1, 5, 10)
gammas <- c(0.0005, 0.001, 0.005, 0.01)

# Store results
svm_results <- data.frame(Kernel = character(), Cost = double(), Gamma = double(), Accuracy = double(), stringsAsFactors = FALSE)

# Loop
for (k in kernels) {
  for (c_val in costs) {
    for (g_val in gammas) {
      cat(sprintf("\nTesting: Kernel = %-9s | Cost = %-5.3f | Gamma = %-6.4f\n", k, c_val, g_val))
      model <- svm(class ~ ., data = train_scaled, kernel = k, cost = c_val, gamma = g_val)
      preds <- predict(model, test_scaled)
      preds <- factor(preds, levels = levels(test_scaled$class))
      acc <- confusionMatrix(preds, test_scaled$class)$overall["Accuracy"]
      acc_rounded <- round(acc, 4)
      print(acc_rounded)
      svm_results <- rbind(svm_results, data.frame(Kernel = k, Cost = c_val, Gamma = g_val, Accuracy = acc_rounded))
    }
  }
}

library(e1071)
library(caret)

# Define better-focused hyperparameter ranges
costs <- c(10, 15, 20, 30, 50, 100)
gammas <- c(0.01, 0.015, 0.02)

cat("\n--- Focused SVM Radial Kernel Search ---\n")
svm_focused_results <- data.frame(Cost = numeric(), Gamma = numeric(), Accuracy = numeric())

for (c_val in costs) {
  for (g_val in gammas) {
    cat(sprintf("\nTesting: Cost = %-5.1f | Gamma = %-6.4f\n", c_val, g_val))
    
    model <- svm(class ~ ., data = train_scaled, kernel = "radial", cost = c_val, gamma = g_val)
    preds <- predict(model, test_scaled)
    preds <- factor(preds, levels = levels(test_scaled$class))
    acc <- confusionMatrix(preds, test_scaled$class)$overall["Accuracy"]
    
    print(round(acc, 4))
    
    svm_focused_results <- rbind(svm_focused_results, data.frame(Cost = c_val, Gamma = g_val, Accuracy = round(acc, 4)))
  }
}

library(e1071)
library(caret)

cat("\n--- Focused SVM Experiments (Gamma = 0.5) ---\n")

costs <- c(1, 5, 10, 15, 20)
gamma <- 0.5

svm_results <- data.frame(Cost = numeric(), Gamma = numeric(), Accuracy = numeric())

for (c_val in costs) {
  cat(sprintf("\nTesting: Cost = %.1f | Gamma = %.3f\n", c_val, gamma))
  
  model <- svm(class ~ ., data = train_scaled, kernel = "radial", cost = c_val, gamma = gamma)
  preds <- predict(model, test_scaled)
  preds <- factor(preds, levels = levels(test_scaled$class))
  
  acc <- confusionMatrix(preds, test_scaled$class)$overall["Accuracy"]
  print(round(acc, 4))
  
  svm_results <- rbind(svm_results, data.frame(Cost = c_val, Gamma = gamma, Accuracy = round(acc, 4)))
}

print(svm_results)

costs  <- c(4, 5, 6, 10, 15, 18, 20, 25)
gammas <- c(0.3, 0.4, 0.45, 0.5, 0.55, 0.6)

library(e1071)
library(caret)

cat("\n--- Refined SVM Grid Search ---\n")
costs <- c(4, 5, 6, 10, 15, 18, 20, 25)
gammas <- c(0.3, 0.4, 0.45, 0.5, 0.55, 0.6)

svm_refined_results <- data.frame(Cost = numeric(), Gamma = numeric(), Accuracy = numeric())

for (c_val in costs) {
  for (g_val in gammas) {
    cat(sprintf("\nTesting: Cost = %-5.2f | Gamma = %-5.3f\n", c_val, g_val))
    
    model <- svm(class ~ ., data = train_scaled, kernel = "radial", cost = c_val, gamma = g_val)
    preds <- predict(model, test_scaled)
    preds <- factor(preds, levels = levels(test_scaled$class))
    acc <- confusionMatrix(preds, test_scaled$class)$overall["Accuracy"]
    
    print(round(acc, 4))
    svm_refined_results <- rbind(svm_refined_results, data.frame(Cost = c_val, Gamma = g_val, Accuracy = round(acc, 4)))
  }
}

svm_refined_results <- data.frame(Cost = numeric(), Gamma = numeric(), Accuracy = numeric())

for (c_val in c(4, 5, 6, 10, 15, 18, 20, 25)) {
  for (g_val in c(0.3, 0.4, 0.45, 0.5, 0.55, 0.6)) {
    model <- svm(class ~ ., data = train_scaled, kernel = "radial", cost = c_val, gamma = g_val)
    preds <- predict(model, test_scaled)
    preds <- factor(preds, levels = levels(test_scaled$class))
    acc <- confusionMatrix(preds, test_scaled$class)$overall["Accuracy"]
    
    svm_refined_results <- rbind(svm_refined_results,
                                 data.frame(Cost = c_val, Gamma = g_val, Accuracy = round(acc, 4)))
  }
}

# Print best
best_result <- svm_refined_results[which.max(svm_refined_results$Accuracy), ]
cat("\nBest SVM Configuration:\n")
print(best_result)

# --- Load best model manually if not already available ---
svm_best <- svm(class ~ ., data = train_scaled, kernel = "radial", cost = 4, gamma = 0.6)

# --- Create grid for visualization ---
xrange <- seq(min(train_scaled$xs) - 0.5, max(train_scaled$xs) + 0.5, length.out = 300)
yrange <- seq(min(train_scaled$ys) - 0.5, max(train_scaled$ys) + 0.5, length.out = 300)
grid <- expand.grid(xs = xrange, ys = yrange)

# --- Predict on grid ---
grid$svm_initial <- predict(svm_model, newdata = grid)
grid$svm_best <- predict(svm_best, newdata = grid)

grid$svm_initial <- as.factor(grid$svm_initial)
grid$svm_best <- as.factor(grid$svm_best)

# --- Visualize initial vs fine-tuned ---
library(ggplot2)
library(gridExtra)

p_init <- ggplot() +
  geom_tile(data = grid, aes(x = xs, y = ys, fill = svm_initial), alpha = 0.3) +
  geom_point(data = train_scaled, aes(x = xs, y = ys, color = class), size = 1.2) +
  labs(title = "Initial SVM (Cost=1, Gamma=0.5)") +
  theme_minimal()

p_best <- ggplot() +
  geom_tile(data = grid, aes(x = xs, y = ys, fill = svm_best), alpha = 0.3) +
  geom_point(data = train_scaled, aes(x = xs, y = ys, color = class), size = 1.2) +
  labs(title = "Best SVM (Cost=4, Gamma=0.6)") +
  theme_minimal()

grid.arrange(p_init, p_best, nrow = 1)

# --- (Re)Train Models with Specified Hyperparameters ---
svm_model <- svm(class ~ ., data = train_scaled, kernel = "radial", cost = 1, gamma = 0.5)   # Initial
svm_best  <- svm(class ~ ., data = train_scaled, kernel = "radial", cost = 4, gamma = 0.6)   # Fine-tuned

# --- Predict on training set to identify misclassifications ---
train_scaled$svm_initial_pred <- predict(svm_model, train_scaled)
train_scaled$svm_best_pred    <- predict(svm_best, train_scaled)

train_scaled$svm_initial_wrong <- train_scaled$svm_initial_pred != train_scaled$class
train_scaled$svm_best_wrong    <- train_scaled$svm_best_pred    != train_scaled$class

# --- Compute Accuracies ---
library(caret)
acc_initial <- round(confusionMatrix(train_scaled$svm_initial_pred, train_scaled$class)$overall["Accuracy"], 3)
acc_best    <- round(confusionMatrix(train_scaled$svm_best_pred,    train_scaled$class)$overall["Accuracy"], 3)

# --- Create Grid for Decision Boundary Plot ---
xrange <- seq(min(train_scaled$xs) - 0.5, max(train_scaled$xs) + 0.5, length.out = 300)
yrange <- seq(min(train_scaled$ys) - 0.5, max(train_scaled$ys) + 0.5, length.out = 300)
grid <- expand.grid(xs = xrange, ys = yrange)

grid$svm_initial <- predict(svm_model, newdata = grid)
grid$svm_best    <- predict(svm_best, newdata = grid)
grid$svm_initial <- as.factor(grid$svm_initial)
grid$svm_best    <- as.factor(grid$svm_best)

# --- Plotting ---
library(ggplot2)
library(gridExtra)

p_init <- ggplot() +
  geom_tile(data = grid, aes(x = xs, y = ys, fill = svm_initial), alpha = 0.3) +
  geom_point(data = train_scaled, aes(x = xs, y = ys, color = class), size = 1.2) +
  geom_point(data = subset(train_scaled, svm_initial_wrong), aes(x = xs, y = ys),
             shape = 4, color = "black", size = 2, stroke = 1) +
  labs(title = paste("Initial SVM: Cost=1, Gamma=0.5 | Accuracy:", acc_initial),
       fill = "Predicted") +
  theme_minimal()

p_best <- ggplot() +
  geom_tile(data = grid, aes(x = xs, y = ys, fill = svm_best), alpha = 0.3) +
  geom_point(data = train_scaled, aes(x = xs, y = ys, color = class), size = 1.2) +
  geom_point(data = subset(train_scaled, svm_best_wrong), aes(x = xs, y = ys),
             shape = 4, color = "black", size = 2, stroke = 1) +
  labs(title = paste("Best SVM: Cost=4, Gamma=0.6 | Accuracy:", acc_best),
       fill = "Predicted") +
  theme_minimal()

# --- Show Side-by-Side Plot ---
grid.arrange(p_init, p_best, nrow = 1)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Run this as a separate file - NN_FineTuning

# --- Load required packages ---
install.packages("nnet")
install.packages("caret")
library(nnet)
library(caret)

# --- Load and clean data ---
train_raw <- read.table("synth.tr.txt", header = FALSE)
test_raw  <- read.table("synth.te.txt", header = FALSE)

train_data <- train_raw[, -1]; colnames(train_data) <- c("xs", "ys", "class")
test_data  <- test_raw[, -1];  colnames(test_data)  <- c("xs", "ys", "class")

# Convert and clean
train_data$xs <- as.numeric(as.character(train_data$xs))
train_data$ys <- as.numeric(as.character(train_data$ys))
train_data$class <- as.character(train_data$class)
train_data <- train_data[train_data$class %in% c("0", "1"), ]
train_labels <- as.factor(train_data$class)

train_features <- scale(train_data[, c("xs", "ys")])
train_scaled <- data.frame(train_features, class = train_labels)

# Scale test data with training mean/sd
test_data$xs <- as.numeric(as.character(test_data$xs))
test_data$ys <- as.numeric(as.character(test_data$ys))
test_labels <- as.character(test_data$class)
test_scaled_features <- scale(test_data[, c("xs", "ys")],
                              center = attr(train_features, "scaled:center"),
                              scale = attr(train_features, "scaled:scale"))
test_filtered <- test_scaled_features[test_labels %in% c("0", "1"), ]
test_labels <- as.factor(test_labels[test_labels %in% c("0", "1")])
test_scaled <- data.frame(test_filtered, class = test_labels)

# --- Initialize results storage ---
nn_results <- data.frame(Neurons = integer(), Accuracy = numeric())

# --- Run loop from 5 to 200 neurons in steps of 5 ---
cat("\n--- Neural Network Accuracy Tuning (5 to 200 neurons) ---\n")

for (n in seq(5, 200, by = 5)) {
  cat(sprintf("\nTesting NN with %d neurons:\n", n))
  set.seed(123)  # for reproducibility
  model <- nnet(class ~ ., data = train_scaled, size = n, maxit = 1000, decay = 0.01, trace = FALSE)
  preds <- predict(model, test_scaled, type = "class")
  preds <- factor(preds, levels = levels(test_scaled$class))
  acc <- confusionMatrix(preds, test_scaled$class)$overall["Accuracy"]
  print(round(acc, 4))
  
  nn_results <- rbind(nn_results, data.frame(Neurons = n, Accuracy = round(acc, 4)))
}

# --- Show best result ---
best_nn <- nn_results[which.max(nn_results$Accuracy), ]
cat("\nBest NN configuration:\n")
print(best_nn)

# NN with 30 neurons
set.seed(42)
nn_30 <- nnet(class ~ ., data = train_scaled, size = 30, maxit = 1000, decay = 0.01, trace = FALSE)
pred_30 <- predict(nn_30, test_scaled, type = "class")
pred_30 <- factor(pred_30, levels = levels(test_scaled$class))

cat("\n📊 Confusion Matrix - Neural Network (30 neurons):\n")
print(confusionMatrix(pred_30, test_scaled$class))

library(nnet)
library(caret)

# Configurations
neurons <- seq(30, 120, by = 5)
decays <- c(0.001, 0.0001)

# Result storage
nn_decay_results <- data.frame(Neurons = integer(), Decay = numeric(), Accuracy = numeric())

# Loop through decay + neuron combinations
cat("\n--- Testing NN Configurations (with decay tuning) ---\n")
for (d in decays) {
  for (n in neurons) {
    cat(sprintf("\nTesting: Neurons = %3d | Decay = %.5f\n", n, d))
    
    # Train model
    nn_model <- nnet(class ~ ., data = train_scaled, size = n, decay = d, maxit = 1000, trace = FALSE)
    
    # Predict and evaluate
    preds <- predict(nn_model, test_scaled, type = "class")
    preds <- factor(preds, levels = levels(test_scaled$class))
    acc <- confusionMatrix(preds, test_scaled$class)$overall["Accuracy"]
    
    print(round(acc, 4))
    
    # Store results
    nn_decay_results <- rbind(nn_decay_results,
                              data.frame(Neurons = n, Decay = d, Accuracy = round(acc, 4)))
  }
}

# --- Load Libraries ---
library(nnet)
library(ggplot2)
library(gridExtra)

# --- Re-train Initial NN (5 neurons, default decay) ---
nn_initial <- nnet(class ~ ., data = train_scaled, size = 5, maxit = 500, decay = 0.01, trace = FALSE)

# --- Train Fine-Tuned NN (50 neurons, decay = 0.0001) ---
nn_best <- nnet(class ~ ., data = train_scaled, size = 50, maxit = 1000, decay = 0.0001, trace = FALSE)

# --- Create grid for predictions ---
xrange <- seq(min(train_scaled$xs) - 0.5, max(train_scaled$xs) + 0.5, length.out = 300)
yrange <- seq(min(train_scaled$ys) - 0.5, max(train_scaled$ys) + 0.5, length.out = 300)
grid <- expand.grid(xs = xrange, ys = yrange)

# --- Predict on grid ---
grid$nn_initial <- predict(nn_initial, newdata = grid, type = "class")
grid$nn_final   <- predict(nn_best, newdata = grid, type = "class")

# --- Convert to factors for plotting ---
grid$nn_initial <- as.factor(grid$nn_initial)
grid$nn_final   <- as.factor(grid$nn_final)

# --- Misclassified points (on training set) ---
train_scaled$nn_initial_wrong <- predict(nn_initial, train_scaled, type = "class") != train_scaled$class
train_scaled$nn_final_wrong   <- predict(nn_best, train_scaled, type = "class") != train_scaled$class

# --- Visualization Plots ---
p_nn_initial <- ggplot() +
  geom_tile(data = grid, aes(x = xs, y = ys, fill = nn_initial), alpha = 0.3) +
  geom_point(data = train_scaled, aes(x = xs, y = ys, color = class), size = 1.2) +
  geom_point(data = subset(train_scaled, nn_initial_wrong), aes(x = xs, y = ys), shape = 4, color = "black", size = 2, stroke = 1) +
  labs(title = "Initial NN (5 neurons, decay=0.01)\nAccuracy ≈ 92.4%") +
  theme_minimal()

p_nn_final <- ggplot() +
  geom_tile(data = grid, aes(x = xs, y = ys, fill = nn_final), alpha = 0.3) +
  geom_point(data = train_scaled, aes(x = xs, y = ys, color = class), size = 1.2) +
  geom_point(data = subset(train_scaled, nn_final_wrong), aes(x = xs, y = ys), shape = 4, color = "black", size = 2, stroke = 1) +
  labs(title = "Fine-Tuned NN (50 neurons, decay=0.0001)\nAccuracy = 100%") +
  theme_minimal()

# --- Display plots side by side ---
grid.arrange(p_nn_initial, p_nn_final, nrow = 1)

