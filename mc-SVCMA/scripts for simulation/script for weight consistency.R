#################### Loading packages ######################
library(caret)      
library(ggplot2)    
library(reshape2)   
library(pROC)       
library(gridExtra)  
library(RColorBrewer) 
library(PRROC)
library(LiblineaR)
library(nloptr)
library(mvtnorm)
set.seed(123)
nism <- 100
##################################

# Multi-class hinge loss function
# Computes the average multi-class hinge loss for given scores
multi_hinge_loss <- function(y_true, scores) {
  n <- length(y_true)
  J <- ncol(scores)
  loss <- 0
  for (i in 1:n) {
    true_c <- as.integer(as.character(y_true[i]))
    true_score <- scores[i, true_c]
    max_margin <- 0
    for (c in 1:J) {
      if (c == true_c) next
      margin <- max(0,1 - true_score + scores[i, c])
      if (margin > max_margin) max_margin <- margin
    }
    loss <- loss + max_margin
  }
  return(loss / n)
}

# Train multi-class SVM model using LiblineaR
# Returns model coefficients, scores, and training loss
svm_model <- function(X_train, Y_train, C_val) {
  Y_train <- as.factor(Y_train)
  J <- length(levels(Y_train))
  clf <- LiblineaR(data = X_train, 
                   target = Y_train,  
                   type = 4,    
                   cost = C_val, 
                   bias = 1)
  
  W <- clf$W 
  beta <- cbind(W[, ncol(W)], W[, -ncol(W)])
  
  target_order <- levels(Y_train)
  class_order <- as.character(clf$ClassNames)
  match_idx <- match(target_order, class_order)
  beta <- beta[match_idx, , drop = FALSE]
  
  X_s <- cbind(1, X_train)
  scores <- X_s %*% t(beta)
  loss <- multi_hinge_loss(Y_train, scores)
  
  return(list(
    model = clf,
    beta = beta,
    scores = scores,
    loss = loss,
    class_order = target_order 
  ))
}

# Generate synthetic three-class data with Gaussian distributions
# Features are generated from class-specific means with identity covariance
Data1 <- function(n, p, q, c0){
  Y1 <- sample(c(1, 2, 3), n, replace = TRUE, prob = c(1/3, 1/3, 1/3))
  Sigma <- diag(p)
  mu1 <- c(rep(0.6, q-1), c0, rep(0, p - q))
  mu2 <- c(rep(0, q), rep(0, p - q))      
  mu3 <- c(rep(-0.6, q-1), -c0, rep(0, p - q))
  X1 <- matrix(0, n, p)
  idx1 <- which(Y1 == 1)
  idx2 <- which(Y1 == 2)
  idx3 <- which(Y1 == 3)
  X1[idx1, ] <- mvtnorm::rmvnorm(length(idx1), mean = mu1, sigma = Sigma)
  X1[idx2, ] <- mvtnorm::rmvnorm(length(idx2), mean = mu2, sigma = Sigma)
  X1[idx3, ] <- mvtnorm::rmvnorm(length(idx3), mean = mu3, sigma = Sigma)
  return(list(
    X1 = X1,
    Y1 = Y1
  ))
}

n <- 500
p1 <- 50
q <- 4
J <- 3
K <- 5
train_ratio <- 0.8
c0_values <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
weight_correct <- matrix(NA, nrow = nism, ncol = length(c0_values))

for (idx_c in 1:length(c0_values)) {
  c0 <- c0_values[idx_c]
  Data <- Data1(n, p1, q, c0)
  X <- Data$X1
  Y1 <- Data$Y1
  p <- ncol(X)
  Y <- as.factor(Y1)
  S <- 2
  
  pb <- txtProgressBar(min = 0, max = nism, style = 3)
  
  for (m in 1:nism) {
    # Split data into training and testing sets
    train_idx <- caret::createDataPartition(Y, p = train_ratio, list = FALSE)
    X_train <- X[train_idx, ]
    y_train <- Y[train_idx]
    X_test  <- X[-train_idx, ]
    y_test  <- Y[-train_idx]
    X_test <- cbind(1, X_test)
    C_val <- 1
    
    # Define feature subsets for candidate models
    feat_cols <- list()
    feat_cols[[1]] <- 1:3   
    feat_cols[[2]] <- 1:4
    
    beta_matrix <- array(0, dim = c(J, p + 1, S))
    
    # Train models on each feature subset
    for (s in 1:S) {
      fcols <- feat_cols[[s]]
      X_train_s <- X_train[, fcols, drop = FALSE]
      svm_result_s <- svm_model(X_train_s, y_train, C_val)
      beta <- svm_result_s$beta
      beta_matrix[, 1, s] <- beta[, 1]
      beta_matrix[, feat_cols[[s]] + 1, s] <- beta[, -1]
    }
    
    # Cross-validation to compute optimal weights
    n1 <- length(y_train)
    cv_beta_matrix <- array(0, dim = c(J, p + 1, S, K))
    folds <- createFolds(y_train, k = K, list = TRUE, returnTrain = FALSE)
    cv_X_test <- list()
    cv_y_test <- list()
    X_train_total <- X_train
    
    for (k in 1:K) {
      A_k <- folds[[k]]
      B_k <- setdiff(1:n1, A_k)
      y_train_fold <- y_train[B_k]
      y_test_fold <- y_train[A_k]
      X_train_fold <- X_train_total[B_k, , drop = FALSE]
      X_test_fold <- cbind(1, X_train_total[A_k, ])
      cv_y_test[[k]] <- y_test_fold
      cv_X_test[[k]] <- X_test_fold
      cv_Cval <- 1
      
      for (s in 1:S) {
        X_train_s <- X_train_fold[, feat_cols[[s]], drop = FALSE]
        svm_result_fold <- svm_model(X_train_s, y_train_fold, cv_Cval)
        cv_beta <- svm_result_fold$beta
        cv_beta_matrix[, 1, s, k] <- cv_beta[, 1]
        cv_beta_matrix[, feat_cols[[s]] + 1, s, k] <- cv_beta[, -1]
      }
    }
    
    # Objective function for cross-validation loss
    w_obj <- function(w) {
      cv_loss <- 0
      for (k in 1:K) {
        beta_k <- cv_beta_matrix[, , , k]
        wbeta_k <- matrix(0, J, p + 1)
        for (s in 1:S) {
          wbeta_k <- wbeta_k + w[s] * beta_k[, , s]
        }
        X_test_s <- cv_X_test[[k]]
        y_test_fold <- cv_y_test[[k]]
        cv_scores <- X_test_s %*% t(wbeta_k)
        cv_loss <- cv_loss + multi_hinge_loss(y_test_fold, cv_scores) * length(y_test_fold)
      }
      return(cv_loss / n1)
    }
    
    # Equality constraint: sum of weights = 1
    heq <- function(w) { 
      sum(w) - 1 
    }
    
    # Optimize weights using COBYLA algorithm
    p0 <- rep(1 / S, S)
    res <- nloptr(
      x0 = p0,
      eval_f = w_obj,
      lb = rep(0, S),
      ub = rep(1, S),
      eval_g_eq = heq,
      opts = list(
        algorithm = "NLOPT_LN_COBYLA",
        xtol_rel = 1e-7,
        maxeval = 3000,
        print_level = 0
      )
    )
    
    cv_weight <- res$solution
    weight_correct[m, idx_c] <- cv_weight[2]  # Weight assigned to the correct model (Model 2)
    
    setTxtProgressBar(pb, m)
  }
  
  close(pb)
  cat("c0 =", c0, "mean weight =", mean(weight_correct[, idx_c]), "\n")
}

# Create boxplot data
c0_labels <- c0_values
df_boxplot <- data.frame(
  Repetition = rep(1:nism, length(c0_values)),
  c0 = rep(c0_values, each = nism),
  Weight = as.vector(weight_correct)
)
df_boxplot$c0 <- factor(df_boxplot$c0, levels = c0_values)

# Generate boxplot showing weight distribution across c0 values
p_box <- ggplot(df_boxplot, aes(x = c0, y = Weight)) +
  stat_boxplot(geom = "errorbar", width = 0.25, color = "gray50") +
  geom_boxplot(fill = "lightblue", color = "black", alpha = 0.7, 
               outlier.color = "black", outlier.shape = 1, outlier.size = 1.5) +
  stat_summary(fun = mean, geom = "point", shape = 18, 
               size = 3, color = "red") +
  stat_summary(fun = mean, geom = "line", aes(group = 1), 
               color = "red", linetype = "dashed", linewidth = 0.8) +
  labs(x = expression(c[0]), y = "Weight on Correct Model") +
  scale_x_discrete(labels = c0_labels) +
  theme_bw() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 16, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 16),
    axis.title.x = element_text(size = 24),
    axis.title.y = element_text(size = 16),
    panel.grid.major.x = element_blank()
  ) +
  ylim(0, 1)
print(p_box)

# Summary statistics for weights
results_summary <- data.frame(
  c0 = c0_values,
  mean_weight = colMeans(weight_correct, na.rm = TRUE),
  sd_weight = apply(weight_correct, 2, sd, na.rm = TRUE),
  median_weight = apply(weight_correct, 2, median, na.rm = TRUE)
)
print(results_summary)