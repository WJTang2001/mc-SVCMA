################################################################################
# functions.R
# Description: Core functions for multi-class SVM model averaging
################################################################################

# ==============================================================================
# multi_hinge_loss
# Description: Computes the average multi-class hinge loss
# Input:
#   y_true  - vector of true class labels (factor or numeric)
#   scores  - matrix of decision scores, n x J, where J is number of classes
# Output:
#   average hinge loss (scalar)
# ==============================================================================
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
      margin <- max(0, 1 - true_score + scores[i, c])
      if (margin > max_margin) max_margin <- margin
    }
    loss <- loss + max_margin
  }
  return(loss / n)
}

# ==============================================================================
# svm_model
# Description: Trains a multi-class SVM model using LiblineaR.
# Input:
#   X_train - training feature matrix (n x p)
#   Y_train - training class labels (factor)
#   C_val   - regularization parameter (cost)
# Output:
#   list containing:
#     - model: LiblineaR model object
#     - beta: coefficient matrix (J x (p+1)), intercept in first column
#     - scores: decision scores on training data
#     - loss: training hinge loss
#     - class_order: ordered class labels
# ==============================================================================
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

# ==============================================================================
# get_min_loss
# Description: Computes the minimum attainable loss on test data.
# Input:
#   beta_matrix - coefficient array (J x (p+1) x S), S = number of models
#   X_test      - test feature matrix with intercept column (n x (p+1))
#   y_test      - test class labels
# Output:
#   minimum hinge loss (scalar)
# ==============================================================================
get_min_loss <- function(beta_matrix, X_test, y_test) {
  J <- nrow(beta_matrix)
  S <- dim(beta_matrix)[3]
  
  w_min_fun <- function(w) {
    wbeta <- matrix(0, J, ncol(beta_matrix))
    for (s in 1:S) wbeta <- wbeta + w[s] * beta_matrix[, , s]
    scores <- X_test %*% t(wbeta)
    return(multi_hinge_loss(y_test, scores))
  }
  
  heq <- function(w) sum(w) - 1
  p0 <- rep(1 / S, S)
  
  res <- nloptr(x0 = p0, eval_f = w_min_fun,
                lb = rep(0, S), ub = rep(1, S),
                eval_g_eq = heq,
                opts = list(algorithm = "NLOPT_LN_COBYLA",
                            xtol_rel = 1e-7, maxeval = 3000, print_level = 0))
  return(w_min_fun(res$solution))
}

# ==============================================================================
# class_evaluate
# Description: Computes comprehensive multi-class classification evaluation
#              metrics including accuracy, macro/micro/weighted averages,
#              ROC-AUC, PR-AUC, and probability-based losses.
# Input:
#   y_true - true class labels (factor)
#   y_pred - predicted class labels (factor)
#   y_prob - predicted class probabilities (n x J matrix)
# Output:
#   list containing all evaluation metrics
# ==============================================================================
class_evaluate <- function(y_true, y_pred, y_prob) {
  y_true <- factor(y_true, levels = c("1", "2", "3"))
  y_pred <- factor(y_pred, levels = c("1", "2", "3"))
  c <- levels(y_true)
  J <- length(c)
  n <- length(y_true)
  
  TP <- numeric(J)
  FP <- numeric(J)
  FN <- numeric(J)
  TN <- numeric(J)
  precision <- numeric(J)
  recall <- numeric(J)
  F1 <- numeric(J)
  roc_auc_j <- numeric(J)
  pr_auc_j <- numeric(J)
  names(TP) <- names(FP) <- names(FN) <- names(TN) <- c
  names(precision) <- names(recall) <- names(F1) <- names(roc_auc_j) <- names(pr_auc_j) <- c
  
  cm <- table(y_pred, y_true)
  for (j in 1:J) {
    idx <- which(rownames(cm) == c[j])
    TP[j] <- cm[idx, idx]
    FP[j] <- sum(cm[, idx]) - TP[j]
    FN[j] <- sum(cm[idx, ]) - TP[j]
    TN[j] <- sum(cm) - TP[j] - FP[j] - FN[j]
    
    precision[j] <- TP[j] / (TP[j] + FP[j])
    recall[j] <- TP[j] / (TP[j] + FN[j])
    F1[j] <- 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    
    y_bin <- as.numeric(y_true == c[j])
    prob_bin <- y_prob[, c[j]]
    roc_auc_j[j] <- roc(y_bin, prob_bin, quiet = TRUE)$auc
    pr_obj <- pr.curve(scores.class0 = prob_bin[y_bin == 1],
                       scores.class1 = prob_bin[y_bin == 0],
                       curve = FALSE)
    pr_auc_j[j] <- pr_obj$auc.integral
  }
  
  accuracy <- sum(TP) / n
  
  micro_precision <- sum(TP) / sum(TP + FP)
  micro_recall <- sum(TP) / sum(TP + FN)
  micro_F1 <- (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
  
  macro_precision <- mean(precision)
  macro_recall <- mean(recall)
  macro_F1 <- mean(F1)
  macro_roc_auc <- mean(roc_auc_j)
  macro_pr_auc <- mean(pr_auc_j)
  
  n_j <- table(y_true)
  weighted_precision <- sum(n_j / n * precision)
  weighted_recall <- sum(n_j / n * recall)
  weighted_F1 <- sum(n_j / n * F1)
  weighted_roc_auc <- sum(n_j / n * roc_auc_j)
  weighted_pr_auc <- sum(n_j / n * pr_auc_j)
  
  y_ic <- matrix(0, nrow = n, ncol = J)
  for (i in 1:n) {
    y_ic[i, which(c == y_true[i])] <- 1
  }
  log_loss <- -mean(rowSums(y_ic * log(y_prob)))
  mse <- mean((y_ic - y_prob)^2)
  mae <- mean(abs(y_ic - y_prob))
  
  return(list(
    n = n,
    J = J,
    c = c,
    confusion_matrix = cm,
    TP = TP, FP = FP, FN = FN, TN = TN,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    F1 = F1,
    roc_auc = roc_auc_j,
    pr_auc = pr_auc_j,
    micro_precision = micro_precision,
    micro_recall = micro_recall,
    micro_F1 = micro_F1,
    macro_precision = macro_precision,
    macro_recall = macro_recall,
    macro_F1 = macro_F1,
    macro_roc_auc = macro_roc_auc,
    macro_pr_auc = macro_pr_auc,
    weighted_precision = weighted_precision,
    weighted_recall = weighted_recall,
    weighted_F1 = weighted_F1,
    weighted_roc_auc = weighted_roc_auc,
    weighted_pr_auc = weighted_pr_auc,
    log_loss = log_loss,
    mse = mse,
    mae = mae
  ))
}

# ==============================================================================
# SVMIC
# Description: Computes model selection criteria:
#              - SVMICL: hinge_loss * n + J * p * log(n)
#              - SVMICH: hinge_loss * n + J * p * (log(n))^(3/2)
# Input:
#   y_train          - training class labels
#   svm_result       - output from svm_model() for a single model
#   X_train_scaled   - training feature matrix for that model
# Output:
#   list containing SVMICL and SVMICH values
# ==============================================================================
SVMIC <- function(y_train, svm_result, X_train_scaled) {
  n <- length(y_train)
  J <- length(levels(y_train))
  hinge_loss <- svm_result$loss
  p_s <- ncol(X_train_scaled)
  SVMICLs <- hinge_loss * n + J * p_s * log(n)
  SVMICHs <- hinge_loss * n + J * p_s * ((log(n))^(3/2))
  return(list(SVMICLs = SVMICLs, SVMICHs = SVMICHs))
}

# ==============================================================================
# softmax
# Description: Softmax transformation for converting decision scores to
#              class probabilities.
# Input:
#   x - vector of decision scores
# Output:
#   vector of probabilities summing to 1
# ==============================================================================
softmax <- function(x) {
  exp(x - max(x)) / sum(exp(x - max(x)))
}

# ==============================================================================
# weight_pred
# Description: Makes predictions using weighted combination of candidate models.
#              Computes weighted coefficients, decision scores, and evaluates.
# Input:
#   beta_matrix    - coefficient array (J x (p+1) x S)
#   X_test_total   - test feature matrix with intercept (n x (p+1))
#   y_test         - test class labels
#   weights        - vector of model weights (length S)
# Output:
#   evaluation results from class_evaluate()
# ==============================================================================
weight_pred <- function(beta_matrix, X_test_total, y_test, weights) {
  wbeta <- matrix(0, nrow(beta_matrix), ncol(beta_matrix))
  for (s in 1:length(weights)) {
    wbeta <- wbeta + weights[s] * beta_matrix[, , s]
  }
  test_scores <- X_test_total %*% t(wbeta)
  
  y_pred <- apply(test_scores, 1, which.max)
  y_pred <- as.factor(y_pred)
  
  y_prob <- t(apply(test_scores, 1, softmax))
  colnames(y_prob) <- as.character(1:(nrow(beta_matrix)))
  eval_result <- class_evaluate(y_test, y_pred, y_prob)
  return(eval_result)
}

# ==============================================================================
# Data1
# Description: Generates synthetic three-class data with Gaussian distributions.
#              Classes have means at mu1, mu2, mu3 with common covariance.
# Input:
#   n - number of samples
#   p - number of features
#   q - number of informative features (first q features have signal)
# Output:
#   list containing:
#     - X1: feature matrix (n x p)
#     - Y1: class labels
# ==============================================================================
Data1 <- function(n, p, q) {
  Y1 <- sample(c(1, 2, 3), n, replace = TRUE, prob = c(1/3, 1/3, 1/3))
  Sigma <- matrix(0.8, nrow = p, ncol = p)
  diag(Sigma) <- 1
  mu1 <- c(rep(0.6, q), rep(0, p - q))
  mu2 <- c(rep(0, q), rep(0, p - q))
  mu3 <- c(rep(-0.6, q), rep(0, p - q))
  X1 <- matrix(0, n, p)
  idx1 <- which(Y1 == 1)
  idx2 <- which(Y1 == 2)
  idx3 <- which(Y1 == 3)
  X1[idx1, ] <- mvtnorm::rmvnorm(length(idx1), mean = mu1, sigma = Sigma)
  X1[idx2, ] <- mvtnorm::rmvnorm(length(idx2), mean = mu2, sigma = Sigma)
  X1[idx3, ] <- mvtnorm::rmvnorm(length(idx3), mean = mu3, sigma = Sigma)
  return(list(X1 = X1, Y1 = Y1))
}