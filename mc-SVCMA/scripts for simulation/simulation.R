#################### Main Simulation ######################
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

source("functions.R")

set.seed(123)
nism <- 100

############################# Parameters ##################
n <- 1000
p1 <- 50
q <- 4
J <- 3
train_ratio <- 0.8
K <- 5

############################# Pre-screening ###################
Data <- Data1(n, p1, q)
X1 <- Data$X1
Y1 <- Data$Y1

model <- LiblineaR(data = X1, target = Y1, type = 4, cost = 1, bias = 1)
W <- model$W
importance <- apply(abs(W[, -ncol(W), drop = FALSE]), 2, sum)
sorted_idx <- order(importance, decreasing = TRUE)
importance_sorted <- importance[sorted_idx]
W_total <- sum(importance)
cum_importance <- cumsum(importance_sorted)

theta_scl <- 0.7
m_selected <- which(cum_importance >= theta_scl * W_total)[1]
selected_idx <- sorted_idx[1:m_selected]

X <- X1[, selected_idx, drop = FALSE]
p <- ncol(X)
Y <- as.factor(Y1)
S <- p

method_names <- c("SVMICL", "SVMICH", "SCL", "SCH", "EW", "OvR-SVCMA", "mc-SVCMA")
methods <- c("SVMICL", "SVMICH", "SCL", "SCH", "EW", "mc-SVCMA")

results_acc <- matrix(0, nrow = nism, ncol = 7)
results_macro_precision <- matrix(0, nrow = nism, ncol = 6)
results_macro_roc_auc <- matrix(0, nrow = nism, ncol = 6)
results_macro_pr_auc <- matrix(0, nrow = nism, ncol = 6)
results_log_loss <- matrix(0, nrow = nism, ncol = 6)
results_mse <- matrix(0, nrow = nism, ncol = 6)
results_mae <- matrix(0, nrow = nism, ncol = 6)
results_nhl <- matrix(0, nrow = nism, ncol = 6)
results_hinge <- matrix(0, nrow = nism, ncol = 6)

colnames(results_macro_precision) <- colnames(results_macro_roc_auc) <-
  colnames(results_macro_pr_auc) <- colnames(results_log_loss) <-
  colnames(results_mse) <- colnames(results_mae) <-
  colnames(results_nhl) <- colnames(results_hinge) <- methods

pb <- txtProgressBar(min = 0, max = nism, style = 3)

for (m in 1:nism) {
  train_idx <- caret::createDataPartition(Y, p = train_ratio, list = FALSE)
  X_train <- X[train_idx, ]
  y_train <- Y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- Y[-train_idx]
  n_test <- nrow(X_test)
  X_test <- cbind(1, X_test)
  C_val <- 1
  n1 <- length(y_train)
  
  vars <- 1:p
  feat_cols <- list()
  for (i in 1:p) {
    feat_cols[[i]] <- vars[1:i]
  }
  
  beta_matrix <- array(0, dim = c(J, p + 1, S))
  SVMICLs <- numeric(S)
  SVMICHs <- numeric(S)
  
  for (s in 1:S) {
    fcols <- feat_cols[[s]]
    X_train_s <- X_train[, fcols, drop = FALSE]
    svm_result_s <- svm_model(X_train_s, y_train, C_val)
    beta <- svm_result_s$beta
    beta_matrix[, 1, s] <- beta[, 1]
    beta_matrix[, feat_cols[[s]] + 1, s] <- beta[, -1]
    SVMIC_result <- SVMIC(y_train, svm_result_s, X_train_s)
    SVMICLs[s] <- SVMIC_result$SVMICLs
    SVMICHs[s] <- SVMIC_result$SVMICHs
  }
  
  cv_beta_matrix <- array(0, dim = c(J, p + 1, S, K))
  folds <- createFolds(y_train, k = K, list = TRUE, returnTrain = FALSE)
  cv_X_test <- list()
  cv_y_test <- list()
  
  for (k in 1:K) {
    A_k <- folds[[k]]
    B_k <- setdiff(1:n1, A_k)
    y_train_fold <- y_train[B_k]
    y_test_fold <- y_train[A_k]
    X_train_fold <- X_train[B_k, , drop = FALSE]
    X_test_fold <- cbind(1, X_train[A_k, ])
    cv_y_test[[k]] <- y_test_fold
    cv_X_test[[k]] <- X_test_fold
    
    for (s in 1:S) {
      X_train_s <- X_train_fold[, feat_cols[[s]], drop = FALSE]
      svm_result_fold <- svm_model(X_train_s, y_train_fold, 1)
      cv_beta <- svm_result_fold$beta
      cv_beta_matrix[, 1, s, k] <- cv_beta[, 1]
      cv_beta_matrix[, feat_cols[[s]] + 1, s, k] <- cv_beta[, -1]
    }
  }
  
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
  
  heq <- function(w) { sum(w) - 1 }
  
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
  
  best_SVMICLs <- which.min(SVMICLs)
  SVMICL_beta <- beta_matrix[, , best_SVMICLs]
  SVMICL_test_scores <- X_test %*% t(SVMICL_beta)
  SVMICL_y_pred <- apply(SVMICL_test_scores, 1, which.max)
  SVMICL_y_pred <- as.factor(SVMICL_y_pred)
  SVMICL_y_prob <- t(apply(SVMICL_test_scores, 1, softmax))
  colnames(SVMICL_y_prob) <- as.character(1:J)
  SVMICL_eval <- class_evaluate(y_test, SVMICL_y_pred, SVMICL_y_prob)
  
  best_SVMICHs <- which.min(SVMICHs)
  SVMICH_beta <- beta_matrix[, , best_SVMICHs]
  SVMICH_test_scores <- X_test %*% t(SVMICH_beta)
  SVMICH_y_pred <- apply(SVMICH_test_scores, 1, which.max)
  SVMICH_y_pred <- as.factor(SVMICH_y_pred)
  SVMICH_y_prob <- t(apply(SVMICH_test_scores, 1, softmax))
  colnames(SVMICH_y_prob) <- as.character(1:J)
  SVMICH_eval <- class_evaluate(y_test, SVMICH_y_pred, SVMICH_y_prob)
  
  hinge_svmicl <- multi_hinge_loss(y_test, SVMICL_test_scores)
  hinge_svmich <- multi_hinge_loss(y_test, SVMICH_test_scores)
  min_loss <- get_min_loss(beta_matrix, X_test, y_test)
  nhl_svmicl <- hinge_svmicl / min_loss
  nhl_svmich <- hinge_svmich / min_loss
  
  cv_eval <- weight_pred(beta_matrix, X_test, y_test, cv_weight)
  
  eq_weight <- rep(1 / S, S)
  eq_eval <- weight_pred(beta_matrix, X_test, y_test, eq_weight)
  
  SCL_weight <- (exp(-(SVMICLs - min(SVMICLs)) / 2)) / (sum(exp(-(SVMICLs - min(SVMICLs)) / 2)))
  SCL_eval <- weight_pred(beta_matrix, X_test, y_test, SCL_weight)
  
  SCH_weight <- (exp(-(SVMICHs - min(SVMICHs)) / 2)) / (sum(exp(-(SVMICHs - min(SVMICHs)) / 2)))
  SCH_eval <- weight_pred(beta_matrix, X_test, y_test, SCH_weight)
  
  wbeta_scl <- matrix(0, J, p + 1)
  for (s in 1:S) wbeta_scl <- wbeta_scl + SCL_weight[s] * beta_matrix[, , s]
  scores_scl <- X_test %*% t(wbeta_scl)
  hinge_scl <- multi_hinge_loss(y_test, scores_scl)
  nhl_scl <- hinge_scl / min_loss
  
  wbeta_sch <- matrix(0, J, p + 1)
  for (s in 1:S) wbeta_sch <- wbeta_sch + SCH_weight[s] * beta_matrix[, , s]
  scores_sch <- X_test %*% t(wbeta_sch)
  hinge_sch <- multi_hinge_loss(y_test, scores_sch)
  nhl_sch <- hinge_sch / min_loss
  
  wbeta_eq <- matrix(0, J, p + 1)
  for (s in 1:S) wbeta_eq <- wbeta_eq + (1 / S) * beta_matrix[, , s]
  scores_eq <- X_test %*% t(wbeta_eq)
  hinge_eq <- multi_hinge_loss(y_test, scores_eq)
  nhl_eq <- hinge_eq / min_loss
  
  wbeta_cv <- matrix(0, J, p + 1)
  for (s in 1:S) wbeta_cv <- wbeta_cv + cv_weight[s] * beta_matrix[, , s]
  scores_cv <- X_test %*% t(wbeta_cv)
  hinge_cv <- multi_hinge_loss(y_test, scores_cv)
  nhl_cv <- hinge_cv / min_loss
  
  ######################### OvR-SVMMA ############################
  classes <- levels(as.factor(y_train))
  bin_accuracy <- numeric(1)
  iter_results <- list(tp = numeric(J))
  
  for (j in 1:J) {
    y_binary <- ifelse(y_train == classes[j], 1, -1)
    y_binary <- as.factor(y_binary)
    y_test_binary <- ifelse(y_test == classes[j], 1, -1)
    
    X_train_bin <- X_train
    n1_bin <- length(y_binary)
    p_bin <- ncol(X_train_bin)
    M_bin <- p_bin
    
    feat_bin <- list()
    for (i in 1:p_bin) {
      feat_bin[[i]] <- 1:i
    }
    
    beta_bin <- array(0, dim = c(p_bin + 1, M_bin))
    
    for (m_idx in 1:M_bin) {
      bin_fcols <- feat_bin[[m_idx]]
      X_train_s_bin <- X_train_bin[, bin_fcols, drop = FALSE]
      
      clf_bin <- LiblineaR(data = X_train_s_bin, target = y_binary,
                           type = 3, cost = 1, bias = 1)
      
      W_bin <- clf_bin$W
      betab <- c(W_bin[, ncol(W_bin)], W_bin[, -ncol(W_bin)])
      
      beta_bin[1, m_idx] <- betab[1]
      beta_bin[bin_fcols + 1, m_idx] <- betab[-1]
    }
    
    folds_bin <- createFolds(y_binary, k = K, list = TRUE, returnTrain = FALSE)
    
    cv_beta_bin <- array(0, dim = c(p_bin + 1, M_bin, K))
    cv_X_bin <- list()
    cv_y_bin <- list()
    
    for (k in 1:K) {
      val_idx <- folds_bin[[k]]
      train_idx_fold <- setdiff(1:n1_bin, val_idx)
      
      y_train_fold_bin <- y_binary[train_idx_fold]
      y_test_fold_bin <- y_binary[val_idx]
      
      X_train_fold_bin <- X_train_bin[train_idx_fold, , drop = FALSE]
      X_test_fold_bin <- cbind(1, X_train_bin[val_idx, ])
      
      cv_y_bin[[k]] <- y_test_fold_bin
      cv_X_bin[[k]] <- X_test_fold_bin
      
      for (m_idx in 1:M_bin) {
        bin_fcols <- feat_bin[[m_idx]]
        X_train_s_bin <- X_train_fold_bin[, bin_fcols, drop = FALSE]
        
        cv_clf <- LiblineaR(data = X_train_s_bin, target = y_train_fold_bin,
                            type = 3, cost = 1, bias = 1)
        
        cv_W_bin <- cv_clf$W
        cv_betab <- c(cv_W_bin[, ncol(cv_W_bin)], cv_W_bin[, -ncol(cv_W_bin)])
        
        cv_beta_bin[1, m_idx, k] <- cv_betab[1]
        cv_beta_bin[bin_fcols + 1, m_idx, k] <- cv_betab[-1]
      }
    }
    
    w_obj_bin <- function(w_b) {
      cv_loss_bin <- 0
      for (k in 1:K) {
        beta_kb <- cv_beta_bin[, , k]
        wbeta_kb <- matrix(0, 1, p_bin + 1)
        for (m_idx in 1:M_bin) {
          wbeta_kb <- wbeta_kb + w_b[m_idx] * beta_kb[, m_idx]
        }
        X_test_s_bin <- cv_X_bin[[k]]
        y_test_fold_bin <- cv_y_bin[[k]]
        
        cv_scores_bin <- X_test_s_bin %*% t(wbeta_kb)
        y_num <- as.numeric(as.character(y_test_fold_bin))
        cv_loss_bin <- cv_loss_bin + sum(pmax(0, 1 - y_num * cv_scores_bin[, 1]))
      }
      return(cv_loss_bin / n1_bin)
    }
    
    heq_bin <- function(w_b) sum(w_b) - 1
    p0 <- rep(1 / M_bin, M_bin)
    
    res_bin <- nloptr(
      x0 = p0,
      eval_f = w_obj_bin,
      lb = rep(0, M_bin),
      ub = rep(1, M_bin),
      eval_g_eq = heq_bin,
      opts = list(
        algorithm = "NLOPT_LN_COBYLA",
        xtol_rel = 1e-7,
        maxeval = 3000,
        print_level = 0
      )
    )
    
    w_bin <- res_bin$solution
    
    wbeta_bin <- matrix(0, 1, p_bin + 1)
    for (m_idx in 1:M_bin) {
      wbeta_bin <- wbeta_bin + w_bin[m_idx] * beta_bin[, m_idx]
    }
    
    test_scores_bin <- X_test %*% t(wbeta_bin)
    pred_bin <- ifelse(test_scores_bin[, 1] >= 0, 1, -1)
    
    y_true <- as.numeric(as.character(y_test_binary))
    tp <- sum(y_true == 1 & pred_bin == 1)
    iter_results$tp[j] <- tp
  }
  bin_accuracy <- sum(iter_results$tp / n_test)
  
  results_acc[m, ] <- c(SVMICL_eval$accuracy, SVMICH_eval$accuracy,
                        SCL_eval$accuracy, SCH_eval$accuracy,
                        eq_eval$accuracy, bin_accuracy, cv_eval$accuracy)
  
  results_macro_precision[m, ] <- c(SVMICL_eval$macro_precision, SVMICH_eval$macro_precision,
                                    SCL_eval$macro_precision, SCH_eval$macro_precision,
                                    eq_eval$macro_precision, cv_eval$macro_precision)
  
  results_macro_roc_auc[m, ] <- c(SVMICL_eval$macro_roc_auc, SVMICH_eval$macro_roc_auc,
                                  SCL_eval$macro_roc_auc, SCH_eval$macro_roc_auc,
                                  eq_eval$macro_roc_auc, cv_eval$macro_roc_auc)
  
  results_macro_pr_auc[m, ] <- c(SVMICL_eval$macro_pr_auc, SVMICH_eval$macro_pr_auc,
                                 SCL_eval$macro_pr_auc, SCH_eval$macro_pr_auc,
                                 eq_eval$macro_pr_auc, cv_eval$macro_pr_auc)
  
  results_log_loss[m, ] <- c(SVMICL_eval$log_loss, SVMICH_eval$log_loss,
                             SCL_eval$log_loss, SCH_eval$log_loss,
                             eq_eval$log_loss, cv_eval$log_loss)
  
  results_mse[m, ] <- c(SVMICL_eval$mse, SVMICH_eval$mse,
                        SCL_eval$mse, SCH_eval$mse,
                        eq_eval$mse, cv_eval$mse)
  
  results_mae[m, ] <- c(SVMICL_eval$mae, SVMICH_eval$mae,
                        SCL_eval$mae, SCH_eval$mae,
                        eq_eval$mae, cv_eval$mae)
  
  results_nhl[m, ] <- c(nhl_svmicl, nhl_svmich, nhl_scl, nhl_sch, nhl_eq, nhl_cv)
  results_hinge[m, ] <- c(hinge_svmicl, hinge_svmich, hinge_scl, hinge_sch, hinge_eq, hinge_cv)
  
  setTxtProgressBar(pb, m)
}

close(pb)

########################### Results Summary ###########################
acc_mean <- colMeans(results_acc)
acc_sd <- apply(results_acc, 2, sd)
macro_precision_mean <- colMeans(results_macro_precision)
macro_precision_sd <- apply(results_macro_precision, 2, sd)
macro_roc_auc_mean <- colMeans(results_macro_roc_auc)
macro_roc_auc_sd <- apply(results_macro_roc_auc, 2, sd)
macro_pr_auc_mean <- colMeans(results_macro_pr_auc)
macro_pr_auc_sd <- apply(results_macro_pr_auc, 2, sd)
log_loss_mean <- colMeans(results_log_loss)
log_loss_sd <- apply(results_log_loss, 2, sd)
mse_mean <- colMeans(results_mse)
mse_sd <- apply(results_mse, 2, sd)
mae_mean <- colMeans(results_mae)
mae_sd <- apply(results_mae, 2, sd)
nhl_mean <- colMeans(results_nhl)
nhl_sd <- apply(results_nhl, 2, sd)
hinge_mean <- colMeans(results_hinge)
hinge_sd <- apply(results_hinge, 2, sd)

metric_names <- c("Macro-Precision", "Macro-AUC", "Macro-PR-AUC",
                  "Log Loss", "MSE", "MAE", "NHL", "Hinge Loss")

mean_list <- list(macro_precision_mean, macro_roc_auc_mean, macro_pr_auc_mean,
                  log_loss_mean, mse_mean, mae_mean, nhl_mean, hinge_mean)

sd_list <- list(macro_precision_sd, macro_roc_auc_sd, macro_pr_auc_sd,
                log_loss_sd, mse_sd, mae_sd, nhl_sd, hinge_sd)

results_matrix <- matrix("", nrow = length(metric_names) * 2, ncol = 7)
colnames(results_matrix) <- c("Metric", methods)

for (i in 1:length(metric_names)) {
  row_mean <- (i - 1) * 2 + 1
  row_sd <- (i - 1) * 2 + 2
  
  results_matrix[row_mean, 1] <- metric_names[i]
  results_matrix[row_sd, 1] <- ""
  
  vals_mean <- mean_list[[i]]
  vals_sd <- sd_list[[i]]
  
  is_loss <- grepl("Loss|MSE|MAE|NHL|Hinge", metric_names[i])
  
  if (is_loss) {
    best_idx <- which.min(vals_mean)
  } else {
    best_idx <- which.max(vals_mean)
  }
  
  for (j in 1:6) {
    mean_str <- sprintf("%.4f", vals_mean[j])
    sd_str <- sprintf("(%.4f)", vals_sd[j])
    
    if (j == best_idx) {
      mean_str <- paste0(mean_str, "†")
    }
    
    results_matrix[row_mean, j + 1] <- mean_str
    results_matrix[row_sd, j + 1] <- sd_str
  }
}

print(results_matrix, quote = FALSE, na.print = "")