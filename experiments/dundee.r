library(cvms)
library(mgcv)
library(lmvar)
library(groupdata2)
library(dplyr)
library(gamclass)
library(lme4)
library(merTools)
library(stargazer)
library(stringr)

args <- commandArgs(trailingOnly = T)
BASE_DIR <- args[1]
set.seed(42)
control <- lmerControl(optCtrl = list(maxfun = 100000))
Unzip <- function(...) rbind(data.frame(), ...)
prefix <- function(x) {
    return(str_sub(x, 1, 2))
}

eye_data <- read.csv("data/DC/all.txt.annotation.filtered.csv", header = T, sep = "\t", quote = '"')
print(nrow(eye_data))
for (file in list.files(BASE_DIR, pattern = "scores.csv", recursive = T)) {
    data <- read.csv(paste(BASE_DIR, file, sep = ""), header = T, sep = "\t")
    print(nrow(data))
    data <- cbind(data, eye_data)
    data$screenN <- scale(data$screenN)
    data$lineN <- scale(data$lineN)
    data$segmentN <- scale(data$segmentN)
    data$length <- scale(data$length)
    data$gmean_freq <- scale(data$log_gmean_freq)
    data$length_prev_1 <- scale(data$length_prev_1)
    data$gmean_freq_prev_1 <- scale(data$log_gmean_freq_prev_1)
    data$log_gmean_freq <- scale(data$log_gmean_freq)
    data$log_gmean_freq_prev_1 <- scale(data$log_gmean_freq_prev_1)

    data$surprisals_sum_raw <- data$surprisals_sum
    data$surprisals_sum <- scale(data$surprisals_sum)
    data$surprisals_sum_prev_1 <- scale(data$surprisals_sum_prev_1)
    data$surprisals_sum_prev_2 <- scale(data$surprisals_sum_prev_2)
    data$surprisals_sum_prev_3 <- scale(data$surprisals_sum_prev_3)

    subdata <- subset(data, time > 0)
    subdata <- subset(subdata, has_num == "False")
    subdata <- subset(subdata, has_num_prev_1 == "False")
    subdata <- subset(subdata, has_punct == "False")
    subdata <- subset(subdata, has_punct_prev_1 == "False")
    subdata <- subset(subdata, is_first == "False")
    subdata <- subset(subdata, is_last == "False")
    subdata <- subset(subdata, pos != "CD")

    print(file)
    print("fit")
    base_mod_linear <- lmer(time ~ log_gmean_freq * length + log_gmean_freq_prev_1 * length_prev_1 + screenN + lineN + segmentN + (1 | article) + (1 | subj_id), data = subdata, REML = FALSE)
    base_linear_fit_logLik <- logLik(base_mod_linear) / nrow(subdata)
    lm_mod <- lmer(time ~ surprisals_sum + surprisals_sum_prev_1 + surprisals_sum_prev_2 + log_gmean_freq * length + log_gmean_freq_prev_1 * length_prev_1 + screenN + lineN + segmentN + (1 | article) + (1 | subj_id), data = subdata, REML = FALSE)
    sup_linear_fit_logLik <- logLik(lm_mod) / nrow(subdata)
    chi_p_linear <- anova(base_mod_linear, lm_mod)$Pr[2]



    out <- paste(BASE_DIR, paste(gsub("/[^/]+$", "", file), "/likelihood.txt", sep = ""), sep = "")
    out2 <- paste(BASE_DIR, paste(gsub("/[^/]+$", "", file), "/residuals.txt", sep = ""), sep = "")
    if (file.exists(out)) {
        file.remove(out)
    }
    if (file.exists(out2)) {
        file.remove(out2)
    }
    file.create(out, showWarnings = TRUE)
    write(paste("linear_fit_logLik: ", sup_linear_fit_logLik), file = out, append = T)
    write(paste("delta_linear_fit_logLik: ", sup_linear_fit_logLik - base_linear_fit_logLik), file = out, append = T)
    write(paste("delta_linear_fit_chi_p: ", chi_p_linear), file = out, append = T)

    residuals.frame <- do.call(Unzip, as.list(residuals(lm_mod)**2))
    colnames(residuals.frame) <- c("residual")
    residual_subdata <- cbind(residuals.frame, subdata)
    write.csv(residual_subdata, out2)
}