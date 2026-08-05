## descriptive.R — 코호트 기술 통계와 그림 (R 판)
##
## scripts/run_describe.py 와 같은 내용을 R 로 합니다. 둘 중 편한 쪽을 쓰세요.
## 파이프라인 본체는 파이썬이고, 이 스크립트는 자료를 눈으로 보는 단계 전용입니다.
##
## 입력: RData 하나, 또는 export_rdata.py 가 만든 CSV 들.
## 출력: runs/describe_R/ (git-ignore 됨)
##
## 주의: 여기 나오는 수치는 전체 자료에 대한 기술 통계입니다. 모델 성능이
##       아니고, 이 그림의 군집 분리를 성능으로 읽으면 안 됩니다.

suppressPackageStartupMessages({
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
rdata_path <- if (length(args) >= 1) args[1] else "multi_omics_set.RData"
out_dir    <- if (length(args) >= 2) args[2] else "runs/describe_R"
obj_name   <- if (length(args) >= 3) args[3] else "multi_omics_set"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

e <- new.env()
load(rdata_path, envir = e)
mo <- get(obj_name, envir = e)
md <- mo$metadata2

et_cols <- c("ET-Bacteroides" = "#E15759",
             "ET-Firmicutes"  = "#4E79A7",
             "ET-Prevotella"  = "#59A14F")

## ---------------------------------------------------------------
## 1. 상수 변수와 반복측정 구조
## ---------------------------------------------------------------
cat("\n=============== 코호트 요약 ===============\n")
cat("표본:", nrow(md), " 피험자:", length(unique(md$metadata.ID)), "\n\n")

n_uniq <- sapply(md, function(x) length(unique(x)))
const  <- names(n_uniq)[n_uniq == 1]
if (length(const)) {
  cat("[주의] 상수 변수:", paste(const, collapse = ", "), "\n")
  cat("       표적으로도 공변량으로도 쓸 수 없습니다.\n\n")
}

## 피험자 내 변동 — 0에 가까우면 사실상 피험자 상수입니다.
## 그런 표적에 표본 단위 분할을 쓰면 성능은 재식별 정확도가 됩니다.
for (v in c("Age", "BMI")) {
  if (v %in% names(md)) {
    w <- tapply(md[[v]], md$metadata.ID, sd)
    cat(sprintf("%-4s 피험자 내 SD: 중앙 %.3f (최대 %.3f)\n",
                v, median(w, na.rm = TRUE), max(w, na.rm = TRUE)))
  }
}
cat("\n분포:\n"); print(table(md$subtype))
print(table(md$Gender)); print(table(md$enteroType))

## ---------------------------------------------------------------
## 2. 블록별 결측
## ---------------------------------------------------------------
cat("\n=============== 블록별 결측 ===============\n")
for (b in c("microbiome", "metabolome", "proteome")) {
  if (is.null(mo[[b]])) next
  X <- as.matrix(mo[[b]])                     # 행=feature, 열=표본
  na_by_feature <- rowMeans(is.na(X))
  mu <- rowMeans(X, na.rm = TRUE)
  ok <- is.finite(mu) & is.finite(na_by_feature)
  r  <- if (sum(ok) > 2 && sd(na_by_feature[ok]) > 0)
          cor(na_by_feature[ok], mu[ok]) else NA_real_
  cat(sprintf("%-11s %5d x %4d  결측 %.4f  결측률~평균 상관 %+.3f  %s\n",
              b, nrow(X), ncol(X), mean(is.na(X)), r,
              if (!is.na(r) && r < -0.15) "-> 검출한계 결측으로 보임" else ""))
  ## 피험자 단위로 몰려 있는지 — 몰려 있으면 무작위 결측이 아닙니다.
  na_by_sample <- colMeans(is.na(X))
  worst <- sort(tapply(na_by_sample, md$metadata.ID, mean), decreasing = TRUE)
  worst <- worst[worst > 0]
  if (length(worst)) {
    cat("   결측 상위 피험자: ",
        paste(sprintf("%s(%.3f)", names(worst)[1:min(3, length(worst))],
                      worst[1:min(3, length(worst))]), collapse = ", "), "\n")
  }
}

## ---------------------------------------------------------------
## 3. 그림
## ---------------------------------------------------------------
p1 <- ggplot(md, aes(Age)) +
  geom_histogram(bins = 25, fill = "#4E79A7", color = "white") +
  labs(title = "A. Age distribution", x = "Age", y = "Count") +
  theme_bw(base_size = 11)

p2 <- ggplot(md, aes(x = enteroType, y = BMI, fill = enteroType)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
  scale_fill_manual(values = et_cols, guide = "none") +
  labs(title = "B. BMI by enterotype", x = NULL, y = "BMI") +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

ggsave(file.path(out_dir, "age.png"), p1, width = 5, height = 4, dpi = 150)
ggsave(file.path(out_dir, "bmi_by_enterotype.png"), p2, width = 5, height = 4, dpi = 150)

## proteome PCA — 기술 통계용. 전체 자료에 적합합니다.
if (!is.null(mo$proteome)) {
  X <- t(as.matrix(mo$proteome))
  X <- apply(X, 2, function(col) { col[is.na(col)] <- median(col, na.rm = TRUE); col })
  X <- X[, apply(X, 2, sd) > 0, drop = FALSE]
  pca <- prcomp(X, center = TRUE, scale. = TRUE)
  ve <- pca$sdev^2 / sum(pca$sdev^2)
  idx <- match(rownames(X), md$sample.ID)
  pdat <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2],
                     enteroType = md$enteroType[idx])
  p3 <- ggplot(pdat, aes(PC1, PC2, color = enteroType)) +
    geom_point(size = 1.8, alpha = 0.85) +
    scale_color_manual(values = et_cols, name = "Enterotype") +
    labs(title = "proteome PCA",
         x = sprintf("PC1 (%.1f%% variance)", 100 * ve[1]),
         y = sprintf("PC2 (%.1f%% variance)", 100 * ve[2])) +
    theme_bw(base_size = 12)
  ggsave(file.path(out_dir, "pca_proteome.png"), p3, width = 7, height = 5, dpi = 150)
  cat(sprintf("\nproteome PCA: PC1 %.1f%%, PC2 %.1f%%\n", 100 * ve[1], 100 * ve[2]))
  cat("주의: 위 PCA 는 전체 자료를 중앙값 대치 후 적합한 것입니다.\n")
  cat("      결측이 검출한계에서 왔다면 중앙값 대치는 편향을 만듭니다.\n")
  cat("      모델링에는 파이썬 쪽 preprocess.BlockPrep 을 쓰세요.\n")
}

cat(sprintf("\n[saved] %s\n", out_dir))
