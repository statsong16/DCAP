## Multi-omics dataset — basic statistics & figures
## Data: multi_omics_set.RData (microbiome / metabolome / proteome / metadata2)

suppressPackageStartupMessages({
  library(ggplot2)
})

# ---- Load ----
e <- new.env()
load("multi_omics_set.RData", envir = e)
mo <- e$multi_omics_set
md <- mo$metadata2

# Consistent enterotype colors
et_cols <- c("ET-Bacteroides" = "#E15759",
             "ET-Firmicutes"  = "#4E79A7",
             "ET-Prevotella"  = "#59A14F")

## =========================================================
## Basic statistics (printed to console)
## =========================================================
cat("\n================ COHORT SUMMARY ================\n")
cat("Samples:", nrow(md), " | Subjects:", length(unique(md$metadata.ID)), "\n")
cat("Timepoints (subtype):\n"); print(table(md$subtype))
cat("\nGender:\n"); print(table(md$Gender))
cat("\nEnterotype:\n"); print(table(md$enteroType))
cat("\nAge  :", sprintf("mean=%.1f sd=%.1f range=%.1f-%.1f",
    mean(md$Age), sd(md$Age), min(md$Age), max(md$Age)), "\n")
cat("BMI  :", sprintf("mean=%.1f sd=%.1f range=%.1f-%.1f",
    mean(md$BMI), sd(md$BMI), min(md$BMI), max(md$BMI)), "\n")

# t-test: BMI by Gender
tt <- t.test(BMI ~ Gender, data = md)
cat(sprintf("\nBMI by Gender  t-test: t=%.2f, df=%.1f, p=%.4f\n",
    tt$statistic, tt$parameter, tt$p.value))
# ANOVA: BMI across enterotypes
av <- summary(aov(BMI ~ enteroType, data = md))
cat("BMI across enterotypes (ANOVA):\n"); print(av)

## =========================================================
## FIGURE 1 — Cohort descriptive statistics (4 panels)
## =========================================================
p1a <- ggplot(md, aes(Age)) +
  geom_histogram(bins = 25, fill = "#4E79A7", color = "white") +
  labs(title = "A. Age distribution", x = "Age (years)", y = "Count") +
  theme_bw(base_size = 11)

p1b <- ggplot(md, aes(x = enteroType, y = BMI, fill = enteroType)) +
  geom_boxplot(outlier.size = 0.6, alpha = 0.85) +
  scale_fill_manual(values = et_cols, guide = "none") +
  labs(title = "B. BMI by enterotype", x = NULL, y = "BMI") +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

gdf <- as.data.frame(table(md$Gender)); names(gdf) <- c("Gender","n")
p1c <- ggplot(gdf, aes(x = Gender, y = n, fill = Gender)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = n), vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = c("Female" = "#F28E2B", "Male" = "#76B7B2"), guide = "none") +
  labs(title = "C. Sex composition", x = NULL, y = "Count") +
  theme_bw(base_size = 11)

edf <- as.data.frame(table(md$enteroType)); names(edf) <- c("enteroType","n")
p1d <- ggplot(edf, aes(x = reorder(enteroType, -n), y = n, fill = enteroType)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = n), vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = et_cols, guide = "none") +
  labs(title = "D. Enterotype composition", x = NULL, y = "Count") +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

# Simple grid layout without extra packages
png("figure1_cohort_stats.png", width = 2000, height = 1500, res = 200)
gridExtra_ok <- requireNamespace("gridExtra", quietly = TRUE)
if (gridExtra_ok) {
  gridExtra::grid.arrange(p1a, p1b, p1c, p1d, ncol = 2,
    top = grid::textGrob("Figure 1. Cohort descriptive statistics (n = 276)",
                         gp = grid::gpar(fontsize = 15, fontface = "bold")))
} else {
  vp <- function(r, c) grid::viewport(layout.pos.row = r, layout.pos.col = c)
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(2, 2)))
  print(p1a, vp = vp(1,1)); print(p1b, vp = vp(1,2))
  print(p1c, vp = vp(2,1)); print(p1d, vp = vp(2,2))
}
invisible(dev.off())
cat("\n[saved] figure1_cohort_stats.png\n")

## =========================================================
## FIGURE 2 — Proteome PCA colored by enterotype
## =========================================================
# proteome: features (rows) x samples (cols) -> transpose to samples x features
X <- t(as.matrix(mo$proteome))                 # 276 samples x 770 proteins
# mean-impute the ~1.7% missing values, drop zero-variance features
X <- apply(X, 2, function(col){ col[is.na(col)] <- mean(col, na.rm = TRUE); col })
X <- X[, apply(X, 2, sd) > 0, drop = FALSE]

pca <- prcomp(X, center = TRUE, scale. = TRUE)
ve  <- pca$sdev^2 / sum(pca$sdev^2)            # variance explained

# align metadata to proteome sample order
idx <- match(rownames(X), md$sample.ID)
pdf2 <- data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2],
                   enteroType = md$enteroType[idx],
                   subtype    = md$subtype[idx])

# PERMANOVA-free group check: variance of PC1/PC2 across enterotypes via Kruskal
kw1 <- kruskal.test(PC1 ~ factor(enteroType), data = pdf2)
cat(sprintf("\nPC1 differs across enterotypes (Kruskal): chi2=%.1f, p=%.3g\n",
    kw1$statistic, kw1$p.value))

p2 <- ggplot(pdf2, aes(PC1, PC2, color = enteroType)) +
  geom_point(size = 1.8, alpha = 0.85) +
  stat_ellipse(level = 0.68, linewidth = 0.7) +
  scale_color_manual(values = et_cols, name = "Enterotype") +
  labs(title = "Figure 2. Proteome PCA (770 proteins, 276 samples)",
       subtitle = "Points = samples; ellipses = 68% CI per enterotype",
       x = sprintf("PC1 (%.1f%% variance)", 100*ve[1]),
       y = sprintf("PC2 (%.1f%% variance)", 100*ve[2])) +
  theme_bw(base_size = 12) +
  theme(legend.position = "right",
        plot.title = element_text(face = "bold"))

ggsave("figure2_proteome_pca.png", p2, width = 8, height = 6, dpi = 200)
cat("[saved] figure2_proteome_pca.png\n")
cat(sprintf("PCA variance explained: PC1=%.1f%%, PC2=%.1f%%, PC3=%.1f%%\n",
    100*ve[1], 100*ve[2], 100*ve[3]))
