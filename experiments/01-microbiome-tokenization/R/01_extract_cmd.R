#!/usr/bin/env Rscript
# =====================================================================
# 01_extract_cmd.R
# curatedMetagenomicData -> CSV (Python 쪽으로 넘길 중간 산출물)
#
# 내보내는 파일 4개:
#   data/taxa_abundance.csv     행=샘플, 열=종(species). 상대풍부도(합=100)
#   data/pathway_abundance.csv  행=샘플, 열=경로(pathway). 두 번째 모달리티
#   data/metadata.csv           행=샘플. study_name, disease, age, gender 등
#   data/taxa_lineage.csv       종별 계통 정보(k/p/c/o/f/g/s) — 그래프 마스크용
#
# 주의: taxa와 pathway는 같은 샘플에서 나온 짝 데이터여야 합니다.
#       교집합 샘플만 남깁니다. (구성 C 모달리티 간 정렬의 전제)
# =====================================================================

suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(dplyr)
  library(tibble)
  library(SummarizedExperiment)
  library(TreeSummarizedExperiment)
})

OUT_DIR <- "data"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------
# 1. 대상 연구 선택
#    - 처음에는 2~4개 연구로 시작하세요. 전체를 받으면 수 GB입니다.
#    - study_name 이 여러 개여야 downstream 에서 study 단위 교차검증이 됩니다.
#      (같은 study 안에서 나눈 CV 는 배치효과 때문에 성능이 부풀려집니다)
# ---------------------------------------------------------------------
STUDIES <- c(
  "ZellerG_2014",      # 대장암 vs 대조
  "FengQ_2015",        # 대장암 vs 대조
  "YuJ_2015"           # 대장암 vs 대조
)
# 치매/노화 쪽으로 바꾸려면 sampleMetadata 에서 condition 을 훑어보세요:
#   sampleMetadata |> count(study_condition, sort = TRUE) |> print(n = 60)

pattern_taxa <- paste0("(", paste(STUDIES, collapse = "|"), ").*relative_abundance")
pattern_path <- paste0("(", paste(STUDIES, collapse = "|"), ").*pathway_abundance")

message("[1/5] taxonomic profile 내려받는 중 ...")
tse_taxa <- curatedMetagenomicData(pattern_taxa, dryrun = FALSE, counts = FALSE, rownames = "short")
tse_taxa <- mergeData(tse_taxa)

message("[2/5] pathway profile 내려받는 중 ...")
tse_path <- curatedMetagenomicData(pattern_path, dryrun = FALSE, counts = FALSE, rownames = "short")
tse_path <- mergeData(tse_path)

# ---------------------------------------------------------------------
# 2. 짝 샘플만 남기기
# ---------------------------------------------------------------------
common <- intersect(colnames(tse_taxa), colnames(tse_path))
message(sprintf("[3/5] 짝 샘플 %d개 (taxa %d, pathway %d)",
                length(common), ncol(tse_taxa), ncol(tse_path)))
stopifnot(length(common) > 50)

tse_taxa <- tse_taxa[, common]
tse_path <- tse_path[, common]

# ---------------------------------------------------------------------
# 3. 저빈도 feature 제거
#    - prevalence 필터만 겁니다. 풍부도 필터는 Python 쪽에서 처리합니다.
#    - 여기서 너무 세게 자르면 조성 구조가 바뀌므로 보수적으로.
# ---------------------------------------------------------------------
prevalence_filter <- function(mat, min_prev = 0.10) {
  keep <- rowMeans(mat > 0) >= min_prev
  mat[keep, , drop = FALSE]
}

taxa_mat <- prevalence_filter(assay(tse_taxa), min_prev = 0.10)
path_mat <- prevalence_filter(assay(tse_path), min_prev = 0.10)
message(sprintf("[4/5] 필터 후  taxa %d개, pathway %d개", nrow(taxa_mat), nrow(path_mat)))

# ---------------------------------------------------------------------
# 4. 계통 정보 (그래프 마스크용)
#    rowData 에 계통 열이 있으면 그걸 쓰고, 없으면 rowname 을 파싱합니다.
# ---------------------------------------------------------------------
lineage_from_rowdata <- function(tse, taxa_names) {
  rd <- as.data.frame(rowData(tse))
  ranks <- intersect(c("kingdom", "phylum", "class", "order", "family", "genus", "species"),
                     tolower(colnames(rd)))
  if (length(ranks) >= 4) {
    colnames(rd) <- tolower(colnames(rd))
    out <- rd[taxa_names, ranks, drop = FALSE]
    out$taxon <- taxa_names
    return(out)
  }
  NULL
}

lin <- lineage_from_rowdata(tse_taxa, rownames(taxa_mat))

if (is.null(lin)) {
  # fallback: "k__Bacteria|p__Firmicutes|...|s__Xxx" 형태 파싱
  parse_lineage <- function(x) {
    parts <- strsplit(x, "\\|")[[1]]
    get_rank <- function(pre) {
      hit <- grep(paste0("^", pre, "__"), parts, value = TRUE)
      if (length(hit) == 0) NA_character_ else sub(paste0("^", pre, "__"), "", hit[1])
    }
    c(kingdom = get_rank("k"), phylum = get_rank("p"), class = get_rank("c"),
      order = get_rank("o"), family = get_rank("f"), genus = get_rank("g"),
      species = get_rank("s"))
  }
  lin <- as.data.frame(t(vapply(rownames(taxa_mat), parse_lineage, character(7))),
                       stringsAsFactors = FALSE)
  lin$taxon <- rownames(taxa_mat)
}

# ---------------------------------------------------------------------
# 5. 저장 (행=샘플이 되도록 전치)
# ---------------------------------------------------------------------
meta <- as.data.frame(colData(tse_taxa)) |>
  rownames_to_column("sample_id") |>
  select(any_of(c("sample_id", "study_name", "subject_id", "study_condition",
                  "disease", "age", "gender", "BMI", "country",
                  "number_reads", "days_from_first_collection")))

write.csv(as.data.frame(t(taxa_mat)) |> rownames_to_column("sample_id"),
          file.path(OUT_DIR, "taxa_abundance.csv"), row.names = FALSE)
write.csv(as.data.frame(t(path_mat)) |> rownames_to_column("sample_id"),
          file.path(OUT_DIR, "pathway_abundance.csv"), row.names = FALSE)
write.csv(meta, file.path(OUT_DIR, "metadata.csv"), row.names = FALSE)
write.csv(lin,  file.path(OUT_DIR, "taxa_lineage.csv"), row.names = FALSE)

message("[5/5] 완료. data/ 에 CSV 4개 생성")
