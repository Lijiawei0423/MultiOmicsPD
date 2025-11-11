library(AnnotationDbi)
library(org.Hs.eg.db)
library(clusterProfiler)
library(ReactomePA)
library(dplyr)
library(openxlsx)

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

BM_group <- 1  # Specify BM group (1, 2, 3, 4)
protlist <- read.csv("import protein list", stringsAsFactors = FALSE)
bm_data <- protlist[protlist$BM == BM_group, ]

res_folder <- paste0("result folder", timestamp)
if (!dir.exists(res_folder)) {
  dir.create(res_folder)
}

res <- file.path(res_folder, paste0("BM", BM_group, ".xlsx"))
protein_names <- bm_data$Omics_feature_x
protein_names <- na.omit(protein_names)
protein_names <- protein_names[protein_names != ""]

# Convert protein/gene symbols to ENTREZID
gene.df <- bitr(protein_names,
                fromType = "SYMBOL",
                toType = c("ENTREZID", "ENSEMBL"),
                OrgDb = org.Hs.eg.db)

cat("Number of successfully converted genes:", nrow(gene.df), "\n")
if (nrow(gene.df) == 0) {
  stop("No genes were successfully converted. Please check the input symbols.")
}

ls <- unique(gene.df$ENTREZID)

# Functional enrichment analysis
GO_all <- enrichGO(gene = ls,
                   OrgDb = org.Hs.eg.db,
                   keyType = "ENTREZID",
                   ont = "ALL",
                   pAdjustMethod = "BH",
                   minGSSize = 1,
                   pvalueCutoff = 0.05,
                   qvalueCutoff = 0.05,
                   readable = TRUE)

GO_BP <- enrichGO(gene = ls,
                  OrgDb = org.Hs.eg.db,
                  keyType = "ENTREZID",
                  ont = "BP",
                  pAdjustMethod = "BH",
                  minGSSize = 1,
                  pvalueCutoff = 0.05,
                  qvalueCutoff = 0.05,
                  readable = TRUE)

KEGG <- enrichKEGG(gene = ls,
                   keyType = 'kegg',
                   organism = "hsa",
                   pAdjustMethod = "BH", 
                   qvalueCutoff = 0.05, 
                   pvalueCutoff = 0.05)

REACTOME <- enrichPathway(gene = ls,
                          pvalueCutoff = 0.05, 
                          organism = "human",
                          pAdjustMethod = "BH",
                          qvalueCutoff = 0.05,
                          readable = TRUE)

# Save all results to Excel
wb <- createWorkbook()
enrichment_results <- list(
  GO_all = GO_all,
  GO_BP = GO_BP,
  KEGG = KEGG,
  REACTOME = REACTOME
)
for (name in names(enrichment_results)) {
  df <- as.data.frame(enrichment_results[[name]])
  if (nrow(df) > 0) {
    addWorksheet(wb, name)
    writeData(wb, name, df)
  }
}