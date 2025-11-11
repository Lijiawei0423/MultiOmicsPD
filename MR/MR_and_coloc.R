# ======================== MR Pipeline (Simplified & Modularized) ========================
set.seed(2025)
suppressPackageStartupMessages({
  library(TwoSampleMR)
  library(dplyr)
  library(fs)
  library(data.table)
  library(stringr)
})

# ---- Load data ----
ieugwasr::user()
finn_oc <- fread("finngen_R10_G6_PARKINSON", header = TRUE)
timestamp <- format(Sys.time(), "%y%m%d_%H%M")
all_res <- paste0("MR_", timestamp)
dir_create <- function(path) dir.create(path, recursive = TRUE, showWarnings = FALSE)

# ---- Omic lists (placeholders) ----
hb    <- read.csv("<HB_LIST_PATH>", header = TRUE)
metab <- read.csv("<METAB_LIST_PATH>", header = TRUE)
prot  <- read.csv("<PROT_LIST_PATH>", header = TRUE)

# ---- Utility for console output ----
cat_flush <- function(core_id, ...) {
  cat("[Core", core_id, "]", ..., "\n")
  flush.console()
}

# ========================== BLOOD BIOMARKERS ANALYSIS ==========================
hb_res_path <- file.path(all_res, "hb")
dir_create(hb_res_path)

for (i in seq_len(nrow(hb))) {
  core_id <- 1
  hb_name <- hb$hb_omic[i]
  ieu_id  <- hb$hb_ieu_id[i]
  ieu_val <- paste0("ukb-d-", ieu_id, "_irnt")
  
  res_path <- file.path(hb_res_path, paste0(hb_name, "-", ieu_id))
  mr_res_file <- file.path(res_path, paste0(hb_name, "-", ieu_id, "_MR_result.csv"))
  if (file.exists(mr_res_file)) { cat_flush(core_id, "SKIP:", ieu_val); next }
  
  cat_flush(core_id, "Processing:", ieu_val)
  dir_create(res_path)
  
  # --- Extract exposure instruments ---
  exp_clumped <- tryCatch({
    extract_instruments(ieu_val, p1 = 5e-08, clump = TRUE, r2 = 0.01, kb = 1000)
  }, error = function(e) { cat_flush(core_id, "ERROR:", ieu_val, conditionMessage(e)); NULL })
  
  if (is.null(exp_clumped) || nrow(exp_clumped) == 0) { cat_flush(core_id, "No IVs found"); next }
  exp_clumped <- subset(exp_clumped, !(chr.exposure == 6 & pos.exposure >= 25500000 & pos.exposure <= 34000000))
  
  write.csv(exp_clumped, file.path(res_path, paste0(hb_name, "-", ieu_id, "-IV.csv")), row.names = FALSE)
  
  # --- Prepare outcome ---
  instr_data <- merge(exp_clumped, finn_oc, by.x = "SNP", by.y = "rsids")
  outcome_data <- instr_data %>% select(SNP, ref, alt, pval, af_alt, beta, sebeta)
  write.csv(outcome_data, file.path(res_path, paste0(hb_name, "-", ieu_id, "-oc.csv")), row.names = FALSE)
  
  outcome <- read_outcome_data(
    filename = file.path(res_path, paste0(hb_name, "-", ieu_id, "-oc.csv")),
    snps = exp_clumped$SNP, sep = ",",
    snp_col = "SNP", beta_col = "beta", se_col = "sebeta",
    effect_allele_col = "alt", other_allele_col = "ref",
    eaf_col = "af_alt", pval_col = "pval"
  )
  
  # --- MR analysis ---
  mrdata <- harmonise_data(exp_clumped, outcome)
  result <- mr(mrdata)
  OR_res <- generate_odds_ratios(result)
  hetero <- mr_heterogeneity(mrdata)
  pleio  <- mr_pleiotropy_test(mrdata)
  
  saveRDS(list(result = result, OR = OR_res, hetero = hetero, pleio = pleio, path = res_path),
          file = file.path("result"))
  
  cat_flush(core_id, "Finished:", hb_name, "-", ieu_id)
}


# ========================== METABOLOMICS ANALYSIS ==========================
metab_res_path <- file.path(all_res, "metab")
dir_create(metab_res_path)

file_list <- list.files("<METAB_DATA_PATH>", pattern = "\\.tsv$", full.names = TRUE, recursive = TRUE)
process_metab <- function(file_path, core_id = 1) {
  omic <- basename(file_path)
  res_path <- file.path(metab_res_path, omic)
  mr_res_file <- file.path(res_path, paste0(omic, "_MR_result.csv"))
  if (file.exists(mr_res_file)) { cat_flush(core_id, "SKIP:", omic); return(NULL) }
  
  cat_flush(core_id, "Processing:", omic)
  
  # import lead snps data
  exp <- read_exposure_data(file_path, sep = "\t", snp_col = "ID", beta_col = "BETA", se_col = "SE",
                            effect_allele_col = "A1", other_allele_col = "AX",
                            eaf_col = "A1_FREQ", pval_col = "P", log_pval = FALSE,
                            chr_col = "CHR", pos_col = "POS")
  
  iv <- merge(exp, finn_oc, by.x = "SNP", by.y = "rsids")
  dir_create(res_path)
  write.csv(iv, file.path(res_path, paste0(omic, "_IV.csv")), row.names = FALSE)
  outcome <- read_outcome_data(file.path(res_path, paste0(omic, "_IV.csv")), snps = iv$SNP,
                               sep = ",", snp_col = "SNP", beta_col = "beta", se_col = "sebeta",
                               effect_allele_col = "alt", other_allele_col = "ref",
                               eaf_col = "af_alt", pval_col = "pval")
  
  mrdata <- harmonise_data(iv, outcome)
  result <- mr(mrdata)
  OR_res <- generate_odds_ratios(result)
  hetero <- mr_heterogeneity(mrdata)
  pleio  <- mr_pleiotropy_test(mrdata)
  
  saveRDS(list(result = result, OR = OR_res, hetero = hetero, pleio = pleio, path = res_path),
          file = file.path("result"))
  
  cat_flush(core_id, "Finished:", omic)
}
invisible(lapply(file_list, process_metab))
 



# ========================== PROTEOMICS ANALYSIS ==========================
prot_res_path <- file.path(all_res, "prot")
dir_create(prot_res_path)

# process UKBPPP pQTL tar file
process_protein_tar <- function(tar_file) {
  protein <- file_path_sans_ext(basename(tar_file))
  cat("Processing protein:", protein, "\n"); flush.console()
  tmp_dir <- file.path(temp_dir, paste0("tmp_", protein))
  dir.create(tmp_dir, showWarnings = FALSE)
  untar(tar_file, exdir = tmp_dir)
  gz_files <- list.files(tmp_dir, pattern = "\\.gz$", recursive = TRUE, full.names = TRUE)
  result_list <- list()
  for (gz in gz_files) {
    reg_file <- tempfile(fileext = ".regenie")
    gunzip(gz, destname = reg_file, remove = FALSE, overwrite = TRUE)
    
    # convert nlogP to P
    # filter: p<5e-8
    dt <- tryCatch(fread(reg_file, header = TRUE, sep = " "), 
                   error = function(e) fread(reg_file, header = TRUE, sep = "\t"))
    file.remove(reg_file)
    if (!"LOG10P" %in% names(dt)) next
    setnames(dt, "LOG10P", "nlogP")
    dt[, P := 10^(-nlogP)]
    dt <- dt[nlogP > -log10(5e-8)]
    if ("CHROM" %in% names(dt)) dt[CHROM %in% c("23", 23), CHROM := "X"]
    result_list[[length(result_list) + 1]] <- dt
  }
  
  if (length(result_list) > 0) {
    final_dt <- rbindlist(result_list, fill = TRUE)
    fwrite(final_dt, file.path(filtered_dir, paste0(protein, ".csv")))
    cat("  Saved", nrow(final_dt), "rows for", protein, "\n")
  } else {
    cat("  No valid data for", protein, "\n")
  }
  unlink(tmp_dir, recursive = TRUE)
}
tar_files <- list.files(temp_dir, pattern = "\\.tar$", full.names = TRUE)
for (tar_file in tar_files) process_protein_tar(tar_file)



# convert snps location to rsid
library(SNPlocs.Hsapiens.dbSNP144.GRCh38)
snps <- SNPlocs.Hsapiens.dbSNP144.GRCh38
csv_files <- list.files(filtered_dir, pattern = "\\.csv$", full.names = TRUE)
convert_to_rsid <- function(file) {
  exp <- fread(file)
  base_name <- file_path_sans_ext(basename(file))
  cat("Converting:", base_name, "\n"); flush.console()
  
  exp[, CHROM := as.character(CHROM)]
  exp <- exp[CHROM %in% c(as.character(1:22), "X", "Y", "MT")]
  exp <- exp[!is.na(GENPOS)]
  exp[, rsid := NA_character_]
  
  for (chr in unique(exp$CHROM)) {
    pos <- exp[CHROM == chr, GENPOS]
    if (length(pos) == 0) next
    chr_snps <- snpsBySeqname(snps, chr)
    if (length(chr_snps) == 0) next
    
    idx <- match(pos, pos(chr_snps))
    exp[CHROM == chr, rsid := mcols(chr_snps)$RefSNP_id[idx]]
  }
  
  exp <- exp[!is.na(rsid)]
  desired_order <- c("rsid","CHROM","GENPOS","ID","ALLELE0","ALLELE1",
                     "A1FREQ","INFO","N","TEST","BETA","SE","CHISQ","nlogP","P")
  existing_cols <- intersect(desired_order, names(exp))
  exp <- exp[, ..existing_cols]
  setnames(exp, "rsid", "RSID")
  fwrite(exp, file.path(rsid_res_dir, paste0(base_name, "PROTEIN_pQTL_with_RSID.csv")))
  cat("  RSID conversion done for", base_name, "(", nrow(exp), "rows)\n")
}
for (file in csv_files) convert_to_rsid(file)


rsid_files <- list.files("<PROTEIN_pQTL_with_RSID_PATH>", pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
process_protein <- function(file_path, core_id = 1) {
  protein <- gsub("\\.csv$", "", basename(file_path))
  res_path <- file.path(prot_res_path, protein)
  mr_res_file <- file.path(res_path, paste0(protein, "_MR_result.csv"))
  if (file.exists(mr_res_file)) { cat_flush(core_id, "SKIP:", protein); return(NULL) }
  
  cat_flush(core_id, "Processing protein:", protein)
  
  exp <- read_exposure_data(file_path, sep = ",", snp_col = "RSID", beta_col = "BETA",
                            se_col = "SE", effect_allele_col = "ALLELE1", other_allele_col = "ALLELE0",
                            eaf_col = "A1FREQ", samplesize_col = "N", pval_col = "P", log_pval = FALSE)
  
  exp_clumped <- clump_data(exp, clump_kb = 1000, clump_r2 = 0.01, pop = "EUR")
  exp_clumped <- subset(exp_clumped, !(chr.exposure == 6 & pos.exposure >= 25500000 & pos.exposure <= 34000000))
  if (nrow(exp_clumped) == 0) { cat_flush(core_id, "No IVs after clumping:", protein); return(NULL) }
  
  iv <- merge(exp_clumped, finn_oc, by.x = "SNP", by.y = "rsids")
  iv$r2.exp <- (2 * iv$beta.exposure^2 * iv$eaf.exposure * (1 - iv$eaf.exposure)) /
    (2 * (iv$beta.exposure^2) * iv$eaf.exposure * (1 - iv$eaf.exposure) +
       2 * iv$samplesize.exposure * iv$eaf.exposure * (1 - iv$eaf.exposure) * iv$se.exposure^2)
  iv$F.exp <- iv$r2.exp * (iv$samplesize.exposure - 2) / (1 - iv$r2.exp)
  iv <- subset(iv, F.exp >= 10)
  if (nrow(iv) == 0) { cat_flush(core_id, "No valid IVs (F>=10):", protein); return(NULL) }
  
  dir_create(res_path)
  write.csv(iv, file.path(res_path, paste0(protein, "_IV.csv")), row.names = FALSE)
  
  outcome <- read_outcome_data(file.path(res_path, paste0(protein, "_IV.csv")),
                               snps = iv$SNP, sep = ",", snp_col = "SNP",
                               beta_col = "beta", se_col = "sebeta",
                               effect_allele_col = "alt", other_allele_col = "ref",
                               eaf_col = "af_alt", pval_col = "pval")
  
  mrdata <- harmonise_data(iv, outcome)
  result <- mr(mrdata)
  OR_res <- generate_odds_ratios(result)
  hetero <- mr_heterogeneity(mrdata)
  pleio  <- mr_pleiotropy_test(mrdata)
  
  saveRDS(list(result = result, OR = OR_res, hetero = hetero, pleio = pleio, path = res_path),
          file = file.path("result"))
  
  cat_flush(core_id, "Finished protein:", protein)
}
invisible(lapply(rsid_files, process_protein))





# ========================== RESULT SAVING & SIGNIFICANCE CHECK ==========================
sig_folder <- file.path(all_res, "0-SIG")
nsensitive_folder <- file.path(all_res, "0-nsensitive")
dir_create(sig_folder); dir_create(nsensitive_folder)

tmp_files <- list.files(all_res, pattern = "_MR_tmp\\.rds$", recursive = TRUE, full.names = TRUE)
for (f in tmp_files) {
  res <- readRDS(f)
  with(res, {
    write.csv(result, file.path(path, "MR_result.csv"), row.names = FALSE)
    write.csv(OR, file.path(path, "MR_OR_result.csv"), row.names = FALSE)
    write.csv(hetero, file.path(path, "MR_heter.csv"), row.names = FALSE)
    write.csv(pleio, file.path(path, "MR_pleio.csv"), row.names = FALSE)
    
    if (any(result$pval < 0.05, na.rm = TRUE)) {
      hetero_issue <- any(hetero$Q_pval < 0.05, na.rm = TRUE)
      pleio_issue  <- any(pleio$pval < 0.05, na.rm = TRUE)
      dest <- if (hetero_issue | pleio_issue) nsensitive_folder else sig_folder
      dest_path <- file.path(dest, basename(path))
      if (dir.exists(dest_path)) unlink(dest_path, recursive = TRUE)
      file.rename(path, dest_path)
    }
  })
  file.remove(f)
}

# ---- Copy script for reproducibility ----
script_path <- sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])
if (length(script_path) > 0) {
  file.copy(script_path, file.path(all_res, paste0("MR_script_run_", timestamp, ".R")), overwrite = TRUE)
}
cat("=== MR pipeline completed ===\n")






# ========================== COLOC ANALYSIS FOR PROTEINS ==========================
library(coloc)
# ---------------------- Load outcome GWAS (FinnGen) ----------------------
finn_path <- "<PATH_TO_FINNGEN_GWAS>"
finn <- fread(finn_path)
finn$ncase <- "case number"
finn$ncontrol <- "control number"
finn$samplesize <- finn$ncase + finn$ncontrol

outcome <- finn %>% 
  dplyr::select(rsids, "#chrom", pos, alt, ref, af_alt, beta, sebeta, pval) %>%
  dplyr::rename(SNP = rsids,
                chrom = `#chrom`,
                effect_allele = alt,
                other_allele = ref,
                eaf = af_alt,
                se = sebeta,
                P = pval) %>%
  unique() %>%
  mutate(varbeta = se^2,
         MAF = ifelse(eaf < 0.5, eaf, 1 - eaf),
         s = ncase / samplesize,
         z = beta / se) %>%
  na.omit()

# ---------------------- Define proteins ----------------------
prots <- c("proteins")

# ---------------------- Paths ----------------------
rsid_res_dir <- "<PATH_TO_PROTEIN_RSID_CSV>"
coloc_res_dir <- "<PATH_TO_SAVE_COLOC_RESULTS>"
dir.create(coloc_res_dir, recursive = TRUE, showWarnings = FALSE)

rsid_files <- list.files(rsid_res_dir, pattern = "_RSID38\\.csv$", full.names = TRUE)

# ---------------------- Console helper ----------------------
cat_flush <- function(msg) { cat(msg, "\n"); flush.console() }

# ---------------------- COLOC analysis function ----------------------
coloc_protein <- function(file_path, protein_name){
  
  # Create result folder
  res_file <- file.path(coloc_res_dir, paste0(protein_name, "_snp_pph4.csv"))
  if(file.exists(res_file)){
    cat_flush(paste("SKIP:", protein_name, "(result exists)"))
    return(NULL)
  }
  
  cat_flush(paste("Processing protein:", protein_name))
  
  # ---- Load exposure pQTL ----
  exp <- fread(file_path)
  colnames(exp) <- c('SNP','chrom','start','end','effect_allele','other_allele',
                     'eaf','beta','se','P','samplesize')
  exp$varbeta <- exp$se^2
  exp$MAF <- ifelse(exp$eaf < 0.5, exp$eaf, 1 - exp$eaf)
  exp$z <- exp$beta / exp$se
  
  # ---- Identify lead SNP and select +/-1Mb region ----
  lead <- exp %>% arrange(P) %>% slice(1)
  leadchr <- lead$chrom
  leadstart <- lead$start
  leadend <- lead$end
  
  QTLdata <- exp %>%
    filter(chrom == leadchr,
           start > leadstart - 1e6,
           end < leadend + 1e6) %>%
    distinct(SNP, .keep_all = TRUE) %>%
    na.omit()
  
  # Replace P=0 with minimal P
  QTLdata$P[QTLdata$P == 0] <- NA
  min_p <- min(QTLdata$P, na.rm = TRUE)
  QTLdata$P[is.na(QTLdata$P)] <- min_p * 0.01
  
  # ---- Match SNPs with outcome ----
  shared_SNP <- intersect(QTLdata$SNP, outcome$SNP)
  QTLdata <- QTLdata %>% filter(SNP %in% shared_SNP) %>% arrange(SNP)
  GWASdata <- outcome %>% filter(SNP %in% shared_SNP) %>% arrange(SNP)
  
  # ---- Run coloc ----
  coloc_res <- coloc.abf(
    dataset1 = list(pvalues = GWASdata$P, snp = GWASdata$SNP,
                    type = "cc", s = GWASdata$s[1], N = GWASdata$samplesize[1]),
    dataset2 = list(pvalues = QTLdata$P, snp = QTLdata$SNP,
                    type = "quant", N = QTLdata$samplesize[1]),
    MAF = QTLdata$MAF
  )
  
  # Sort results by posterior probability for H4
  SNP_result <- coloc_res$results %>% arrange(desc(SNP.PP.H4))
  
  # ---- Save result ----
  fwrite(SNP_result, file = res_file)
  cat_flush(paste("Saved COLOC results for", protein_name, "->", res_file))
}

# ---------------------- Run analysis for all proteins ----------------------
for(i in seq_along(rsid_files)){
  protein_name <- str_remove(basename(rsid_files[i]), "\\.csv$")
  coloc_protein(rsid_files[i], protein_name)
}
