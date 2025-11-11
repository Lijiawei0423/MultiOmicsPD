# ========================== PPMI PARTIAL CORRELATION ==========================
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tibble)
  library(ppcor)
})
set.seed(2025)

# ------------------ Paths & Output ------------------
time_stamp <- format(Sys.time(), "%y%m%d%H%M")  # Timestamp for output folder
res_dir <- file.path("RESULTS", paste0("PPMI_pcor_", time_stamp))  # Result directory
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)  # Create directory if not exist
plasma_res_file <- file.path(res_dir, "PPMI_pcor_plasma.csv")  # Plasma output file
csf_res_file    <- file.path(res_dir, "PPMI_pcor_csf.csv")     # CSF output file

# ------------------ Parameters ------------------
prots <- c("protein")  # Proteins to analyze
covars <- c("age", "EDUCYRS", "duration_yrs", "BMI", 
            "sex_bin", "race_bin") # Convert sex and race to binary (0/1)
traits <- c("abeta","asyn","ptau","NFL_CSF","nfl_serum",
            "mean_caudate","mean_putamen","mean_striatum",
            "updrs1_score","updrs2_score","updrs3_score",
            "moca","MSEADLG")  # Traits to correlate

# ------------------ Load Data ------------------
# import PPMI baseline data
# merge dataframe: protein(X), covariates, biomarkers and clinical scores(Y)

plasma_merged <- merge_data(plasma_prot_df, cov_df, trait_df)  # Merge plasma data
csf_merged <- merge_data(csf_prot_df, cov_df, trait_df)        # Merge CSF data

# =================== Combined Partial Correlation Function ===================
ppcor_PPMI <- function(merged_data, prots, traits, covars, res_file){
  results <- list()  # Initialize results list
  for(prot in prots){
    if(!prot %in% names(merged_data)) next  # Skip if protein not in data
    for(trait in traits){
      if(!trait %in% names(merged_data)) next  # Skip if trait not in data
      covs <- covars[covars %in% names(merged_data)]  # Use available covariates
      df_sub <- na.omit(merged_data[, c(prot, trait, covs), drop=FALSE])  # Subset and remove NA
      n_sub <- nrow(df_sub)  # Sample size
      if(n_sub < 3){  # Skip if too few samples
        results[[length(results)+1]] <- tibble(
          x=prot, y=trait, covs=paste(covs, collapse=","),
          n=n_sub, pearson_r=NA_real_, pearson_p=NA_real_
        )
        next
      }
      df_z <- df_sub  # Initialize z-scored data
      df_z[[trait]] <- if(is.numeric(df_z[[trait]]) && length(unique(df_z[[trait]]))>10) scale(df_z[[trait]]) else df_z[[trait]]  # z-score trait
      for(cv in covs){ if(is.numeric(df_z[[cv]]) && length(unique(df_z[[cv]]))>10) df_z[[cv]] <- scale(df_z[[cv]]) }  # z-score covariates
      pc <- tryCatch(pcor.test(df_z[[prot]], df_z[[trait]], df_z[, covs, drop=FALSE], method="pearson"), error=function(e) NULL)  # Partial correlation
      results[[length(results)+1]] <- tibble(
        x=prot, y=trait, covs=paste(covs, collapse=","),
        n=n_sub,
        pearson_r = if(!is.null(pc)) pc$estimate else NA_real_,
        pearson_p = if(!is.null(pc)) pc$p.value else NA_real_
      )
    }
  }
  results_df <- bind_rows(results)  # Combine results
  fwrite(results_df, res_file)      # Save results to file
  return(results_df)
}

# ------------------ Run Analysis ------------------
plasma_res_all <- ppcor_PPMI(plasma_merged, prots, traits, covars, plasma_res_file)  # Run plasma partial correlation
csf_res_all    <- ppcor_PPMI(csf_merged, prots, traits, covars, csf_res_file)       # Run CSF partial correlation

cat("=== ALL partial correlation results saved to:", res_dir, "===\n")
