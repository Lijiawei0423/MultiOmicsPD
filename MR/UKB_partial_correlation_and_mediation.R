set.seed(2025)
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tibble)
  library(ppcor)
  library(mediation)
})

# ---------------------- Omics to analyze ----------------------
prot <- c("protein")                   # Proteomic features
metab <- c("metabolic_feature")        # Metabolomic features
hb <- c("H&B_biomarker")               # H&B biomarker codes
hb_map <- c("code" = "name")           # Map H&B codes to names
min_n <- 30                             # Minimum sample size
y_col <- "target_y"                     # Outcome variable

covars <- c("age","educ","tdi","smk","alc","bmi","fastingtime","cl_med",
            "sex","race_bin")           # Covariates (binary where needed)

sims <- 1000                            # Mediation simulations

# ---------------------- Partial correlation function ----------------------
pcor_run_one <- function(trait, omic, dt) { 
  vars <- c(trait, omic, covars) 
  dt_sub <- dt[complete.cases(dt[, ..vars]), ..vars] 
  if(nrow(dt_sub) < min_n) return(NULL) 
  tryCatch({
    pc_spearman <- pcor.test(dt_sub[[trait]], dt_sub[[omic]], dt_sub[, ..covars], method="spearman")
    tibble(trait=trait, omic=omic, n=nrow(dt_sub),
           spearman_r=pc_spearman$estimate, spearman_stat=pc_spearman$statistic, spearman_p=pc_spearman$p.value)
  }, error=function(e) NULL)
}

# ---------------------- Run partial correlation for all omics ----------------------
run_pcor_all <- function(traits, omics, trait_df, prot_df, metab_df, hb_df=NULL, out_dir="pcor_results") {
  if(!dir.exists(out_dir)) dir.create(out_dir)
  for(omic in omics){
    cat("Processing Omic:", omic, "\n"); flush.console()
    # import dt_all: omic, covs, target, traits
    setDT(dt_all)
    res_list <- lapply(traits, function(trait) pcor_run_one(trait, omic, dt_all))
    results <- bind_rows(Filter(Negate(is.null), res_list))
    if(length(hb_map) > 0 && omic %in% names(hb_map)) results$omic <- hb_map[results$omic]
    out_file <- file.path(out_dir, paste0(omic_group, "_", omic, "_pcor.csv"))
    fwrite(results, out_file)
    cat("Saved results for:", omic, "->", out_file, "\n"); flush.console()
  }
}

# ---------------------- Mediation function ----------------------
medi_run_one <- function(trait, omic, dt){
  if(nrow(dt) < min_n) return(NULL)
  tryCatch({
    med_fit <- lm(as.formula(paste0("`", omic, "` ~ `", trait, "` + ", paste(covars, collapse=" + "))), data=dt)
    out_fit <- glm(as.formula(paste0("`", y_col, "` ~ `", trait, "` + `", omic, "` + ", paste(covars, collapse=" + "))),
                   data=dt, family=binomial())
    mo <- mediate(med_fit, out_fit, treat=trait, mediator=omic, robustSE=TRUE, sims=sims, boot=FALSE, dropobs=TRUE)
    tibble(trait=trait, mediator=ifelse(omic %in% names(hb_map), hb_map[omic], omic), outcome=y_col, n=nrow(dt),
           acme_est=mo$d0, acme_lo=mo$d0.ci[1], acme_hi=mo$d0.ci[2], acme_p=mo$d0.p,
           ade_est=mo$z0, ade_lo=mo$z0.ci[1], ade_hi=mo$z0.ci[2], ade_p=mo$z0.p,
           total_est=mo$tau.coef, total_lo=mo$tau.ci[1], total_hi=mo$tau.ci[2], total_p=mo$tau.p,
           prop_med=mo$n0, prop_lo=mo$n0.ci[1], prop_hi=mo$n0.ci[2])
  }, error=function(e) NULL)
}

# ---------------------- Run mediation for all omics ----------------------
run_mediation <- function(traits, omic, omic_df, trait_df, covs_df, out_file){
  results <- list()
  for(trait in traits){
    # import dt_all: omic, covs, target, traits
    res <- medi_run_one(trait, omic, dt_sub)
    if(!is.null(res)) results[[length(results)+1]] <- res
  }
  results_df <- bind_rows(results)
  fwrite(results_df, out_file)
  return(results_df)
}

# ---------------------- Example Run ----------------------
all_omics <- c(prot, metab, hb)
run_pcor_all(traits=traits_to_run, omics=all_omics, trait_df=trait_df,
             prot_df=prot_data_file, metab_df=metab_data_file, hb_df=hb_data_file,
             out_dir="pcor_results")
for(omic in all_omics){
  run_mediation(traits=traits_to_run, omic=omic,
                omic_df=if(omic %in% prot) prot_data_file else if(omic %in% metab) metab_data_file else hb_data_file,
                trait_df=trait_df, covs_df=covs_target_df, out_file="mediation_results")
}
cat("All analyses finished.\n")
