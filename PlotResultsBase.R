# 1. Load data
base_df <- read.csv("delfino_individual_base.csv")
glp_df  <- read.csv("delfino_individual_glp1.csv")
params  <- read.csv("dummy_disease_params.csv")
labels  <- readLines("data/ukb_simulated_data/labels.csv")

# 2. Identify incidence columns and calculate counts
inc_cols <- names(base_df)[grep("^inc_", names(base_df))]
base_counts <- colSums(base_df[, inc_cols] > 0)
glp_counts  <- colSums(glp_df[, inc_cols] > 0)

# 3. Create a data frame for the counts found in the data
counts_df <- data.frame(
  Code = sub("inc_", "", inc_cols),
  Base = as.vector(base_counts),
  GLP  = as.vector(glp_counts),
  stringsAsFactors = FALSE
)

# 4. Create the name map from params/labels
# R is 1-indexed, TokenID is 0-indexed
name_lookup <- data.frame(
  Code = params$Code,
  Name = labels[params$TokenID + 1],
  stringsAsFactors = FALSE
)

# 5. Merge safely (solves the 1256 vs 1255 mismatch)
disease_summary <- merge(name_lookup, counts_df, by = "Code", all.y = TRUE)
disease_summary$Delta <- disease_summary$Base - disease_summary$GLP

# 6. View the results (Sorted by most frequent in Base)
disease_summary <- disease_summary[order(disease_summary$Base, decreasing = TRUE), ]
disease_summary
print(head(disease_summary, 20))

# 7. Check our specific Trial Targets
targets <- c("E11", "I21", "I63", "I50", "N18")
print(disease_summary[disease_summary$Code %in% targets, ])