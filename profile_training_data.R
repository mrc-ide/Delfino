# 1. SETUP
data_dir <- file.path("data", "ukb_simulated_data")
labels <- readLines(file.path(data_dir, "labels.csv"))
params <- read.csv("dummy_disease_params.csv")

# 2. LOAD BINARY DATA
# train.bin is uint16, with 48 tokens per patient
file_info <- file.info(file.path(data_dir, "train.bin"))
num_elements <- file_info$size / 2
raw_data <- readBin(file.path(data_dir, "train.bin"), what = "integer", n = num_elements, size = 2, signed = FALSE)

# Reshape into a matrix (Patients x 48 tokens)
num_patients <- length(raw_data) / 48
data_matrix <- matrix(raw_data, ncol = 48, byrow = TRUE)

# 3. CALCULATE PREVALENCE
# We only care about the diseases defined in your params file
disease_ids <- params$TokenID

# Check if each patient has each disease at least once in their 48-token history
counts <- sapply(disease_ids, function(tid) {
  sum(rowSums(data_matrix == tid) > 0)
})

# 4. CREATE SUMMARY TABLE
prevalence_df <- data.frame(
  Code = params$Code,
  Name = labels[params$TokenID + 1],
  Count = counts,
  Prevalence_Pct = (counts / num_patients) * 100
)

# Sort by frequency
prevalence_df <- prevalence_df[order(prevalence_df$Count, decreasing = TRUE), ]

# 5. PRINT THE TRUTH
cat("\n--- 📊 TRAINING DATA PREVALENCE PROFILE ---\n")
cat("Total Patients in train.bin:", num_patients, "\n")

cat("\n[Top 20 Most Frequent Diseases in Training Set]\n")
print(head(prevalence_df, 20), row.names = FALSE)

cat("\n[Trial Target Presence Check]\n")
targets <- c("E11", "I21", "I63", "I50", "N18")
print(prevalence_df[prevalence_df$Code %in% targets, ], row.names = FALSE)

# Save for reference
write.csv(prevalence_df, "training_data_prevalence.csv", row.names = FALSE)