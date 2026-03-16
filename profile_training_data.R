# --- Corrected profile_training_data.R ---

# 1. SETUP
data_dir <- file.path("data", "ukb_simulated_data")
labels <- readLines(file.path(data_dir, "labels.csv"))
params <- read.csv("dummy_disease_params.csv")

# 2. LOAD BINARY DATA (The Ground Truth Way)
# train.bin is uint32 triplets: [PatientID, AgeInDays, TokenID]
file_path <- file.path(data_dir, "train.bin")
file_size <- file.info(file_path)$size
num_elements <- file_size / 4 # uint32 is 4 bytes

# Read as 32-bit integers
raw_data <- readBin(file_path, what = "integer", n = num_elements, size = 4, signed = FALSE)

# Reshape into a 3-column matrix
# Note: R fills matrices by column by default, so we specify byrow = TRUE
data_matrix <- matrix(raw_data, ncol = 3, byrow = TRUE)
colnames(data_matrix) <- c("PatientID", "Age", "TokenID")

# Calculate unique patients
unique_patients <- unique(data_matrix[, "PatientID"])
total_n <- length(unique_patients)

# 3. CALCULATE PREVALENCE
# We search for the TokenIDs (Raw indices from labels.csv)
disease_ids <- params$TokenID

cat("\n--- 📊 TRAINING DATA PREVALENCE PROFILE (3-Column Loader) ---\n")
cat(sprintf("Total Unique Patients: %d\n", total_n))

# Use a vectorized approach to find which patients have which tokens
# This is much faster than a loop in R
prevalence_results <- lapply(disease_ids, function(tid) {
  # Find all rows matching this TokenID
  matching_rows <- data_matrix[data_matrix[, "TokenID"] == tid, "PatientID"]
  
  # Count unique patients among those rows
  count <- length(unique(matching_rows))
  
  return(data.frame(
    TokenID = tid,
    Count = count,
    Prevalence_Pct = (count / total_n) * 100
  ))
})

# 4. CREATE SUMMARY TABLE
prevalence_df <- do.call(rbind, prevalence_results)

# Attach names and codes
prevalence_df$Code <- params$Code
prevalence_df$Name <- labels[prevalence_df$TokenID + 1] # +1 because R is 1-indexed

# Sort by frequency
prevalence_df <- prevalence_df[order(-prevalence_df$Count), ]

# 5. PRINT THE TRUTH
print(head(prevalence_df[, c("Code", "Name", "Count", "Prevalence_Pct")], 20), row.names = FALSE)

# 6. SAVE RESULTS
write.csv(prevalence_df, "training_prevalence_profile.csv", row.names = FALSE)
cat("\nProfile saved to training_prevalence_profile.csv\n")