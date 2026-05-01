library(tidyverse)

# 1. SETUP
file_path <- "temp_manual_0_7143_trajectories.csv"

# read_csv (from readr) is much faster than read.csv
traj_data <- read_csv(file_path)

cat(sprintf("Profiling trajectories for %d patients...\n", nrow(traj_data)))

# 2. FAST EXTRACTION ENGINE
# We use vectorized string operations instead of a for-loop
process_trajectories <- function(df) {
  
  # Split the column into Input and Generated parts
  df_split <- df %>%
    separate(Trajectory, into = c("Input_Part", "Gen_Part"), 
             sep = "=====================", extra = "merge")
  
  # Function to parse codes and names from a text block
  parse_block <- function(text_column) {
    # 1. Split block into individual lines
    lines <- str_split(text_column, "\n")
    
    # 2. Flatten and extract components using regex
    # Pattern: [Age]: [Code] [Description]
    map_df(lines, ~{
      matches <- str_match(.x, "^[0-9.-]+: ([A-Z][0-9]+) (.*)$")
      as.data.frame(matches) %>%
        filter(!is.na(V2)) %>%
        select(Code = V2, Name = V3) %>%
        distinct()
    })
  }
  
  # Process both sections
  cat("Processing Input Histories...\n")
  input_stats <- parse_block(df_split$Input_Part) %>%
    group_by(Code, Name) %>%
    summarise(Input_Count = n(), .groups = 'drop') %>%
    mutate(Input_Prev_Pct = (Input_Count / nrow(df)) * 100)
  
  cat("Processing Generated Futures...\n")
  gen_stats <- parse_block(df_split$Gen_Part) %>%
    group_by(Code, Name) %>%
    summarise(Gen_Count = n(), .groups = 'drop') %>%
    mutate(Gen_Prev_Pct = (Gen_Count / nrow(df)) * 100)
  
  # Combine and Sort
  final_profile <- full_join(input_stats, gen_stats, by = c("Code", "Name")) %>%
    replace_na(list(Input_Count = 0, Input_Prev_Pct = 0, Gen_Count = 0, Gen_Prev_Pct = 0)) %>%
    arrange(desc(Gen_Prev_Pct))
  
  return(final_profile)
}

# 3. RUN AND SAVE
final_profile <- process_trajectories(traj_data)
print(head(final_profile, 20))
write.csv(final_profile, "trajectory_comparison_full.csv", row.names = FALSE)