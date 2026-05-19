library(tidyverse)
library(scales)

# --- 1. CONFIGURATION ---
FILE_CONTROL   <- "control_0_7143_incidence.csv"
FILE_ALWAYS    <- "treated_always_0_7143_incidence.csv"
FILE_TRIGGER   <- "treated_on_diagnosis_E66-E11-E67_0_7143_incidence.csv"

HEOR_METRICS <- c("Total_Costs", "Total_QALYs", "Total_YLDs", "Total_YLLs", "Total_DALYs")
METRIC_LABELS <- c("Total Costs (£)", "Total QALYs", "YLDs (Disability)", "YLLs (Mortality)", "Total DALYs (Burden)")

# --- 2. DATA LOADING & PROCESSING ---
load_heor <- function(file, scenario_name) {
  if(!file.exists(file)) stop(paste("File not found:", file))
  
  read_csv(file, show_col_types = FALSE) %>%
    select(PatientID, all_of(HEOR_METRICS)) %>%
    pivot_longer(cols = -PatientID, names_to = "Metric", values_to = "Value") %>%
    mutate(Scenario = scenario_name)
}

df_heor <- bind_rows(
  load_heor(FILE_CONTROL, "Control"),
  load_heor(FILE_ALWAYS,  "Everyone Treated"),
  load_heor(FILE_TRIGGER, "Target on Obesity/Diabetes")
) %>%
  mutate(
    Scenario = factor(Scenario, levels = c("Control", "Everyone Treated", "Target on Obesity/Diabetes")),
    Metric = factor(Metric, levels = HEOR_METRICS, labels = METRIC_LABELS)
  )

# Calculate means and Standard Errors for error bars
df_summary <- df_heor %>%
  group_by(Scenario, Metric) %>%
  summarise(
    mean_val = mean(Value),
    se_val = sd(Value) / sqrt(n()),
    .groups = 'drop'
  )

# --- 3. PLOTTING FUNCTION ---
# We create one plot per metric to keep them legible for PowerPoint
plot_heor_metric <- function(target_metric) {
  
  plot_data <- df_summary %>% filter(Metric == target_metric)
  
  # Format y-axis for costs
  y_labels <- if(grepl("Costs", target_metric)) label_comma(prefix = "£") else label_comma()
  
  ggplot(plot_data, aes(x = Scenario, y = mean_val, fill = Scenario)) +
    geom_col(color = "black", size = 0.8, width = 0.7) +
    # geom_errorbar(aes(ymin = mean_val - (1.96 * se_val), ymax = mean_val + (1.96 * se_val)), 
    #               width = 0.2, size = 1) +
    scale_fill_manual(values = c("Control" = "black", 
                                 "Everyone Treated" = "#E69F00", 
                                 "Target on Obesity/Diabetes" = "#56B4E9")) +
    scale_y_continuous(labels = y_labels, expand = expansion(mult = c(0, 0.1))) +
    labs(
      title = target_metric,
      #subtitle = "Mean Outcome per Patient (95% CI Error Bars)",
      x = NULL,
      y = NULL
    ) +
    theme_minimal(base_size = 24) +
    theme(
      plot.title = element_text(face = "bold", size = 32, hjust = 0.5),
      #plot.subtitle = element_text(size = 22, color = "gray30", hjust = 0.5, margin = margin(b=20)),
      axis.text.x = element_text(face = "bold", color = "black"),
      axis.text.y = element_text(size = 20, color = "black"),
      legend.position = "none", # Legend redundant as X-axis labels are clear
      panel.grid.major.x = element_blank(),
      axis.line.y = element_line(color = "black")
    )
}

# --- 4. SAVE PLOTS ---
for(m in METRIC_LABELS) {
  p <- plot_heor_metric(m)
  file_safe_name <- gsub("[^[:alnum:]]", "_", m)
  ggsave(paste0("HEOR_Summary_", file_safe_name, ".png"), p, width = 12, height = 8, dpi = 300)
}