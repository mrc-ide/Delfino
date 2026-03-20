library(tidyverse)
library(patchwork)

# --- 1. CONFIGURATION ---
FILE_CONTROL   <- "control_0_7143_incidence.csv"
FILE_ALWAYS    <- "treated_always_0_7143_incidence.csv"
FILE_TRIGGER   <- "treated_on_diagnosis_E66-E11-E67_0_7143_incidence.csv"

TARGET_DISEASES <- c("I10", "E11", "I21", "I63", "I50", "N18")
DISEASE_NAMES   <- c("Hypertension", "Type 2 Diabetes", "Heart Attack", 
                     "Stroke", "Heart Failure", "Kidney Disease")

# Mapping from delfino.py affected_diseases
# Efficacy = (1 - HR) * 100
EFFICACY_MAP <- c(
  "I10" = "99.9% Risk Reduction\n(Test Cure)",
  "E11" = "94% Risk Reduction\n(SURMOUNT-3)",
  "I50" = "50% Risk Reduction\n(SUMMIT / STEP-HFpEF)",
  "I21" = "22% Risk Reduction\n(SELECT / SUSTAIN-6)",
  "I63" = "22% Risk Reduction\n(SELECT / SUSTAIN-6)",
  "N18" = "24% Risk Reduction\n(FLOW)",
  "Death" = "18% Mortality Reduction\n(LEADER/SELECT/FLOW)"
)

# --- 2. DATA LOADING & PREP ---
load_and_label <- function(file, scenario_name) {
  if(!file.exists(file)) stop(paste("File not found:", file))
  
  read_csv(file, show_col_types = FALSE) %>%
    select(PatientID, SimulationStartAge, any_of(TARGET_DISEASES)) %>%
    pivot_longer(cols = any_of(TARGET_DISEASES), 
                 names_to = "Code", values_to = "AgeAtEvent") %>%
    mutate(
      Scenario = scenario_name,
      # CALENDAR TIME: Years after simulation start
      YearsFromStart = AgeAtEvent - SimulationStartAge
    )
}

df_all <- bind_rows(
  load_and_label(FILE_CONTROL, "Control (No Treatment)"),
  load_and_label(FILE_ALWAYS,  "Everyone Treated"),
  load_and_label(FILE_TRIGGER, "Target on Obesity/Diabetes diagnosis")
) %>%
  mutate(Disease = factor(Code, levels = TARGET_DISEASES, labels = DISEASE_NAMES))

# --- 3. PLOTTING FUNCTION ---
plot_disease <- function(target_code, time_mode = "calendar") {
  
  use_col <- if(time_mode == "calendar") "YearsFromStart" else "AgeAtEvent"
  x_label <- if(time_mode == "calendar") "Years Since Trial Start" else "Age (Years)"
  
  plot_data <- df_all %>%
    filter(Code == target_code) %>%
    filter(AgeAtEvent > 0) %>% 
    group_by(Scenario) %>%
    arrange(!!sym(use_col)) %>%
    mutate(
      Count = row_number(),
      Incidence = (Count / 7143) * 100
    ) %>%
    ungroup()
  
  if(nrow(plot_data) == 0) {
    message(paste("Warning: No events found for", target_code, "in", time_mode, "mode."))
    return(NULL)
  }
  
  disease_name <- DISEASE_NAMES[which(TARGET_DISEASES == target_code)]
  eff_text <- EFFICACY_MAP[target_code]
  
  ggplot(plot_data, aes(x = .data[[use_col]], y = Incidence, color = Scenario)) +
    geom_step(linewidth = 1.8) + 
    scale_color_manual(values = c(
      "Control (No Treatment)" = "black", 
      "Everyone Treated" = "#E69F00", 
      "Target on Obesity/Diabetes diagnosis" = "#56B4E9"
    )) +
    labs(
      title = paste0(disease_name, " (", target_code, ")"),
      subtitle = paste0("Assumed Efficacy: ", eff_text),
      x = x_label,
      y = "Cumulative Incidence (%)"
    ) +
    theme_minimal(base_size = 22) + 
    theme(
      plot.title = element_text(face = "bold", size = 34, margin = margin(b = 6), hjust = 0.5),
      plot.subtitle = element_text(size = 28, color = "gray30", margin = margin(b = 20), hjust = 0.5),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(size = 20, color = "black"),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black")
    )
}

# --- 4. GENERATE AND SAVE ---
for(i in seq_along(TARGET_DISEASES)) {
  code <- TARGET_DISEASES[i]
  
  # A. Calendar Plots
  p_cal <- plot_disease(code, time_mode = "calendar")
  if(!is.null(p_cal)) {
    ggsave(paste0("Slide_Calendar_", code, ".png"), p_cal, width = 11, height = 8, dpi = 300)
  }
  
  # B. Age Plots
  p_age <- plot_disease(code, time_mode = "age")
  if(!is.null(p_age)) {
    ggsave(paste0("Slide_Age_", code, ".png"), p_age, width = 11, height = 8, dpi = 300)
  }
}