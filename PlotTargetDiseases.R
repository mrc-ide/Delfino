library(tidyverse)
library(patchwork)

# --- 1. CONFIGURATION ---
FILE_CONTROL   <- "control_0_7143_incidence.csv"
# FILE_ALWAYS    <- "treated_always_0_7143_incidence.csv"
FILE_TRIGGEER_OB    <- "treated_on_diagnosis_E66_0_7143_incidence.csv"
FILE_TRIGGER   <- "treated_on_diagnosis_E66-E11-E67_0_7143_incidence.csv"

# TARGET_DISEASES <- c("E11", "I21", "I63", "I50", "N18", "Death")
# DISEASE_NAMES   <- c("Type 2 Diabetes", "Heart Attack", 
#                      "Stroke", "Heart Failure", "Kidney Disease", "Death")
TARGET_DISEASES <- c("E11", "I21", "I63", "I50", "N18")
DISEASE_NAMES   <- c("Type 2 Diabetes", "Heart Attack", 
                     "Stroke", "Heart Failure", "Kidney Disease")

# Mapping from delfino.py affected_diseases
# Efficacy = (1 - HR) * 100
EFFICACY_MAP <- c(
  # "I10" = "99.9% Risk Reduction\n(Test Cure)",
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
  load_and_label(FILE_CONTROL, "Status Quo"),
  # load_and_label(FILE_ALWAYS,  "Everyone Treated"),
  load_and_label(FILE_TRIGGEER_OB, "Target on Obesity diagnosis"),
  load_and_label(FILE_TRIGGER, "Target on Obesity or Diabetes diagnosis")
) %>%
  mutate(Disease = factor(Code, levels = TARGET_DISEASES, labels = DISEASE_NAMES))

# --- 3. PLOTTING FUNCTION ---
plot_disease <- function(target_code, time_mode = "calendar") {
  
  use_col <- if(time_mode == "calendar") "YearsFromStart" else "AgeAtEvent"
  x_label <- if(time_mode == "calendar") "Years Since Policy Started" else "Age (Years)"
  
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
  
  # Find the furthest time point across all scenarios for this disease
  max_time <- max(plot_data[[use_col]], na.rm = TRUE)
  
  # For each scenario, add a "dummy" row at max_time with the last known incidence
  plot_data <- plot_data %>%
    group_by(Scenario) %>%
    group_modify(~ {
      last_row <- slice_tail(.x, n = 1)
      dummy_row <- last_row
      dummy_row[[use_col]] <- max_time
      bind_rows(.x, dummy_row)
    }) %>%
    ungroup()
  
  if(nrow(plot_data) == 0) {
    message(paste("Warning: No events found for", target_code, "in", time_mode, "mode."))
    return(NULL)
  }
  
  disease_name <- DISEASE_NAMES[which(TARGET_DISEASES == target_code)]
  if (disease_name == "Death") Plot_Title = "Death" else Plot_Title = paste0(disease_name, " (", target_code, ")")
  
  # eff_text <- EFFICACY_MAP[target_code]
  
  ggplot(plot_data, aes(x = .data[[use_col]], y = Incidence, color = Scenario)) +
    geom_step(linewidth = 1.8) + 
    scale_x_continuous(expand = c(0, 0)) +
    scale_color_manual(values = c(
      "Status Quo" = "black", 
      # "Everyone Treated" = "#E69F00", 
      "Target on Obesity diagnosis" = "lightblue",
      "Target on Obesity or Diabetes diagnosis" = "blue"
    )) +
    labs(
      title = Plot_Title,
      # subtitle = paste0("Assumed Efficacy: ", eff_text),
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

# Collect plots into lists
calendar_plots <- list()
age_plots <- list()

for(code in TARGET_DISEASES) {
  calendar_plots[[code]] <- plot_disease(code, time_mode = "calendar")
  age_plots[[code]] <- plot_disease(code, time_mode = "age")
}

# Filter out any NULLs (diseases with no events)
calendar_plots <- Filter(Negate(is.null), calendar_plots)
age_plots <- Filter(Negate(is.null), age_plots)


layout_design <- "
  112233
  #4455#
"
# 1. Calendar Grid
# combined_calendar <- wrap_plots(calendar_plots, ncol = 3, nrow = 2) + 
combined_calendar <- wrap_plots(calendar_plots, design = layout_design) + 
  plot_layout(guides = "collect") & 
  theme(
    legend.position = "bottom",
    # 1. MAKE LEGEND BIGGER
    legend.text = element_text(size = 32, face = "bold"),
    legend.key.size = unit(1.5, "cm"), # Makes the colored lines in the legend longer/thicker
    
    # 2. PUT MORE SPACE BETWEEN PLOTS
    # margin(top, right, bottom, left)
    plot.margin = margin(20, 20, 20, 20) 
  )

ggsave("Grid_Calendar_Full.png", combined_calendar, width = 25, height = 15, dpi = 300)

# 2. Age Grid
combined_age <- wrap_plots(age_plots, ncol = 3, nrow = 2) + 
  plot_layout(guides = "collect") & 
  theme(
    legend.position = "bottom",
    # 1. MAKE LEGEND BIGGER
    legend.text = element_text(size = 22, face = "bold"),
    legend.key.size = unit(1.5, "cm"),
    
    # 2. PUT MORE SPACE BETWEEN PLOTS
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave("Grid_Age_Full.png", combined_age, width = 20, height = 12, dpi = 300)
