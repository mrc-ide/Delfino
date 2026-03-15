library(tidyverse)

# 1. Load Results and Metadata
base_data <- read_csv("delfino_individual_base.csv")
glp_data <- read_csv("delfino_individual_glp1.csv")
params <- read_csv("dummy_disease_params.csv")
# Load labels - assuming you can read the text file line by line
labels_list <- readLines("data/ukb_simulated_data/labels.csv")



# Create a mapping table: Code -> FullName
code_mapping <- params %>%
  mutate(FullName = labels_list[TokenID + 1]) %>% # +1 because R is 1-indexed
  select(Code, FullName)

# 2. Identify and Process Deltas
inc_cols <- names(base_data)[startsWith(names(base_data), "inc_")]


base_data[,inc_cols] - glp_data[,inc_cols]


get_counts <- function(df, label) {
  df %>%
    select(all_of(inc_cols)) %>%
    pivot_longer(everything(), names_to = "Disease", values_to = "Age") %>%
    filter(Age > 0) %>%
    mutate(Code = str_replace(Disease, "inc_", "")) %>%
    group_by(Code) %>%
    mutate(Cumulative_Cases = row_number(), Group = label)
}

plot_data_raw <- bind_rows(get_counts(base_data, "Baseline"), get_counts(glp_data, "GLP-1"))

# Join with names
plot_data <- plot_data_raw %>%
  left_join(code_mapping, by = "Code") %>%
  mutate(DisplayTitle = ifelse(is.na(FullName), Code, FullName))

# Find top 10 by Delta for filtering
top_10_codes <- plot_data %>%
  group_by(Code, Group) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = Group, values_from = n, values_fill = 0) %>%
  mutate(delta = abs(Baseline - `GLP-1`)) %>%
  arrange(desc(delta)) %>%
  head(10) %>%
  pull(Code)

# 3. Final ggplot
ggplot(plot_data %>% filter(Code %in% top_10_codes), 
       aes(x = Age, y = Cumulative_Cases, color = Group)) +
  geom_step(size = 1.2) +
  facet_wrap(~ DisplayTitle, scales = "free_y", ncol = 2) +
  theme_minimal(base_size = 14) +
  scale_color_manual(values = c("Baseline" = "gray70", "GLP-1" = "#E64B35FF")) +
  labs(title = "Delfino: Clinical Impact of GLP-1 Intervention",
       subtitle = "Top 10 diseases by absolute difference in incidence",
       x = "Age (Years)", y = "Cumulative Cases") +
  theme(legend.position = "bottom", strip.text = element_text(face = "bold", size = 10))

ggsave("delfino_r_named_plots.png", width = 14, height = 12)