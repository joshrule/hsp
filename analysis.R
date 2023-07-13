library(tidyverse)
library(tidyboot)
library(patchwork)

filename_regex <- "^([^_]+)_(\\d+)_(\\d+)_(\\d+)\\.csv$"

data <- list.files(path = "hsp/out", pattern = "*.csv", full.names = T) %>%
  map(
    ~read_csv(.x, comment = "#") %>%
      mutate(file = basename(.x))
  ) %>%
  list_rbind() %>%
  mutate(
    task = str_replace(file, filename_regex, "\\1"),
    increment = as.numeric(str_replace(file, filename_regex, "\\2")),
    epochs = as.numeric(str_replace(file, filename_regex, "\\3")),
    run = as.numeric(str_replace(file, filename_regex, "\\4")),
    max_reward = ceiling(100 / (increment + 1)) * 50,
    task = factor(
      task,
      levels = c("counter", "countergrid"),
      labels = c("Counter", "Counter + Grid"),
    )
  )

mean_data <- data %>%
  group_by(epoch, task, increment, max_reward) %>%
  tidyboot(
    column = total_reward,
    summary_function = median,
    nboot = 1000,
    statistics_functions = list("ci_lower" = ci_lower, "ci_upper" = ci_upper)
  )

ggplot() +
  geom_line(
    data = data,
    aes(x = epoch, y = total_reward / max_reward, group = run),
    color = "#999999",
    alpha = 0.5,
    linewidth = 0.5,
  ) +
  geom_line(
    data = mean_data,
    aes(x = epoch, y = empirical_median / max_reward),
    color = "#e6550d",
    alpha = 1,
    linewidth = 1,
  ) +
  facet_grid(task~increment) +
  labs(
   x = "Epoch",
   y = "Reward (% of Maximum)"
  ) +
  theme_bw(base_size = 16) +
  theme(
    strip.background = element_blank()
  )
