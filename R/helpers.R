sort_hidden_act <- function(data, i) {
  cleaned <- data %>%
    rename(speaker=...1) %>% 
    pivot_longer(cols = -c("speaker"), names_to = 'vowel', values_to = 'kld', values_drop_na = T) %>% 
    mutate(layer = i)
  
  return(cleaned)
}

by_vowel_plot <- function(results, mean_vowel_results) {
  results %>% 
  ggplot(aes(x=vowel, y=kld)) +
  geom_point(size=1, alpha=0.4) +
  geom_hline(yintercept = 0.5, color="red", size=1, linetype="dashed") +
  ylab("Kullback–Leibler divergence") +
  geom_errorbar(
    data = mean_vowel_results,
    aes(ymin = kld - se, ymax = kld + se, y = kld),
    color = "red", 
  ) +
  geom_point(
    data = mean_vowel_results,
    aes(y = kld),
    color = "red", # color for the mean points
    size = 1
  ) +
  scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2)
    ) +
  theme(
    legend.position = "right", 
    axis.text.x = element_text(hjust=1, size = 10),
    axis.text.y = element_text(size = 10),
    axis.title.x = element_text(size = 20, face = "bold"),
    axis.title.y = element_text(size = 20, face = "bold"),
    legend.text = element_text (size = 10),
    legend.title = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 20),
    strip.text.x = element_text(size= 20, face = "bold"),
    strip.text.y = element_text(size= 20, face = "bold"),
    panel.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "grey", size = 0.5),
    panel.grid.minor = element_line(color = "lightgrey", size = 0.25))
  
}

by_speaker_plot <- function(results, mean_speaker_results) {
  results %>% 
  mutate(speaker=as.character(speaker)) %>% 
    ggplot(aes(x=speaker, y=kld)) +
    geom_point(size=1, alpha=0.4) +
    geom_hline(yintercept = 0.5, color="red", size=1, linetype="dashed") +
    ylab("Kullback–Leibler divergence") +
    geom_errorbar(
      data = mean_speaker_results,
      aes(ymin = kld - se, ymax = kld + se, y = kld),
      color = "blue", 
    ) +
    geom_point(
      data = mean_speaker_results,
      aes(y = kld),
      color = "blue", # color for the mean points
      size = 1
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2)
    ) +
    theme(
          legend.position = "right", 
          axis.text.x = element_text(hjust=1, size = 10, angle=90),
          axis.text.y = element_text(size = 10),
          axis.title.x = element_text(size = 20, face = "bold"),
          axis.title.y = element_text(size = 20, face = "bold"),
          legend.text = element_text (size = 10),
          legend.title = element_text(size = 10, face = "bold"),
          plot.title = element_text(size = 20),
          strip.text.x = element_text(size= 20, face = "bold"),
          strip.text.y = element_text(size= 20, face = "bold"),
          panel.background = element_rect(fill = "white"),
          panel.grid.major = element_line(color = "grey", size = 0.5),
          panel.grid.minor = element_line(color = "lightgrey", size = 0.25))
}

get_summary_vowels <- function(results) {
  summary <- results %>% 
  group_by(vowel) %>% 
    summarise(se= sd(kld)/sqrt(num_speaker), kld = mean(kld), .groups = "drop") 
  
  return(summary)
}


get_summary_speakers <- function(results) {
  summary <- results %>% 
    group_by(speaker) %>% 
    summarise(se= sd(kld)/sqrt(num_speaker), kld = mean(kld), .groups = "drop") %>% 
    mutate(speaker=as.character(speaker))
  
  return(summary)
}

