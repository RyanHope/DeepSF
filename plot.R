#setwd("C:/Users/Ryan/Dropbox/cmu-sf")
setwd("~/Dropbox/cmu-sf/autoturn-data")

require(data.table)
require(ggplot2)
require(grid)
require(gridExtra)
require(cowplot)
source("multiplot.R")

distance_reward <- function(x) {1-abs(max(min(x,180),40)-110)/70}

sfplots <- function(.folder) {
  d = rbindlist(lapply(list.files(path=.folder, pattern="*.tsv", full.names=TRUE), function(x) {.d=fread(x,header=T);if ("mean_eps" %in% names(x)) .d[,mean_eps:=NULL]; .d}))
  .k = as.numeric(d[,ceiling(max(episode)/50)])
  q = d[,.SD,.SDcols=c("id","episode","mean_q","maxscore","mean_absolute_error","outer_deaths","loss",
                       "inner_deaths","episode_reward","shell_deaths","raw_pnts","resets","total",
                       "fortress_kills","isi_pre","isi_post","reset_vlners")]
  qm = melt(q,id.vars=c("episode","id"))
  leftpad = unit(x=c(rel(1),rel(1),rel(1),rel(8)),units="mm")
  plots = list(
    p1 <- ggplot(qm[variable=="isi_pre"|variable=="isi_post"], aes(x=episode, y=value, color=factor(variable), group=variable)) + 
      geom_line(size=.5, alpha=.5) +
      facet_wrap(~id, nrow=1, scales="free_x") +
      geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
      geom_hline((aes(yintercept=7.5))) +
      ylab("Inter-shot interval") +
      xlab("Episode") +
      theme_bw() +
      theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position="top"),
    p2 <- ggplot(qm[variable=="outer_deaths"|variable=="inner_deaths"|variable=="shell_deaths"], aes(x=episode, y=value, color=factor(variable), group=variable)) + 
      geom_line(size=.5, alpha=.5) +
      facet_wrap(~id, nrow=1, scales="free_x") +
      geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
      ylab("Deaths") +
      xlab("Episode") +
      theme_bw() +
      theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position="top"),
    p3 <- ggplot(qm[variable=="total"|variable=="maxscore"][variable=="total",variable:="finalscore"][,], aes(x=episode, y=value, color=factor(variable), group=variable)) + 
      geom_line(size=.5, alpha=.5) +
      facet_wrap(~id, nrow=1, scales="free_x") +
      geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
      ylab("Score") +
      xlab("Episode") +
      theme_bw() +
      theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position="top"),
    p4 <- ggplot(qm[variable=="resets"], aes(x=episode, y=value)) + 
      geom_line(size=.5, alpha=.5) +
      facet_wrap(~id, nrow=1, scales="free_x") +
      geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
      ylab("Resets") +
      xlab("Episode") +
      theme_bw() +
      theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position="top"),
    p5 <- ggplot(qm[variable=="fortress_kills"], aes(x=episode, y=value)) + 
      geom_line(size=.5, alpha=.5) +
      facet_wrap(~id, nrow=1, scales="free_x") +
      geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
      ylab("Fortress Kills") +
      xlab("Episode") +
      theme_bw() +
      theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position="top"),
    p6 <- ggplot(qm[variable=="raw_pnts"], aes(x=episode, y=value)) + 
      geom_line(size=.5, alpha=.5) +
      facet_wrap(~id, nrow=1, scales="free_x") +
      geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
      ylab("Raw Points") +
      xlab("Episode") +
      theme_bw() +
      theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position="top"),
    p7 <- ggplot(qm[variable=="episode_reward"], aes(x=episode, y=value)) + 
      geom_line(size=.5, alpha=.5) +
      facet_wrap(~id, nrow=1, scales="free_x") +
      geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
      ylab("Reward") +
      xlab("Episode") +
      theme_bw() +
      theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position="top"),
    p8 <- ggplot(qm[variable=="reset_vlners"], aes(x=episode, y=value)) + 
      geom_line(size=.5, alpha=.5) +
      facet_wrap(~id, nrow=1, scales="free_x") +
      geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
      ylab("Vlner at Reset") +
      xlab("Episode") +
      theme_bw() +
      theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position="top"),
    p9 <- ggplot(qm[variable=="mean_q"], aes(x=episode, y=value)) + 
      geom_line(size=.5, alpha=.5) +
      facet_wrap(~id, nrow=1, scales="free_x") +
      geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
      ylab("Mean Q") +
      xlab("Episode") +
      theme_bw() +
      theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position="top")
  )
  p.1 <- plot_grid(p1, p2, p3, labels=c("A","B","C"), align="v", ncol=1, hjust=-.5)
  p.2 <- plot_grid(p8, p4, p5, p6, p7, p9, labels=c("D","E","F","G","H","I"), align="v", ncol=1, hjust=-.5)
  plot_grid(p.1, p.2, align="h")
}

while (T) {
  print(p <- sfplots("../deepsf-data/"))
  png("dqn-current.png", width = 1400, height = 900)
  print(p)
  dev.off()
  Sys.sleep(120)
}
