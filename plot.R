setwd("~/Dropbox/cmu-sf/autoturn-data")

require(data.table)
require(ggplot2)
require(cowplot)

sfplot <- function(d, vars, .ylab=NULL, .grouped=FALSE, .SE=FALSE, .k=5, .thresholds=NULL) {
  leftpad <- unit(x=c(rel(1),rel(1),rel(1),rel(8)),units="mm")
  if (.grouped)
    p <- ggplot(d[variable %in% vars], aes(x=episode, y=value, color=factor(variable), group=variable)) + 
      facet_wrap(~id, nrow=1, scales="free_x")
  else
    p <- ggplot(d[variable %in% vars], aes(x=episode, y=value)) + 
      facet_grid(variable~id, scales="free")
  if (!is.null(.thresholds))
    p <- p + geom_hline((aes(yintercept=threshold)), data=.thresholds, color="magenta", alpha=.5)
  if (!.SE)
    p <- p + geom_line(size=.5, alpha=.5)    
  p <- p + geom_smooth(se=.SE, method = "gam", formula = y ~ s(x, k = .k), size=.5) +
    xlab("Episode") +
    theme_bw()
  if (!is.null(.ylab))
    p <- p + ylab(.ylab)
  else
    p <- p + theme(axis.title.y=element_blank())
  if (.grouped)
    p <- p + theme(plot.margin=leftpad) +
      scale_color_brewer(palette="Set1") +
      theme(legend.position=c(.01,.99),
            legend.justification=c(0,1),
            legend.title=element_blank(),
            legend.text=element_text(size=7),
            legend.key.size=unit(.5,"line"),
            legend.background = element_rect(fill=alpha('black', 0.1)))
  p
}

sfplots <- function(.folder) {
  d <- rbindlist(lapply(list.files(path=.folder, pattern="*.tsv", full.names=TRUE), function(x) {
    .d <- fread(x,header=T)
    if ("mean_eps" %in% names(x))
      .d[,mean_eps:=NULL]
    .d[,finalscore:=total]
    .d[,c("action_noop_p","action_thrust_p","action_shoot_p"):=.(action_noop/5400,action_thrust/5400,action_shoot/5400)]
    .d
  }))
  .k <- as.numeric(d[,ceiling(max(episode)/25)])
  q <- d[,.SD,.SDcols=c("id","episode","mean_q","maxscore","mean_absolute_error","outer_deaths","loss",
                       "inner_deaths","episode_reward","shell_deaths","raw_pnts","resets","finalscore",
                       "fortress_kills","isi_pre","isi_post","reset_vlners","reward_mean","thrust_durations","shoot_durations",
                       "action_noop","action_thrust","action_shoot","action_noop_p","action_thrust_p","action_shoot_p","kill_vlners")]
  qm <- melt(q,id.vars=c("episode","id"))
  qm[,episode:=as.integer(episode)]
  
  p1 <- sfplot(qm, c("isi_pre","isi_post"), .ylab="Inter-shot interval", .grouped=TRUE, .k=.k, .thresholds=data.table(variable="isi_pre",threshold=7.5))
  p2 <- sfplot(qm, c("outer_deaths","inner_deaths","shell_deaths"), .ylab="Deaths", .grouped=TRUE, .k=.k)
  p3 <- sfplot(qm, c("finalscore","maxscore"), .ylab="Score", .grouped=TRUE, .k=.k)
  p4 <- sfplot(qm, c("thrust_durations","shoot_durations"), .ylab="Mean Durations", .grouped=TRUE, .k=.k)
  p5 <- sfplot(qm, c("action_noop_p","action_thrust_p","action_shoot_p"), .ylab="Action Proportion", .grouped=TRUE, .k=.k)
  p6 <- sfplot(qm, c("mean_q","episode_reward","mean_absolute_error","loss"), .k=.k)
  p7 <- sfplot(qm, c("resets","fortress_kills","kill_vlners","reset_vlners"), .k=.k, .thresholds=data.table(variable=c("reset_vlners","kill_vlners"),threshold=c(10,10)))
  p.1 <- plot_grid(p1, p2, p3, p4, p5, labels=c("A","B","C","D","E"), align="v", ncol=1, hjust=-.5)
  p.2 <- plot_grid(p6, p7, labels=c("E","F"), align="v", ncol=1, hjust=-.5)
  plot_grid(p.1, p.2, align="h")
}

while (T) {
  .folder <- "../deepsf-data/"
  print(p <- sfplots(.folder))
  Sys.sleep(120)
}
