setwd("~/Dropbox/cmu-sf/autoturn-data")

require(data.table)
require(ggplot2)
require(cowplot)
require(compoisson)

subject5num <<- c(0,1480,1995,2335,2952)

sfplot <- function(d, vars, .ylab=NULL, .grouped=FALSE, .SE=FALSE, .k=5, .thresholds=NULL, .fivenum=NULL, .mean=NULL) {
  leftpad <- unit(x=c(rel(1),rel(1),rel(1),rel(8)),units="mm")
  .doColor = length(unique(d[,variable])) > 1
  if (.grouped) {
    if (.doColor)
      p <- ggplot(d[variable %in% vars], aes(x=episode, y=value, color=factor(variable), group=variable)) + 
        facet_wrap(~id, nrow=1, scales="free_x")
    else
      p <- ggplot(d[variable %in% vars], aes(x=episode, y=value, group=variable), color="black") + 
        facet_wrap(~id, nrow=1, scales="free_x")
  } else
    p <- ggplot(d[variable %in% vars], aes(x=episode, y=value)) + 
      facet_grid(variable~id, scales="free")
  if (!is.null(.thresholds))
    p <- p + geom_hline((aes(yintercept=threshold)), data=.thresholds, color="magenta", alpha=.5)
  if (!is.null(.fivenum))
    p <- p + 
      geom_hline((aes(yintercept=.fivenum[1])), color="magenta", alpha=.5, linetype="dotted") +
      geom_hline((aes(yintercept=.fivenum[2])), color="magenta", alpha=.5, linetype="dashed") +
      geom_hline((aes(yintercept=.fivenum[3])), color="magenta", alpha=.5) +
      geom_hline((aes(yintercept=.fivenum[4])), color="magenta", alpha=.5, linetype="dashed") +
      geom_hline((aes(yintercept=.fivenum[5])), color="magenta", alpha=.5, linetype="dotted")
  
  if (!is.null(.mean))
    p <- p + geom_line(aes(x=episode, y=value, group=variable), size=.5, color="gray", data=d[variable==.mean])
  if (!.SE)
    p <- p + geom_line(size=.5, alpha=.5)    
  p <- p + geom_smooth(se=.SE, method = "gam", formula = y ~ s(x, k = as.numeric(d[,ceiling(max(episode)/25)])), size=.5) +
    xlab("Episode") +
    theme_bw()
  if (!is.null(.ylab))
    p <- p + ylab(.ylab)
  else
    p <- p + theme(axis.title.y=element_blank())
  if (.grouped) {
    p <- p + theme(plot.margin=leftpad) +
      theme(legend.position=c(.01,.99),
            legend.justification=c(0,1),
            legend.title=element_blank(),
            legend.text=element_text(size=7),
            legend.key.size=unit(.5,"line"),
            legend.background = element_rect(fill=alpha('black', 0.1)))
    if (.doColor)
      p <- p + scale_color_brewer(palette="Set1")
  }
  p
}

sfplots <- function(.folder) {
  d <- rbindlist(lapply(list.files(path=.folder, pattern="*.tsv", full.names=TRUE), function(x) {
    .d <- fread(x,header=T)
    if ("mean_eps" %in% names(x))
      .d[,mean_eps:=NULL]
    .d[,finalscore:=total]
    if ("action_thrustshoot" %in% names(.d))
      .d[,c("action_noop_p","action_thrust_p","action_shoot_p","action_thrustshoot_p"):=.(action_noop/5400,action_thrust/5400,action_shoot/5400,action_thrustshoot/5400)]
    else
      .d[,c("action_noop_p","action_thrust_p","action_shoot_p"):=.(action_noop/5400,action_thrust/5400,action_shoot/5400)]
    .d#[episode>10]
  }))
  sdcols = c("id","episode","mean_q","maxscore","mean_absolute_error","outer_deaths","loss",
             "inner_deaths","episode_reward","shell_deaths","raw_pnts","resets","finalscore","max_vlner",
             "fortress_kills","isi_pre","isi_post","reset_vlners","reward_mean","thrust_durations","shoot_durations",
             "action_noop","action_thrust","action_shoot","action_noop_p","action_thrust_p","action_shoot_p","kill_vlners")
  if ("action_thrustshoot" %in% names(d))
    sdcols = c(sdcols, "action_thrustshoot", "action_thrustshoot_p")
  q <- d[,.SD,.SDcols=sdcols]
  q[,total_deaths:=outer_deaths+inner_deaths+shell_deaths]
  qm <- melt(q,id.vars=c("episode","id"))
  qm[,episode:=as.integer(episode)]
  
  p1 <- sfplot(qm, c("isi_pre","isi_post"), .ylab="Inter-shot interval", .grouped=TRUE, .thresholds=data.table(variable="isi_pre",threshold=7.5))
  p2 <- sfplot(qm, c("outer_deaths","inner_deaths","shell_deaths"), .ylab="Deaths", .grouped=TRUE, .mean="total_deaths")
  p3 <- sfplot(qm, c("finalscore","maxscore"), .ylab="Score", .grouped=TRUE, .fivenum=subject5num)
  p4 <- sfplot(qm, c("thrust_durations","shoot_durations"), .ylab="Mean Durations", .grouped=TRUE)
  if ("action_thrustshoot" %in% names(d)) {
    p5 <- sfplot(qm, c("action_noop_p","action_thrust_p","action_shoot_p","action_thrustshoot_p"), .ylab="Action Proportion", .grouped=TRUE)
  } else {
    p5 <- sfplot(qm, c("action_noop_p","action_thrust_p","action_shoot_p"), .ylab="Action Proportion", .grouped=TRUE)
  }
  p6 <- sfplot(qm, c("max_vlner","kill_vlners"), .ylab="Fortress Vlner", .grouped=TRUE, .thresholds=data.table(variable="kill_vlners",threshold=10))
  p7.1 <- sfplot(qm, c("mean_q"), .grouped=TRUE)
  p7.2 <- sfplot(qm, c("mean_absolute_error"), .grouped=TRUE)
  p7.3 <- sfplot(qm, c("loss"), .grouped=TRUE)
  p8.1 <- sfplot(qm, c("episode_reward"), .grouped=TRUE)
  p8.2 <- sfplot(qm, c("fortress_kills"), .grouped=TRUE)
  p8.3 <- sfplot(qm, c("resets"), .grouped=TRUE)
  p.1 <- plot_grid(p1, p2, p3, p4, p6, labels=c("A","B","C","D","E"), align="v", ncol=1, hjust=-.5)
  p.2 <- plot_grid(p5, p7.1, p7.2, p7.3, p8.1, p8.2, p8.3, labels=c("F","G","H","I","J","K","L"), align="v", ncol=1, hjust=-.5, rel_heights=c(1.5,1,1,1,1,1,1))
  plot_grid(p.1, p.2, align="h", ncol=2)
}

while (T) {
  .folder <- "../deepsf-data/"
  print(p <- sfplots(.folder))
  Sys.sleep(120)
}
