setwd("~/Dropbox/cmu-sf/autoturn-data")

require(data.table)
require(ggplot2)
require(cowplot)
require(compoisson)

subject5num <<- c(0,1480,1995,2335,2952)

sfplot <- function(d, vars, .ylab=NULL, .grouped=FALSE, .SE=FALSE, .k=5, .thresholds=NULL, .fivenum=NULL, .mean=NULL) {
  leftpad <- unit(x=c(rel(1),rel(1),rel(1),rel(8)),units="mm")
  if (.grouped)
    p <- ggplot(d[variable %in% vars], aes(x=episode, y=value, color=factor(variable), group=variable)) + 
      facet_wrap(~id, nrow=1, scales="free_x")
  else
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
    if ("action_thrustshoot" %in% names(.d))
      .d[,c("action_noop_p","action_thrust_p","action_shoot_p","action_thrustshoot_p"):=.(action_noop/5400,action_thrust/5400,action_shoot/5400,action_thrustshoot/5400)]
    else
      .d[,c("action_noop_p","action_thrust_p","action_shoot_p"):=.(action_noop/5400,action_thrust/5400,action_shoot/5400)]
    .d#[episode>10]
  }))
  .k <- as.numeric(d[,ceiling(max(episode)/25)])
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
  
  p1 <- sfplot(qm, c("isi_pre","isi_post"), .ylab="Inter-shot interval", .grouped=TRUE, .k=.k, .thresholds=data.table(variable="isi_pre",threshold=7.5))
  p2 <- sfplot(qm, c("outer_deaths","inner_deaths","shell_deaths"), .ylab="Deaths", .grouped=TRUE, .k=.k, .mean="total_deaths")
  p3 <- sfplot(qm, c("finalscore","maxscore"), .ylab="Score", .grouped=TRUE, .k=.k, .fivenum=subject5num)
  p4 <- sfplot(qm, c("thrust_durations","shoot_durations"), .ylab="Mean Durations", .grouped=TRUE, .k=.k)
  if ("action_thrustshoot" %in% names(d)) {
    p5 <- sfplot(qm, c("action_noop_p","action_thrust_p","action_shoot_p","action_thrustshoot_p"), .ylab="Action Proportion", .grouped=TRUE, .k=.k)
  } else {
    p5 <- sfplot(qm, c("action_noop_p","action_thrust_p","action_shoot_p"), .ylab="Action Proportion", .grouped=TRUE, .k=.k)
  }
  p6 <- sfplot(qm, c("max_vlner","kill_vlners"), .ylab="Fortress Vlner", .grouped=TRUE, .k=.k, .thresholds=data.table(variable="kill_vlners",threshold=10))
  p7 <- sfplot(qm, c("mean_q","mean_absolute_error","loss"), .k=.k)
  p8 <- sfplot(qm, c("episode_reward","resets","fortress_kills"), .k=.k)
  p.1 <- plot_grid(p1, p2, p3, labels=c("A","B","C"), align="v", ncol=1, hjust=-.5)
  p.2 <- plot_grid(p4, p5, p6, labels=c("D","E","F"), align="v", ncol=1, hjust=-.5)
  p.3 <- plot_grid(p.1, p.2, align="h", ncol=2, hjust=-.5)
  p.4 <- plot_grid(p7, p8, labels=c("G","H"), align="h", ncol=2, hjust=-.5)
  plot_grid(p.3, p.4, align="v", nrow=2, rel_heights=c(3, 1))
}

epsu <- function(n, eps) {
  runs <- array(0, n)
  for (i in 1:n) {
    .eps <- eps
    while (runif(1) > .eps) {
      runs[i] <- runs[i] + 1
      .eps <- .eps + .eps * (1+.eps)
    }
  }
  runs
}

while (F) {
  dp1 = rbindlist(list(
    data.table(runs=rle(runif(10000)<=.1)$lengths, type="unif", param=".1"),
    data.table(runs=rle(runif(10000)<=.001)$lengths, type="unif", param=".001"),
    data.table(runs=rle(runif(10000)<=.000001)$lengths, type="unif", param=".000001")
  ))
  dp2 = data.table()
  for (lam in c(10,20,30)) {
    .dp2 = data.table(runs=rpois(1,lam), type="pois", param=as.character(lam))
    while (.dp2[, sum(runs)]<10000)
      .dp2 = rbind(.dp2, data.table(runs=rpois(1,lam), type="pois", param=as.character(lam)))
    dp2 = rbind(dp2, .dp2)
  }
  dp3 = data.table()
  for (eps in c(.1,.001,.000001)) {
    .dp3 = data.table(runs=epsu(1,eps), type="eps", param=as.character(eps))
    while (.dp3[, sum(runs)]<10000)
      .dp3 = rbind(.dp3, data.table(runs=epsu(1,eps), type="eps", param=as.character(eps)))
    dp3 = rbind(dp3, .dp3)
  }
  dp = rbind(dp1, dp2, dp3)
  ggplot(dp) + 
    geom_histogram(aes(x=runs,group=param,fill=param),position="dodge",binwidth=1) + 
    facet_wrap(~type) +
    theme_bw() + 
    theme(legend.position=c(.95,.95),
          legend.justification=c(1,1))
  
  
  good_subjects = c("sfsa08","sfsa10","sfsa16","sfsa20","sfsa28","sfsa30","sfsa43","sfsa47")
  bad_uids = c("mturk-b2_A37EV8RZ82WT8E")
  d2 = readRDS("ALLDATA_cmuStudents+distanceNoevals+impactNoevals+mturk-b2.rds")
  d2[,uid:=paste(group,sid,sep="_")]
  
}

rpoisn <- function(n, k, s) {
  lam = c()
  while (length(lam)<n)
    lam = c(lam, rpois(1, round(rnorm(1, k, s))))
  lam
}

while (T) {
  .folder <- "../deepsf-data/"
  print(p <- sfplots(.folder))
  Sys.sleep(120)
}
