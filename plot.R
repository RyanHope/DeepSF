#setwd("C:/Users/Ryan/Dropbox/cmu-sf")
setwd("C:/Users/Ryan/Dropbox/cmu-sf/autoturn-data")

require(data.table)
require(ggplot2)

distance_reward <- function(x) {1-abs(max(min(x,180),40)-110)/70}

while (T) {
  d = fread("dqn-elu_log.tsv", header=T)
  q = d[,.SD,.SDcols=c("episode","mean_q","maxscore","mean_absolute_error","outer_deaths","loss",
                       "inner_deaths","episode_reward","shell_deaths","raw_pnts","resets","total",
                       "fortress_kills","isi_pre","isi_post","reset_vlners")]
  qm = melt(q,id.vars=c("episode"))
  qm[,grp:=ifelse(variable=="isi_pre",2,ifelse(variable=="isi_post",3,1))]
  qm[variable %in% c("isi_pre","isi_post"), variable:="isi"]
  qm[variable=="isi" & value>25, value:=Inf]
  qm[,grp:=factor(grp,levels=c(1,2,3),labels=c("vlner == *","vlner < 10","vlner >= 10"),ordered=T)]
  p = ggplot(qm, aes(x=episode, y=value, group=grp, color=factor(grp))) + 
    geom_line(size=.5, alpha=.5) +
    facet_wrap(~variable, ncol=2, scales="free_y") +
    geom_smooth(method = "gam", formula = y ~ s(x, k = 5), se=FALSE) +
    ylab("Value") +
    xlab("Episode") +
    theme_bw() +
    scale_color_manual(values=c("black","#998ec3","#f1a340"))
  print(p)
  Sys.sleep(60)
}


while (T) {
  d = rbindlist(lapply(list.files(path="../deepsf-data/", pattern="*.tsv", full.names=TRUE), fread, header=T))
  .k = as.numeric(d[,ceiling(max(episode)/50)])
  q = d[,.SD,.SDcols=c("id","episode","mean_q","maxscore","mean_absolute_error","outer_deaths","loss",
                       "inner_deaths","episode_reward","shell_deaths","raw_pnts","resets","total",
                       "fortress_kills","isi_pre","isi_post","reset_vlners")]
  qm = melt(q,id.vars=c("episode","id"))
  qm[(variable=="isi_pre"|variable=="isi_post")&value>15, value:=Inf]
  p = ggplot(qm, aes(x=episode, y=value, group=id, color=factor(id))) + 
    geom_line(size=.5, alpha=.5) +
    facet_wrap(~variable, ncol=2, scales="free_y") +
    geom_smooth(se=FALSE, method = "gam", formula = y ~ s(x, k = .k)) +
    ylab("Value") +
    xlab("Episode") +
    theme_bw() +
    scale_color_brewer(palette="Set1")
  print(p)
  Sys.sleep(120)
}
