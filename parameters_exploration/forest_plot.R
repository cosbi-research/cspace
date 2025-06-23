rm(list=ls())
library(ggplot2)
library(forestplot)
library(dplyr)
library(plyr)
library(gridExtra)
graphics.off()
options(show.error.locations = TRUE)

#setwd("/home/tomasoni/git/infrastructure/source/word2vec/evaluation")
out.folder<-'plots'
useful.test.sets <- c('Mayo','sim','rel')
query.map=list(
#  "heart_failure_ids"="Heart Failure\n(81 K)",
#  "pp_heart_failure_ids"="Heart Failure\n(81 K)",
#  "pyp_heart_failure_ids"="Heart Failure\n(81 K)",
#  "liver_ids"="Liver\n(382 K)",
#  "pp_liver_ids"="Liver\n(382 K)",
#  "pyp_liver_ids"="Liver\n(382 K)",
  "20240826_random_sample_1"="Random Sample\n(81 K)",
  "20240826_random_sample_2"="Random Sample\n(382 K)",
#  "20230220_solr_ids"="PubMed/PMC\n(36 MLN)",
#  "20230220_ann_solr_ids"="PubMed/PMC\n(36 MLN)",
  "20230220_ann_pp_solr_ids"="PubMed/PMC 2023\n(36 MLN)",
  "20240826_ann_stop_solr_ids"="PubMed/PMC 2024\n(38 MLN)"
)

results<-read.delim(file.path('results.tsv'), na.strings = "")
fresults<-results[results$Algorithm=='bi-fasttext',]
fresults$Query <- sapply(fresults$Query, function(q) query.map[[q]])
fresults$Dataset <- sub('_lemmatized', '', sub('Scores', '', sub('UMNSRS_', '', sub('Terms','', sub('atedness','', sub('ilarity','', sub('_Terms','',fresults$Dataset)))))))
fresults$A <- gsub('^(.*)\n', '\n\\1\n', gsub('\n$', '', gsub('.([0-9]+)-([0-9]+).?', ' (\\1-\\2)\n', fresults$A)))
fresults$min.A <- gsub('^(.*)\n', '\n\\1\n', gsub('\n$', '', gsub('.([0-9]+)-([0-9]+).?', ' (\\1-\\2)\n', fresults$min.A)))

fresults$mean<- fresults$Max.perf
fresults$lower<- fresults$Max.perf - fresults$Max.perf.Std.Dev
fresults$upper<- fresults$Max.perf + fresults$Max.perf.Std.Dev

# dataset group
fresults$group <- as.factor(paste0(fresults$Algorithm, fresults$Query, fresults$A, fresults$Vector.Size,
                                   fresults$Window.Size, fresults$CBOW, fresults$Skip.Gram, fresults$Negative.Sampling,
                                   fresults$NS.Exponent, fresults$Max.N, fresults$Down.sample.Threshold, fresults$min.A))

# write CBOW to the negative sampling cell if CBOW=1
fresults[fresults$CBOW==1, 'Negative.Sampling'] <-'CBOW'

# remove unwanted test sets
fresults <- fresults[fresults$Dataset %in% useful.test.sets,]

fresults.mean <- ddply(fresults, .(group), summarize,
                       mean.Max.perf = mean(mean),
                       max.Max.perf.epoch = max(Max.perf.epoch))

#fresults$Mayo.Max.perf <- sapply(fresults$group, function(g){
#    print(fresults[fresults$group == g & fresults$Dataset=='Mayo','Max.Perf'] )
#  })

fresults <- merge(fresults, fresults.mean, by = 'group')

# reorder

fresults <- fresults[order(fresults$mean.Max.perf),]
fresults$group <- reorder(fresults$group, fresults$mean.Max.perf)
#row.names(fresults) <- seq(1,190)

# small version with representative experiments only
fresults.small <- do.call(rbind, lapply(query.map, function(q){
  # only best for each type
  #datasets<-c('Mayo','sim','rel')
  a <- fresults[fresults$Query == q, ]
  a <- a[order(a$mean.Max.perf,decreasing = T), ]
  head(a, n=3)
}))


t1 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=Query), vjust=0) +
  ggtitle("Corpus") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t1.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=Query), vjust=0) +
  ggtitle("Corpus") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

p1 <- ggplot(fresults, aes(y=group)) +
  geom_point(data=fresults[fresults$Dataset == 'Mayo', ], mapping=aes(x=mean), color="black", size = 2.5) +
  geom_point(data=fresults[fresults$Dataset == 'sim', ], mapping=aes(x=mean), color="blue", size = 2.5) +
  geom_point(data=fresults[fresults$Dataset == 'rel', ], mapping=aes(x=mean), color="red", size = 2.5) +
  #Add dot plot and error bars
  geom_errorbar(data=fresults[fresults$Dataset == 'Mayo', ], mapping=aes(xmin = lower, xmax = upper), width = 0.25, color="black") +
  geom_errorbar(data=fresults[fresults$Dataset == 'sim', ], mapping=aes(xmin = lower, xmax = upper), width = 0.25, color="blue") +
  geom_errorbar(data=fresults[fresults$Dataset == 'rel', ], mapping=aes(xmin = lower, xmax = upper), width = 0.25, color="red") +
  ggtitle("Max correlation") +
  #Add a line above graph
  #geom_hline(yintercept=4.6, size=2) +
  labs(x="", y = "") +
  scale_x_continuous(breaks=seq(0.1,1,0.1), limits=c(0.1,0.9), expand=c(0,0), position="bottom" ) +
  theme_gray(base_size=14) +
  #Remove legend
  #Also remove y-axis line and ticks
  theme(legend.position = "none",
        plot.title = element_text(hjust =0.5),
        axis.line.x = element_line(size = 0.6),
        axis.ticks.length=unit(0.3,"cm"),
        axis.text.y  = element_blank(),
        axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.y  = element_blank()
  )

p1.small <- ggplot(fresults.small, aes(y=group)) +
  geom_point(data=fresults.small[fresults.small$Dataset == 'Mayo', ], mapping=aes(x=mean), color="black", size = 2.5) +
  geom_point(data=fresults.small[fresults.small$Dataset == 'sim', ], mapping=aes(x=mean), color="blue", size = 2.5) +
  geom_point(data=fresults.small[fresults.small$Dataset == 'rel', ], mapping=aes(x=mean), color="red", size = 2.5) +
  #Add dot plot and error bars
  geom_errorbar(data=fresults.small[fresults.small$Dataset == 'Mayo', ], mapping=aes(xmin = lower, xmax = upper), width = 0.25, color="black") +
  geom_errorbar(data=fresults.small[fresults.small$Dataset == 'sim', ], mapping=aes(xmin = lower, xmax = upper), width = 0.25, color="blue") +
  geom_errorbar(data=fresults.small[fresults.small$Dataset == 'rel', ], mapping=aes(xmin = lower, xmax = upper), width = 0.25, color="red") +
  ggtitle("Max correlation") +
  #Add a line above graph
  #geom_hline(yintercept=4.6, size=2) +
  labs(x="", y = "") +
  scale_x_continuous(breaks=seq(0.1,1,0.1), limits=c(0.1,0.9), expand=c(0,0), position="bottom" ) +
  theme_gray(base_size=14) +
  #Remove legend
  #Also remove y-axis line and ticks
  theme(legend.position = "none",
        plot.title = element_text(hjust =0.5),
        axis.line.x = element_line(size = 0.6),
        axis.ticks.length=unit(0.3,"cm"),
        axis.text.y  = element_blank(),
        axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.y  = element_blank()
  )

t2 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=max.Max.perf.epoch), vjust=0) +
  ggtitle("Epoch") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t2.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=max.Max.perf.epoch), vjust=0) +
  ggtitle("Epoch") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t3 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=A), vjust=0) +
  ggtitle("A") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t3.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=A), vjust=0) +
  ggtitle("A") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t4 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=Vector.Size), vjust=0) +
  ggtitle("Vector Size") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t4.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=Vector.Size), vjust=0) +
  ggtitle("Vector Size") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t5 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=Window.Size), vjust=0) +
  ggtitle("Window Size") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t5.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=Window.Size), vjust=0) +
  ggtitle("Window Size") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t6 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=Negative.Sampling), vjust=0) +
  ggtitle("Neg. Sampling") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t6.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=Negative.Sampling), vjust=0) +
  ggtitle("Neg. Sampling") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t7 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=NS.Exponent), vjust=0) +
  ggtitle("Neg. Exponent") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t7.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=NS.Exponent), vjust=0) +
  ggtitle("Neg. Exponent") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t8 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=Max.N), vjust=0) +
  ggtitle("Max Chars") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t8.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=Max.N), vjust=0) +
  ggtitle("Max Chars") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t9 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=Down.sample.Threshold), vjust=0) +
  ggtitle("Down-sample") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t9.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=Down.sample.Threshold), vjust=0) +
  ggtitle("Down-sample") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t10 <- ggplot(data=fresults) +
  geom_text(aes(y=group, x=1, label=min.A), vjust=0) +
  ggtitle("Min A") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )

t10.small <- ggplot(data=fresults.small) +
  geom_text(aes(y=group, x=1, label=min.A), vjust=0) +
  ggtitle("Min A") +
  xlab("  ") +
  theme_classic(base_size=14) +
  theme(
    axis.line.y = element_blank(),
    axis.line.x = element_line(color = "white"),
    axis.text.y  = element_blank(),
    axis.ticks.y  = element_blank(),
    axis.ticks.x = element_line(color = "white"),
    axis.ticks.length=unit(0.3,"cm"),
    axis.title.y  = element_blank(),
    axis.text.x = element_text(color="white"),
    plot.title = element_text(hjust =0.5)
  )
#png(file.path(out.folder, paste(paste0("forest_plot.png"), sep = "")), width = 1700, height = 2000)
#svg(file.path(out.folder, paste(paste0("forest_plot.svg"), sep = "")), width = 1700, height = 2000)
#p<-grid.arrange(t1, p1, t2, t3, t4, t5, t6, t7, t8, t9, t10, widths=c(0.4,4, 0.4,0.4,0.5,0.6, 0.6, 0.7, 0.6, 0.6, 0.4))
#dev.off()
#ggsave(file=file.path(out.folder, paste(paste0("forest_plot.svg"), sep = "")), plot=p, width=37, height=85, limitsize=F)

#p.small<-grid.arrange(t1.small, p1.small, t2.small, t3.small, t4.small, t5.small, t6.small, t7.small, t8.small, t9.small, t10.small, widths=c(0.4,4, 0.4,0.4,0.5,0.6, 0.6, 0.7, 0.6, 0.6, 0.4))
#dev.off()
#ggsave(file=file.path(out.folder, paste(paste0("forest_plot small.svg"), sep = "")), plot=p.small, width=29, height=6, limitsize=F)

p.paper<-grid.arrange(t1.small, p1.small, widths=c(1,4))
#dev.off()
ggsave(file=file.path(out.folder, paste(paste0("Figure 3.png"), sep = "")), plot=p.paper, width=10, height=4, limitsize=F, dpi=1200)


