# config directory paths and other options
# [wd] working directory
# [src] source directory
# [aux.dir] auxiliary directory
# [data.root] where data should be 
# [data.dir] data read-in directory

wd <- "."
src <- ifelse(grepl("[Hh]elen", Sys.info()["nodename"]), wd, "src")
aux.dir <- "../auxiliary"
data.root <- file.path("../../data", basename(getwd()))
data.dir <- file.path(data.root, "data")

stopifnot(dir.exists(src) & dir.exists(aux.dir))

options(scipen=999, digits = 9, stringsAsFactors = F)

create.proj(research=T,training=T,production=F,portable=F)
file.edit("start.template.R")
