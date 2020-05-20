source("dirs.R")
if (!exists(file.path(aux.dir, "mylib"))) source(file.path(aux.dir, "mylib.R"))
mylib(c("data.table", "magrittr"))
