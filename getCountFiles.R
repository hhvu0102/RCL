#!/usr/bin/env Rscript
# Arguments
#	input file
#	region length
#	threads
#	output file

args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)<=3) {
	stop("At least four arguments must be supplied (input file).", call.=FALSE)
}

#full new
process3_1 <- function(sub, s) {
	sub <- dplyr::distinct(sub[,1:3])
	sub$mid <- round((sub$end - sub$start)/2) + sub$start
	sub$start <- sub$mid - inputLength/2
	sub$end <- sub$mid + inputLength/2
	# attempt to fix regions at ends, but do we know right end of reference?
	#if (sub$start < 0) {
	#	sub$end <- sub$end - sub$start
	#	sub$start <- 0
	#}
	sub <- sub[,1:3]
	sub$name <- s
	colnames(sub) <- c("chr", "start", "end", "name")

	return(sub)
}

getSegments <- function(s) {
	sub <- file[file$name == s,]

	if (sub$end[1] - sub$start[1] < inputLength) {
		sub <- process3_1(sub, s)
	} else {
		sub1 <- sub[which(sub$counts >= quantile(sub$counts, 0.95)),]
		#sub1c <- sub1
		sub1 <- sub1[,c("chr.c", "start.c", "end.c")]
		colnames(sub1) <- c("chr", "start", "end")

		if (nrow(sub1) > 1) {
			sub1.sort <- bedr.sort.region(sub1, check.chr = FALSE)
			sub1.merge <- bedr.merge.region(sub1.sort, distance = inputLength, verbose = T, check.chr = FALSE)
			sub1.merge$mid <- round((sub1.merge$end - sub1.merge$start)/2) + sub1.merge$start
			sub1.merge$start <- sub1.merge$mid - inputLength/2
			sub1.merge$end <- sub1.merge$mid + inputLength/2

			for (i in 1:nrow(sub1.merge)) {
				sub1.merge$name[i] <- paste0(s, "_", i)
			}
			sub <- sub1.merge[,c("chr", "start", "end", "name")]
		} else {
			sub$mid <- round((sub$end - sub$start)/2) + sub$start
			sub$start <- sub$mid - inputLength/2
			sub$end <- sub$mid + inputLength/2
			sub <- sub[,1:4]
			colnames(sub) <- c("chr", "start", "end", "name")
		}
	}

	return(sub)
}



suppressPackageStartupMessages(library(bedr))
suppressPackageStartupMessages(library(doParallel))

#print(args[1])

if (file.size(args[1]) > 0) {
	file <- read.table(args[1], header = F)
	colnames(file) <- c("chr", "start", "end", "name", "chr.c", "start.c", "end.c", "counts")

	add.chr <- F
	if (length(grep("chr", file$chr[1])) == 0) {
		#print("Adding `chr` to chromosome name")
		file$chr <- paste0("chr", file$chr)
		file$chr.c <- paste0("chr", file$chr.c)
		add.chr <- T
	}
	file$name <- paste0(file$chr, file$name)

	inputLength <- as.numeric(args[2])

	no_cores <- args[3]
	cl <- makeCluster(no_cores, type = "FORK")
	registerDoParallel(cl)
	new <- data.frame(matrix(ncol = 4))
	colnames(new) <- c("chr", "start", "end", "name")

	result <- foreach(i=unique(file$name)) %dopar% {getSegments(i)}
	#save(result, file=args[4])

	for (i in 1:length(result)) {
		if (nrow(result[[i]]) > 0)
			new <- rbind(new, result[[i]])
	}
	new <- new[!is.na(new$chr),]
	if (add.chr) {
		new$chr <- sub("chr", "", new$chr)
		new$name <- sub("chr", "", new$name)
	}
	write.table(new, file=args[4], quote = F, sep = "\t", row.names = F)
} else {
	print(paste("Input", args[1], "has 0 line."))
}

