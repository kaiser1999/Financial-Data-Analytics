#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
letter2number <- function(x) {utf8ToInt(x) - utf8ToInt("a") + 1L}

################################################################################
# update color
# if there are two of the same letter in guess and only one of them turns yellow or green,
# then there is only one copy of that letter in the answer
# e.g. answer is "STUDY" and guess is "STARS", then color is "Green" "Gray" "Gray" "Gray" "Gray"
# therefore, answer_tab stores the count for each letter except green, and subtract 1 from its count if yellow
color_update <- function(guess, answer){
  word_len <- length(answer)
  color <- rep("Gray", word_len)
  answer_tab <- table(answer[answer != guess])  # m - m_0
  for (j in 1:word_len){
    if (guess[j] == answer[j]){
      color[j] = "Green"
    }else if((guess[j] %in% answer)){
      w_j <- as.character(guess[j])
      if (! w_j %in% names(answer_tab)) next
      if (answer_tab[w_j] > 0){
        color[j] <- "Yellow"
        answer_tab[w_j] <- answer_tab[w_j] - 1
      }
    }
  }
  return(color)
}

################################################################################
# remove all words in bank that 
# Green: has no letter w_j in position j
# Yellow: has less number of w_j than green+yellow in guess
# Gray: if no green+yellow in w_j, then remove words that contain w_j (see update color)
#       if there are n of them, then remove words that contain w_j more than n
word_bank_update <- function(word_bank_k, color, guess){
  word_len <- length(color)
  color_factor <- factor(color, levels=c("Green", "Yellow", "Gray"))
  # A word_len x 3 (colors) table, use factor to ensure all three colors appear in table
  guess_color_tab <- table(guess, color_factor)
  for (j in 1:word_len){
    if (color[j] == "Green"){ # Green
      word_bank_k <- word_bank_k[word_bank_k[,j] == guess[j],]
    }else if (color[j] == "Yellow"){ # Yellow
      colored_count <- apply(word_bank_k, 1, function(w) sum(w %in% guess[j]))
      w_j <- as.character(guess[j])
      word_bank_k <- word_bank_k[(colored_count >= sum(guess_color_tab[w_j, c("Green", "Yellow")])) & (word_bank_k[,j] != guess[j]),]
    }else{ # Gray
      w_j <- as.character(guess[j])
      w_j_colored_count <- sum(guess_color_tab[w_j, c("Green", "Yellow")])
      if (w_j_colored_count == 0){ # w_j neither yellow nor green, i.e. not in answer
        word_bank_k <- word_bank_k[!apply(word_bank_k, 1, function(w) guess[j] %in% w),]
      }else{ # w_j appears in answer m times, keep words that have at most m w_j and w_j does not appear in position j
        w_j_count <- apply(word_bank_k, 1, function(w) sum(w %in% guess[j]))
        word_bank_k <- word_bank_k[(w_j_count <= w_j_colored_count) & (word_bank_k[,j] != guess[j]),]
      }
    }
    
    if (length(dim(word_bank_k)) < 2) return(word_bank_k)
  }
  
  return(word_bank_k)
}

################################################################################
next_guess <- function(word_bank_k){
  if (length(dim(word_bank_k)) < 2) return(word_bank_k)
  
  entropy <- apply(word_bank_k, 1, function(tilde_w) get_entropy(tilde_w, word_bank_k))
  return(word_bank_k[which.max(entropy),])
}

get_entropy <- function(tilde_w, word_bank_k){
  tilde_color <- t(apply(word_bank_k, 1, function(w) color_update(tilde_w, w)))
  duplicate_count <- table(do.call(paste, as.data.frame(tilde_color)))
  p <- duplicate_count/sum(duplicate_count)
  return(-sum(p*log2(p)))
}

################################################################################
wordle_game <- function(word_bank, guess=NA, answer=NA){
  # transform all strings to numeric
  word_bank_k <- t(sapply(word_bank, letter2number))
  if (is.na(guess)){
    guess <- letter2number("tares")  # next_guess(word_bank_k)
    guess <- letter2number("alter")
  }else{
    guess <- letter2number(tolower(guess))
  }
  if (is.na(answer)){
    answer <- word_bank_k[sample(1:nrow(word_bank_k), 1),]
  }else{
    answer <- letter2number(tolower(answer))
  }
  
  trial <- 1    # Initialize
  while (sum(guess == answer) != length(answer)){
    color <- color_update(guess, answer)
    word_bank_k <- word_bank_update(word_bank_k, color, guess)
    guess <- next_guess(word_bank_k)
    trial <- trial + 1
    
    if (trial > 15) break
  }
  return(c(trial, paste0(LETTERS[guess], collapse=""), paste0(LETTERS[answer], collapse="")))
}

################################################################################
df <- read.csv("wordle.csv")
results <- c()
for (i in 1:length(df$word)){
  re <- wordle_game(df$word, guess="tares", answer=df$word[i])
  print(c(i, re))
  results <- rbind(results, re)
}

count <- as.numeric(results[,1])
table(count)

#count (Ross - alter)
#1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16 
#1  168 2301 5507 3887 1646  668  328  176   84   50   25    9    3    1    1 

#count (Kaiser - alter)
#1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16 
#1  168 2510 5854 3607 1434  606  314  173   95   44   24   14    7    1    3 
# for 16, we have 4 greens but many possible options

#count (Kaiser - tares)
#1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16 
#1  213 2572 5632 3506 1524  678  328  187  109   54   27   13    8    2    1