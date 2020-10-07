library(dplyr)
library(reticulate)
library(lubridate)
library(tidyr)
source_python("model.py")
source_python("model_surge.py")



## Support Functions
lastMonday <- function(x) x - as.numeric(x-1+4)%%7
isSingleString <- function(input) {
  is.character(input) & length(input) == 1
}

## Global Variables
county <- read.csv('data/county_info.csv', colClasses=c("countyFIPS"="character"), as.is = TRUE)
pharma_selector <- c("Actemra",
                     "Aspirin",
                     "Cisatracurium",
                     "Dexamethasone",
                     "Dexmedetomidine",
                     "Midazolam",
                     "Propofol",
                     "Remdesivir",
                     "Rocuronium"
                     )
pharma_selector <- sort(pharma_selector)

item_selector <- c("Surg/Proc. Mask"="Surg/Proc. Mask", 
                   "N95 Respirator"="N95 Respirator", 
                   # "Face Shield"="Face Shield", 
                   # "Goggles"="Goggles", 
                   # "Sterile Exam Gloves"="Sterile Exam Gloves",
                   "Exam Gloves"="Exam Gloves",
                   # "Bouffant"="Bouffant",
                   # "Shoe Covers"="Shoe Covers",
                   "Isolation Gown"="Isolation Gown")
item_selector <- sort(item_selector)

item_dict <- list(
  'Surg/Proc. Mask' = 'isolation_mask',
  'N95 Respirator' = 'n95_respirator',
  'Isolation Gown' = 'isolation_gown',
  'Face Shield' = 'face_shield',
  'Goggles' = 'goggles',
  'Sterile Exam Gloves' = 'sterile_exam_gloves',
  'Exam Gloves' = 'non-sterile_exam_gloves',
  'Bouffant' = 'bouffant',
  'Shoe Covers' = 'shoe_covers',
  "Dexmedetomidine" = "Dexmedetomidine",
  "Propofol" = "Propofol",
  "Midazolam" = "Midazolam",
  "Aspirin" = "Aspirin",
  "Cisatracurium" = "Cisatracurium",
  "Rocuronium" = "Rocuronium",
  "Actemra" = "Actemra",
  "Remdesivir" = "Remdesivir",
  "Dexamethasone" = "Dexamethasone"
)

item_table_names <- list(
  'Surg/Proc. Mask' = 'Surg/Proc. Mask [unit]',
  'N95 Respirator' = 'N95 Respirator [unit]',
  'Isolation Gown' = 'Isolation Gown [unit]',
  'Face Shield' = 'Face Shield [unit]',
  'Goggles' = 'Goggles [unit]',
  'Sterile Exam Gloves' = 'Sterile Exam Gloves [pairs]',
  'Exam Gloves' = 'Exam Gloves [pairs]',
  'Bouffant' = 'Bouffant [unit]',
  'Shoe Covers' = 'Shoe Covers [pairs]',
  "Dexmedetomidine" = "Dexmedetomidine [mg]",
  "Propofol" = "Propofol [mg]",
  "Midazolam" = "Midazolam [mg]",
  "Aspirin" = "Aspirin [mg]",
  "Cisatracurium" = "Cisatracurium [mg]",
  "Rocuronium" = "Rocuronium [mg]",
  "Actemra" = "Actemra [mg]",
  "Remdesivir" = "Remdesivir [mg]",
  "Dexamethasone" = "Dexamethasone [mg]"
)

param = list(
  'ppe_set'= define_ppe_set(),
  'reuse'= define_reuse_policy(),
  'estimates'= define_sets_used()
)

sets_used = define_sets_used()
interaction_coeff = sets_used[['mean_estimate']]
ppe_set = define_ppe_set()
ppe_set_coeff = ppe_set[['crit_care']]
reuse = define_reuse_policy()

color = list(
  'rgba(46, 134, 222,1.0)',  #// muted blue
  'rgba(255, 159, 67,1.0)',  #// safety orange
  'rgba(16, 172, 132,1.0)',  #// cooked asparagus green
  'rgba(238, 82, 83,1.0)',  #// brick red
  'rgba(52, 31, 151,1.0)',  #// muted purple
  'rgba(34, 47, 62,1.0)',  #// chestnut brown
  'rgba(243, 104, 224,1.0)',  #// raspberry yogurt pink
  'rgba(131, 149, 167,1.0)',  #// middle gray
  'rgba(254, 202, 87,1.0)',  #// curry yellow-green
  'rgba(0, 210, 211,1.0)'   #// blue-teal
)

color_fill = list(
  'rgba(46, 134, 222,0.2)',  #// muted blue
  'rgba(255, 159, 67,0.2)',  #// safety orange
  'rgba(16, 172, 132,0.2)',  #// cooked asparagus green
  'rgba(238, 82, 83,0.2)',  #// brick red
  'rgba(52, 31, 151,0.2)',  #// muted purple
  'rgba(34, 47, 62,0.2)',  #// chestnut brown
  'rgba(243, 104, 224,0.2)',  #// raspberry yogurt pink
  'rgba(131, 149, 167,0.2)',  #// middle gray
  'rgba(254, 202, 87,0.2)',  #// curry yellow-green
  'rgba(0, 210, 211,0.2)'   #// blue-teal
)