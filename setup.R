##################################################
## Project: C19HCC COVID-19 Demand Model
## Purpose: A script to download all required packages
## Date: June 2020
##################################################

##adapted from: https://stackoverflow.com/questions/38928326/is-there-something-like-requirements-txt-for-r?noredirect=1&lq=1
pkgLoad <- function() {
  
  req.packages <- c("plyr", "dplyr", "tidyr", "shiny", "shinyjs", "parallel", "ggplot2", "utils", "plotly", "reticulate", "lubridate", "DT", "shinythemes", "shinyWidgets", "markdown")
  
  packagecheck <- match( req.packages, utils::installed.packages()[,1] )
  
  packagestoinstall <- req.packages[ is.na( packagecheck ) ]
  
  if( length( packagestoinstall ) > 0L ) {
    utils::install.packages( packagestoinstall)
  } else {
    print( "All requested packages already installed" )
  }
  
}

pkgLoad()


