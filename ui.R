library(shinythemes)
library(plotly)
library(shinyWidgets)
library(markdown)
library(shinyjs)

ui <- navbarPage(title="PPE Demand Model", header=list(tags$link(rel = "stylesheet", type = "text/css", href="main.css"), tags$link(rel = "stylesheet", type = "text/css", href="https://use.typekit.net/rrd2zlt.css")), id="nav", theme = shinytheme("flatly"),
  
  ## Main Tab for Modeling
  tabPanel("Model", icon=icon("chart-bar"),
           
    ## Side Bar Inputs
    sidebarPanel( width=2,
      fluidRow(selectInput(inputId="model_type",
                           label="Model Type",
                           choices=c("Select state"="", c("Short-Term Forecast", 
                                                          "Surge Planning: Mild", 
                                                          "Surge Planning: Moderate", 
                                                          "Surge Planning: Severe", 
                                                          "Surge Planning: Very Severe")),
                           selected="Short-Term Forecast",
                           multiple=FALSE)),
      fluidRow(selectInput(inputId="geo_level", 
                           label="Geographic Level",
                           choices=c("Select state"="", c("State", "County")), 
                           selected="County", 
                           multiple=FALSE)),
      fluidRow(selectInput(inputId="states", 
                           label="States",
                           choices=c("Select state"="", structure(state.abb, names=state.name), "Washington, DC"="DC"), 
                           selected="NY", 
                           multiple=TRUE)),
      fluidRow(
        conditionalPanel("input.states",
          selectInput(inputId="counties",
                      label="Counties",
                      choices=c("Select County"=""),
                      selected="",
                      multiple=TRUE))),
      fluidRow(
        selectInput(inputId="usage",
                    label="PPE Usage Policy",
                    choices=c("Select Policy"="","Normal"="Normal", "Conserve"="Conserve"), 
                    selected="Normal",
                    multiple=FALSE)),
      fluidRow(
        selectInput(inputId="items",
                    label="PPE Items",
                    choices=c("Select Item"="", item_selector),
                    selected=c("N95 Respirator"),
                    multiple=TRUE)),
      fluidRow(
        selectInput(inputId="pharma",
                    label="Pharmaceuticals",
                    choices=c("Select Pharmaceutical"="", pharma_selector),
                    selected=c(""),
                    multiple=TRUE)),
      fluidRow(
          actionButton(inputId="run_model",
                       label="Submit"),
          actionButton(inputId="clear_inputs",
                       label="Clear",
                       class="btn-reset"),
          actionButton(inputId="settings",
                       label="",
                       class="btn-reset",
                       icon=icon("cog"))
      ),
      fluidRow(
         span(textOutput("input_check_message"), style="color:rgb(255, 118, 117)")
        ),
      fluidRow(class="dwnld-default",
        br(),
        downloadLink("downloadData", "Download Predicted Demand")
        ),
      ),
    
    ## Main Body Outputs
    mainPanel(width=10,
      fluidRow(
        column(4,
               plotlyOutput('plot_predicted_cases')
        ),
        column(4,
               plotlyOutput('plot_patient_dist')
        ),
        column(4,
               plotlyOutput('plot_PPE_demand')
        )
      ),
      br(),
      fluidRow(
        tags$style(HTML("
                          .tabbable > .nav > li > a {
                            background-color: white;
                            color:black; 
                            font-family: trade-gothic-next-condensed, sans-serif;
                          }
                          .tabbable > .nav > li > a:hover {
                            background-color: white;
                            border-color: light silver;
                            color:#005b94; 
                            font-family: trade-gothic-next-condensed, sans-serif;
                            }  
                          .tabbable > .nav > li[class=active] > a {
                            background-color: white;
                            color:#005b94;
                            font-family: trade-gothic-next-condensed, sans-serif;
                            font-weight: bold;
                            }
                        ")),
        tabsetPanel(type="tabs",
                    tabPanel("COVID Hospital Demand",
                             br(),
                             column(12,
                                    DT::dataTableOutput("covid_hospital_table"),
                                    conditionalPanel("false", icon("crosshair"))               
                             ),
                             h5(textOutput("covid_hospital_table_text"))),
                    tabPanel("Other Demand Sources",
                             br(),
                             column(12,
                                    DT::dataTableOutput("other_table"),
                                    conditionalPanel("false", icon("crosshair"))               
                             ),
                             h5(textOutput("other_demand_table_text"))),
                    tabPanel("Total Demand",
                             br(),
                             column(12,
                                    DT::dataTableOutput("summary_table"),
                                    conditionalPanel("false", icon("crosshair"))               
                             ),
                             h5(textOutput(("summary_table_text"))))
                    
        )
      )

    )
  ),

  # Tab for Model Settings
  tabPanel("Settings",
           icon = icon("cog"),
           column(2),
           column(8,
                  fluidRow(
                    h3('COVID Hospital'),
                    h4('PPE Usage Coefficients'),
                    p('Personal protective equipment (PPE) is needed to protect healthcare workers when treating COVID-19 patients. The parameters below define the average amount of PPE used per interaction with a COVID patient.')
                  ),
                  fluidRow(
                    column(4,uiOutput("ppe_usage_coeff_inputs_col_a")),
                    column(4,uiOutput("ppe_usage_coeff_inputs_col_b"))
                  ),
                  fluidRow(
                    h4('Patient Interactions by Severity'),
                    p('The total amount of PPE used depends on the number of interactions between a healthcare worker and COVID patient, where an interaction is defined as the worker having to wear a new set of PPE (under normal operationg procedure) to perform their duties. The frequency of patient interactions is likely correlated with patient symptom severity. The parameters below define the average number of  interactions between a healthcare worker and a patient in a day by patient severity.')
                  ),
                  fluidRow(
                    column(4,
                           numericInput(inputId='non_crit_care_coeff', label='Non-critical Care', value=interaction_coeff[['non_crit_care']], min=1),
                           numericInput(inputId='crit_care_vent_coeff', label='Critical Care with Ventilator', value=interaction_coeff[['crit_care_vent']], min=1)
                           
                    ),
                    column(4,
                           numericInput(inputId='crit_care_coeff', label='Critical Care', value=interaction_coeff[['crit_care_vent']], min=1)
                    )
                  ),
                  fluidRow(
                    h4('PPE Usage Policy [Conserve]'),
                    p('PPE is normally replaced between each patient interaction. However, the rapid spread of COVID-19 resulted in a surge of demand for PPE, in particular, N95 respirators. Given the constrained supply of some PPE products, healthcare facilities have implemented strategies to extend the use of PPE. The parameters below define the "Conserve" PPE usage policy, where each coefficient represents the number of times a PPE item is used on average before being replaced.')
                           ),
                  fluidRow(
                    column(4,uiOutput("ppe_usage_policy_inputs_col_a")),
                    column(4,uiOutput("ppe_usage_policy_inputs_col_b"))
                  )
           ),
           column(2)
  ),
    
  # Tab for about page
  tabPanel("About", icon=icon("info-circle"),
    column(2),
    column(8,
           fluidRow(includeMarkdown("rmd/about.md"))
           ),
    column(2)
  )
)
