library(dplyr)
library(reticulate)
library(DT)

function(input, output, session) {


  ## Model Tab ###########################################
  
  #### Initialize Reactive Values
  reuse_policy <- reactiveValues()
  ppe_set_param <- reactiveValues()
  patient_interactions <- reactiveValues()
  for (name in names(reuse)){
    reuse_policy[[name]] <- reuse[[name]]
  }
  for (name in names(ppe_set_coeff)){
    ppe_set_param[[name]] <- ppe_set_coeff[[name]]
  }
  
  for (name in names(interaction_coeff)){
    patient_interactions[[name]] <- interaction_coeff[[name]]
  }
  #### Functions
  updateTable <- function(table_data) {
    table_data <- table_data %>%
      select(
        State=state,
        County=county,
        Date=date,
        Item=item,
        'Low Estimate'=low,
        'Mean Estimate'=mean,
        'High Estimate'=high,
        'Demand Source'=demand_source
      )
    df <- table_data %>%
      filter(
        is.null(input$items) | Item %in% input$items,
      )
    
    #Update Download Handler
    output$downloadData <- downloadHandler(
      filename = function() {
        paste("COVID_HOSPITAL_DEMAND_PREDICTION.csv", sep="")
      },
      content = function(file) {
        write.csv(df, file)
      }
    )
    
  }
  
  updateSummaryTable <- function(data_hospital, data_other, data) {
    # COVID Hospital Demand Table
    # data_hospital <- py_to_r(all_data$covid_hospital)
    item_filter <- c(input$items, input$pharma)
    data_hospital <- data_hospital %>%
      filter(
        item %in% item_filter
      )
    names(data_hospital)[names(data_hospital) == 'item'] <- 'Item'
    names(data_hospital)[2] <- paste('Week 1<br>', names(data_hospital)[2], sep='')
    names(data_hospital)[3] <- paste('Week 2<br>', names(data_hospital)[3], sep='')
    names(data_hospital)[4] <- paste('Week 3<br>', names(data_hospital)[4], sep='')
    names(data_hospital)[5] <- paste('Week 4<br>', names(data_hospital)[5], sep='')
    names(data_hospital)[names(data_hospital) == 'total'] <- 'Total'
    col_num = 6
    if (input$model_type!='Short-Term Forecast'){
      names(data_hospital)[2] <- 'Month 1'
      names(data_hospital)[3] <- 'Month 2'
      names(data_hospital)[4] <- 'Month 3'
      names(data_hospital)[5] <- 'Month 4'
      col_num = 6 #for formatting
    }
    for (i in data_hospital$Item){
      data_hospital[data_hospital==i] = item_table_names[[i]]
    }
    output$covid_hospital_table <- DT::renderDataTable({
      DT::datatable(data_hospital,
                    rownames=FALSE,
                    extensions='Buttons',
                    options=list( dom = 'Bfrtip', buttons = c('csv', 'excel'), columnDefs = list(list(className = 'dt-center', targets ="_all"))),
                    #options=list(columnDefs = list(list(className = 'dt-center', targets ="_all"))),
                    escape=FALSE
      )%>%
        formatCurrency(names(data_hospital)[2:col_num], currency = '', interval = 3, mark = ',', before = FALSE, digits=0)
    })    
    
    # Other Source Demand Table
    item_filter <- c(input$items, input$pharma)
    data_other <- data_other %>%
      filter(
        item %in% item_filter
      )
    data_other <- arrange(data_other, item)
    names(data_other)[names(data_other) == 'item'] <- 'Item'
    names(data_other)[names(data_other) == 'non_covid_hospital'] <- 'Hospital <br> Non-COVID'
    names(data_other)[names(data_other) == 'first_responders'] <- 'Police/Fire/EMT'
    names(data_other)[names(data_other) == 'long_term_care'] <- 'Long Term Care'
    names(data_other)[names(data_other) == 'home_health'] <- 'Home Care'
    names(data_other)[names(data_other) == 'Weekly'] <- 'Total'
    names(data_other)[names(data_other) == 'monthly'] <- 'Total'
    for (i in data_other$Item){
      data_other[data_other==i] = item_table_names[[i]]
    }
    output$other_table <- DT::renderDataTable({
      DT::datatable(data_other,
                    rownames=FALSE,
                    extensions='Buttons',
                    options=list( dom = 'Bfrtip', buttons = c('csv', 'excel'), columnDefs = list(list(className = 'dt-center', targets ="_all"))),
                    # options=list(columnDefs = list(list(className = 'dt-center', targets ="_all"))),
                    escape=FALSE
      )%>%
        formatCurrency(names(data_other)[2:6], currency = '', interval = 3, mark = ',', before = FALSE, digits=0)
    })   
    
    # Total Demand Table
    item_filter <- c(input$items, input$pharma)
    data <- data %>%
      filter(
        item %in% item_filter
      )
    data <- arrange(data, item)
    names(data)[names(data) == 'item'] <- 'Item'
    names(data)[2] <- paste('Week 1<br>', names(data)[2], sep='')
    names(data)[3] <- paste('Week 2<br>', names(data)[3], sep='')
    names(data)[4] <- paste('Week 3<br>', names(data)[4], sep='')
    names(data)[5] <- paste('Week 4<br>', names(data)[5], sep='')
    names(data)[names(data) == 'total'] <- 'Total'
    col_num = 6
    if (input$model_type!='Short-Term Forecast'){
      names(data)[2] <- 'Month 1'
      names(data)[3] <- 'Month 2'
      names(data)[4] <- 'Month 3'
      names(data)[5] <- 'Month 4'
      col_num=6
    }
    for (i in data$Item){
      data[data==i] = item_table_names[[i]]
    }
    
    output$summary_table <- DT::renderDataTable({
      DT::datatable(data,
                    rownames=FALSE,
                    extensions='Buttons',
                    options=list( dom = 'Bfrtip', buttons = c('csv', 'excel'), columnDefs = list(list(className = 'dt-center', targets ="_all"))),
                    # options=list(columnDefs = list(list(className = 'dt-center', targets ="_all"))),
                    escape=FALSE
      )%>%
        formatCurrency(names(data)[2:col_num], currency = '', interval = 3, mark = ',', before = FALSE, digits=0)
    })
  }
  
  updateVisuals <- function(daily_agg){
    ## Plotly Visualizations
    ax <-list(
      showline = TRUE,
      linewidth = 2,
      showgrid = F,
      zeroline = FALSE
    )
    
    font <- list(
      family = "Arial",
      size = 12
      )
    
    # Styling Parameters
    color_palette <- list()
    legend_pos <- c(list(x = 0, y = -0.2, orientation = 'h'))
    
    # Data Cleaning
    daily_agg <- daily_agg %>%
      mutate(date = as.Date(dates_str, "%m/%d/%Y"))
    
    daily_agg <- daily_agg %>%
      filter(
        date >= Sys.Date() - 75
      )
    
    daily_agg_short <- daily_agg %>%
      filter(
        y >= 0
      )
    
    ## Plots for FORECASTING MODEL
    if (input$model_type=='Short-Term Forecast'){
      # Predicated Cases Plot (FORECASTING MODEL)
      plot1 <- plot_ly(x = daily_agg[["date"]],
                       y = daily_agg[["y_pred"]],
                       type = 'scatter',
                       mode = 'lines',
                       name='Prediction',
      ) %>%
        layout(title='COVID-19 Cumulative Cases',
               legend = legend_pos,
               font = font,
               xaxis=ax,
               yaxis=c(ax, title="Cases")
               )
      
      plot1 <- plot1 %>% add_trace(x=daily_agg_short[["date"]],
                                   y=daily_agg_short[["y"]],
                                   mode='markers',
                                   name='Actual Values')
      output$plot_predicted_cases <- renderPlotly(plot1)
      
      # Patient Distribution (FORECASTING MODEL)
      plot2 <- plot_ly(
        x = daily_agg[["date"]],
        y = daily_agg[["pop_non_crit_care"]],
        name='Hospitalized',
        type = 'scatter',
        mode = 'lines') %>%
        layout(title='COVID-19 Patient Distribution',
               legend = legend_pos,
               xaxis=ax,
               yaxis=c(ax, title="Patients")) %>%
        add_trace(x=daily_agg[["date"]],
                  y=daily_agg[["pop_crit_care"]],
                  mode='lines',
                  name='In ICU') %>%
        add_trace(x=daily_agg[["date"]],
                  y=daily_agg[["pop_crit_care_vent"]],
                  name='On Ventilator',
                  mode='lines')
      output$plot_patient_dist <- renderPlotly(plot2)
      
      # PPE Demand Curves (FORECASTING MODEL)
      item_selected <- c(input$items, input$pharma)
      for (i in 1:length(item_selected)){
        item = item_selected[i]
        col_name = paste(item_dict[[item]], '_mean', sep='')
        if(i==1){
          plot3 <- plot_ly(
            x = daily_agg[["date"]],
            y = daily_agg[[col_name]],
            type = 'scatter',
            name = item_table_names[[item]],
            mode = 'lines') %>%
            layout(title='COVID-19 Hospital Demand',
                   legend = legend_pos,
                   font = font,
                   showlegend = TRUE,
                   xaxis=ax,
                   yaxis=c(ax, title="Demand [unit]"))
        } 
        else{
          plot3 <- plot3 %>%
            add_trace(x=daily_agg[["date"]],
                      y=daily_agg[[col_name]],
                      mode='lines',
                      name=item_table_names[[item]])
        }
      } 
      output$plot_PPE_demand <- renderPlotly(plot3)
    }
    else {
      daily_agg[["date"]] = c(1:length(daily_agg[["date"]]))
      #Predicated Cases Plot (SURGE MODEL)
      plot1 <- plot_ly(x = daily_agg[["date"]],
                       y = daily_agg[["y_pred"]],
                       type = 'scatter',
                       mode = 'lines',
                       name='Prediction',
                      ) %>%
        add_trace(x = daily_agg[["date"]],
                  y = daily_agg[["high"]],
                  type = 'scatter',
                  mode = 'lines',
                  line = list(color = 'transparent'),
                  showlegend = FALSE, 
                  name = 'Upper Bound'
                  ) %>% 
        add_trace(x = daily_agg[["date"]],
                  y = daily_agg[["low"]], 
                  type = 'scatter', 
                  mode = 'lines',
                  fill = 'tonexty', 
                  fillcolor='rgba(39,119,180,0.1)', 
                  line = list(color = 'transparent'),
                  showlegend = FALSE, 
                  name = 'Lower Bound')%>%
        layout(title='COVID-19 Cumulative Cases',
               legend = legend_pos,
               showlegend = TRUE,
               font = font,
               xaxis=c(ax, title="Days Since Outbreak"),
               yaxis=c(ax, title="Cases")
        )
      output$plot_predicted_cases <- renderPlotly(
        plot1
      )
      
      # Patient Distribution (SURGE MODEL)
      plot2 <- plot_ly(
        x = daily_agg[["date"]],
        y = daily_agg[["pop_non_crit_care"]],
        legendgroup='1',
        name='Hospitalized',
        type = 'scatter',
        mode = 'lines') %>%
        add_trace(x = daily_agg[["date"]],
                  y = daily_agg[["pop_non_crit_care_high"]],
                  type = 'scatter',
                  mode = 'lines',
                  line = list(color = 'transparent'),
                  legendgroup='1',
                  showlegend = FALSE, 
                  name = 'Upper Bound'
        ) %>% 
        add_trace(x = daily_agg[["date"]],
                  y = daily_agg[["pop_non_crit_care_low"]], 
                  type = 'scatter', 
                  mode = 'lines',
                  fill = 'tonexty', 
                  fillcolor='rgba(39,119,180,0.1)', 
                  line = list(color = 'transparent'),
                  legendgroup='1',
                  showlegend = FALSE, 
                  name = 'Lower Bound')%>%
        layout(title='COVID-19 Patient Distribution',
               legend = legend_pos,
               xaxis=c(ax, title="Days Since Outbreak"),
               yaxis=c(ax, title="Patients")) %>%
        add_trace(x=daily_agg[["date"]],
                  y=daily_agg[["pop_crit_care"]],
                  mode='lines',
                  line = list(color = 'rgba(255,127,14,1)'),
                  legendgroup='2',
                  name='In ICU') %>%
        add_trace(x = daily_agg[["date"]],
                  y = daily_agg[["pop_crit_care_high"]],
                  type = 'scatter',
                  mode = 'lines',
                  line = list(color = 'transparent'),
                  legendgroup='2',
                  showlegend = FALSE, 
                  name = 'Upper Bound'
        ) %>% 
        add_trace(x = daily_agg[["date"]],
                  y = daily_agg[["pop_crit_care_low"]], 
                  type = 'scatter', 
                  mode = 'lines',
                  fill = 'tonexty', 
                  fillcolor='rgba(255,127,14,0.1)', 
                  line = list(color = 'transparent'),
                  legendgroup='2',
                  showlegend = FALSE, 
                  name = 'Lower Bound')%>%
        add_trace(x=daily_agg[["date"]],
                  y=daily_agg[["pop_crit_care_vent"]],
                  name='On Ventilator',
                  legendgroup='3',
                  line = list(color = 'rgba(44,160,44,1)'),
                  mode='lines')%>%
        add_trace(x = daily_agg[["date"]],
                  y = daily_agg[["pop_crit_care_vent_high"]],
                  type = 'scatter',
                  mode = 'lines',
                  line = list(color = 'transparent'),
                  legendgroup='3',
                  showlegend = FALSE, 
                  name = 'Upper Bound'
        ) %>% 
        add_trace(x = daily_agg[["date"]],
                  y = daily_agg[["pop_crit_care_vent_low"]], 
                  type = 'scatter', 
                  mode = 'lines',
                  fill = 'tonexty', 
                  fillcolor='rgba(44,160,44,0.1)', 
                  line = list(color = 'transparent'),
                  legendgroup='3',
                  showlegend = FALSE, 
                  name = 'Lower Bound')
      
      
      output$plot_patient_dist <- renderPlotly(plot2)
      
      # PPE Demand Curves (SURGE MODEL)
      item_selected <- c(input$items, input$pharma)
      color_idx = 1
      for (i in 1:length(item_selected)){
        item = item_selected[i]
        col_name = paste(item_dict[[item]], '_mean', sep='')
        col_name_high = paste(item_dict[[item]], '_high', sep='')
        col_name_low = paste(item_dict[[item]], '_low', sep='')
        print(col_name_high)
        if(i==1){
          plot3 <- plot_ly(
            x = daily_agg[["date"]],
            y = daily_agg[[col_name]],
            type = 'scatter',
            name = item_table_names[[item]],
            legendgroup=toString(i),
            line = list(color = color[[color_idx]]),
            mode = 'lines') %>%
            add_trace(x = daily_agg[["date"]],
                      y = daily_agg[[col_name_high]],
                      type = 'scatter',
                      mode = 'lines',
                      line = list(color = 'transparent'),
                      legendgroup=toString(i),
                      showlegend = FALSE, 
                      name = 'Upper Bound'
            ) %>% 
            add_trace(x = daily_agg[["date"]],
                      y = daily_agg[[col_name_low]],
                      type = 'scatter', 
                      mode = 'lines',
                      fill = 'tonexty', 
                      fillcolor=color_fill[[color_idx]], 
                      line = list(color = 'transparent'),
                      legendgroup=toString(i),
                      showlegend = FALSE, 
                      name = 'Lower Bound') %>%
            layout(title='COVID-19 Hospital Demand',
                   legend = legend_pos,
                   font = font,
                   showlegend = TRUE,
                   xaxis=c(ax, title="Days Since Outbreak"),
                   yaxis=c(ax, title="Demand [unit]"))
          
          color_idx = color_idx + 1
          
        } 
        else{
          
          if(color_idx > 10){
            color_idx = 1
          }
          plot3 <- plot3 %>%
            add_trace(x=daily_agg[["date"]],
                      y=daily_agg[[col_name]],
                      mode='lines',
                      name=item_table_names[[item]],
                      legendgroup=toString(i),
                      line = list(color = color[[color_idx]])
                      )%>%
            add_trace(x = daily_agg[["date"]],
                      y = daily_agg[[col_name_high]],
                      type = 'scatter',
                      mode = 'lines',
                      line = list(color = 'transparent'),
                      legendgroup=toString(i),
                      showlegend = FALSE, 
                      name = 'Upper Bound'
            ) %>% 
            add_trace(x = daily_agg[["date"]],
                      y = daily_agg[[col_name_low]],
                      type = 'scatter', 
                      mode = 'lines',
                      fill = 'tonexty', 
                      fillcolor=color_fill[[color_idx]], 
                      line = list(color = 'transparent'),
                      legendgroup=toString(i),
                      showlegend = FALSE, 
                      name = 'Lower Bound')
          
          color_idx = color_idx + 1
        }
      }
      output$plot_PPE_demand <- renderPlotly(plot3)
    }
  }
  
  checkInputs <- function(){
    items_selected <- c(input$items, input$pharma)
    ## Check inputs are not empty otherwise return error message
    check <- TRUE
    output$input_check_message <- renderText(NULL)
    #State Input
    if(is.null(input[["states"]])){
      output$input_check_message <- renderText({"Please select a state."})
      check <- FALSE
    }
    #County Input
    if (input$geo_level =="County" && check){ 
      if (is.null(input[["counties"]])) {
        output$input_check_message <- renderText({"Please select a county."})
        check <- FALSE        
      }
      else if (input[["counties"]]==''){
        output$input_check_message <- renderText({"Please select a county."})
        check <- FALSE            
      }
    }
    #Item Inputs
    if (is.null(items_selected) && check) {
      output$input_check_message <- renderText({"Please select an item or pharmaceutical."})
      check <- FALSE
    }
    # print(check)
    return(check)
  }
  
  #### Observers
  
  #Update COVID Hospital Settings
  observe({
    for (name in names(item_dict)){
      #Usage Coefficients
      key = paste0(name,'_coeff')
      if (!is.null(input[[key]])){
        # print(key)
        # print(input[[key]])
        ppe_set_param[[item_dict[[name]]]] <- input[[key]]
      }
      #Reuse Policy
      key = paste0(name,'_reuse')
      if (!is.null(input[[key]])){
        # print(key)
        # print(input[[key]])
        reuse_policy[[item_dict[[name]]]] <- input[[key]]
      }
      
      
    }
  })
  
  observe({
    for (name in names(interaction_coeff)){
      #Patient Interaction Coefficients
      key = paste0(name,'_coeff')
      if (!is.null(input[[key]])){
        patient_interactions[[name]] <- input[[key]]
      }
    }
  })
  
  #Update county selection input
  observe({
    counties <- if (is.null(input$states)| input$geo_level=="State") character(0) else {
      filter(county, state %in% input$states) %>%
        `$`('name_pretty') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$counties[input$counties %in% counties])
    updateSelectizeInput(session, "counties", choices = counties,
                         selected = stillSelected, server = TRUE)
  })
  
  # Update settings tab inputs for COVID Hospital
  observe({
    item_list <- input$items
    col_a <- head(input$items, ceiling(length(input$items)/2))
    col_b <- tail(input$items, length(input$items) - ceiling(length(input$items)/2))
    if (length(input$items) > 0){
      # PPE Usage Coefficients
      output$ppe_usage_coeff_inputs_col_a <- renderUI({
        lapply(1:length(col_a), function(i) {
          numericInput(inputId=paste0(col_a[i],'_coeff'), label=paste0(col_a[i]), value=ppe_set_param[[item_dict[[col_a[i]]]]], min=0)
        })
      })
      # PPE Usage Policy
      output$ppe_usage_policy_inputs_col_a <- renderUI({
        lapply(1:length(col_a), function(i) {
          numericInput(inputId=paste0(col_a[i],'_reuse'), label=paste0(col_a[i]), value=reuse_policy[[item_dict[[col_a[i]]]]], min=1)
        })
      })
      if(length(input$items) > 1){
        # PPE Usage coefficients
        output$ppe_usage_coeff_inputs_col_b <- renderUI({
          lapply(1:length(col_b), function(i) {
            numericInput(inputId=paste0(col_b[i],'_coeff'), label=paste0(col_b[i]), value=ppe_set_param[[item_dict[[col_b[i]]]]], min=0)
          })
        })
        # PPE Usage Policy
        output$ppe_usage_policy_inputs_col_b <- renderUI({
          lapply(1:length(col_b), function(i) {
            numericInput(inputId=paste0(col_b[i],'_reuse'), label=paste0(col_b[i]), value=reuse_policy[[item_dict[[col_b[i]]]]], min=1)
          })
        })
      }
      else{
        output$ppe_usage_policy_inputs_col_b <- NULL
      }
    }
    else{
      output$ppe_usage_policy_inputs_col_a <- NULL
      output$ppe_usage_policy_inputs_col_b <- NULL
      output$ppe_usage_coeff_inputs_col_a <- NULL
      output$ppe_usage_coeff_inputs_col_b <- NULL
      
    }
  })
  
  ## Run Model Button
  demand_model <- eventReactive(input$run_model,{
    county_filter <- county %>%
      filter(
        is.null(input$states) | state %in% input$states,
        is.null(input$counties) | name_pretty %in% input$counties,
      )
    fips <- as.list(county_filter$countyFIPS)
    if("State" %in% input$geo_level){
      fips <- as.list(input$states)
    }
    print(fips)
    reuse_flag = TRUE
    if(input$usage == "Normal"){
      reuse_flag = FALSE
    }
    
    #Apply parameters
    ppe_set = define_ppe_set()
    for (name in names(ppe_set)){
      ppe_set[[name]] = reactiveValuesToList(ppe_set_param)
    }
    sets_used = define_sets_used()
    temp <- reactiveValuesToList(patient_interactions)
    sets_used[['mean_estimate']] <- temp
    sets_used[['low_estimate']] <- lapply(temp, "/", 1)
    sets_used[['high_estimate']] <- lapply(temp, "*", 1)
    
    parameters = list(
      'ppe_set'= ppe_set,
      'reuse'= reactiveValuesToList(reuse_policy),
      'estimates'= sets_used
    )
    
    if (input$model_type == 'Short-Term Forecast'){
      summary <- create_summary_table(fips=fips, parameters=parameters, reuse=reuse_flag)  
    }
    if (input$model_type == 'Surge Planning: Very Severe'){
      summary <- create_summary_table_surge(fips=fips, parameters=parameters, reuse=reuse_flag, surge=TRUE, rank=as.integer(0))
    }
    
    if (input$model_type == 'Surge Planning: Severe'){
      summary <- create_summary_table_surge(fips=fips, parameters=parameters, reuse=reuse_flag, surge=TRUE, rank=as.integer(1))
    }
    
    if (input$model_type == 'Surge Planning: Moderate'){
      summary <- create_summary_table_surge(fips=fips, parameters=parameters, reuse=reuse_flag, surge=TRUE, rank=as.integer(2))
    }
    
    if (input$model_type == 'Surge Planning: Mild'){
      summary <- create_summary_table_surge(fips=fips, parameters=parameters, reuse=reuse_flag, surge=TRUE, rank=as.integer(3))
    }
    
    # Update Table Text
    if(input$model_type=='Short-Term Forecast'){
      output$covid_hospital_table_text <- renderText({"*Weekly demand based on estimated number of COVID patients"})
      output$other_demand_table_text <- renderText({'*Weekly demand based on estimated number of employees'})
      output$summary_table_text <- renderText({'*Weekly combined demand from COVID hospital and other sources'})
    }
    else{
      output$covid_hospital_table_text <- renderText({"*Monthly demand based on estimated number of COVID patients"})
      output$other_demand_table_text <- renderText({'*Monthly demand based on estimated number of employees'})
      output$summary_table_text <- renderText({'*Monthly combined demand from COVID hospital and other sources'})      
    }
    
    #Return Data for Use in App
    return(c("daily_data"=summary[["daily_data"]],
             "daily_agg"=summary[["daily_agg"]],
             "weekly_data"=summary[["weekly_data"]],
             "weekly_agg"=summary[["weekly_agg"]],
             "allocation_table"=summary[["allocation_table"]],
             "summary"=summary[["summary"]],
             "covid_hospital"=summary[["covid_hospital"]],
             "other_source_df"=summary[["other_source_df"]]))
  })
  
  observeEvent(input$run_model, {
    print('run_model event received')
    if (checkInputs()){
      # Get Data for Visualizations
      temp = demand_model()
      data_hospital = py_to_r(temp[["covid_hospital"]])
      data_other = py_to_r(temp[["other_source_df"]])
      data_total = py_to_r(temp[["summary"]])
      
      if (!isSingleString(data)){
        #Update Table
        updateTable(py_to_r(temp[["weekly_data"]]))
        updateSummaryTable(data_hospital, data_other, data_total)
        #Update Visualizations
        updateVisuals(py_to_r(temp[["daily_agg"]]))
      }
      else{
        output$input_check_message <- renderText({"There was insufficient data for this county."})
      }
      
    }
  })
  
  ## Clear Input Button
  observeEvent(input$clear_inputs, {
    updateSelectizeInput(session, "geo_level", selected = "State", choices=c("Select state"="", c("State", "County")), server = TRUE)
    updateSelectizeInput(session, "states", selected = NULL, choices=c(structure(state.abb, names=state.name)), server = TRUE)
    updateSelectizeInput(session, "counties", selected = NULL, choices = list(), server = TRUE)
    updateSelectizeInput(session, "usage", selected = "Normal", choices=c("Normal"="Normal", "Conserve"="Conserve"), server = TRUE)
    updateSelectizeInput(session, "items", selected = NULL, choices=item_selector, server = TRUE)
    updateSelectizeInput(session, "pharma", selected = NULL, choices=pharma_selector, server = TRUE)
    for (name in names(reuse)){
      reuse_policy[[name]] <- reuse[[name]]
    }
    for (name in names(ppe_set_coeff)){
      ppe_set_param[[name]] <- ppe_set_coeff[[name]]
    }
  })
  
  ## Settings Button
  observeEvent(input$settings, {
    updateTabsetPanel(session, "nav", selected="Settings")
  })
  
  
###### Initialize Default View
observeEvent(input$states,{
    print('init_app')
    counties <- filter(county, state %in% 'NY') %>%
      `$`('name_pretty') %>%
      unique() %>%
      sort()
    updateSelectizeInput(session, "counties", choices = counties,
                         selected = "Queens County", server = TRUE)
    # Get Data for Visualizations
    reuse = TRUE
    if(input$usage == "Normal"){
      reuse = FALSE
    }
    summary <- create_summary_table(fips=list('36081'), parameters=param, reuse=reuse)
    data_hospital = py_to_r(summary[["covid_hospital"]])
    data_other = py_to_r(summary[["other_source_df"]])
    data_total = py_to_r(summary[["summary"]])
    
    #Update Table
    updateTable(py_to_r(summary[["weekly_data"]]))
    updateSummaryTable(data_hospital, data_other, data_total)
    
    #Update Visualizations
    updateVisuals(py_to_r(summary[["daily_agg"]]))
  }, once=TRUE)
}
