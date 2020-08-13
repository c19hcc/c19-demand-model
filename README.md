## Background

The demand model and dashboard were developed as a collaboration effort
between Mitre, Cardinal Health, Llamasoft, LogicStream Health, Sodexo
Healthcare Services, and GHX. The forecast model was developed to
perform short-term predictions of future demand for personal protective
equipment (PPE) needed by healthcare and other essential workers when in
contact with COVID-19 patients. Model predictions are driven by the
confirmed number of COVID-19 cases within a specified location. The
surge scenario planning tool was created to assist decision makers in
planning for a potential surge in COVID-19 cases by forecasting
cumulative case counts over a 4-month window at different severity
levels (mild, moderate, severe, and very severe).

## Requirements

To view or update the COVID-19 Demand Model locally, the following pieces of computer software are required:

- A computer with any operating system (Windows, Linux, or MacOS)
- [R](https://www.r-project.org/)
- [RStudio](https://rstudio.com/)   
- [Python 3.0+](https://docs.conda.io/en/latest/miniconda.html)
- Web broswer (Preferred: Google Chrome, Firefox, Edge, Safari)

## Installation


Install Python packages either using `pip` or `conda`.

```
pip install -r requirements.txt
```

or...

```
conda install --file requirements.txt
```

Finally, open and run `setup.R` in RStudio or your IDE of choice. This script checks your local R packages, determines which, if any, new packages need to be installed, and then installs them from your default CRAN mirror.

## Usage

After downloading the required R and Python libraries, open `global.R` in RStudio and click the _"Run App"_ button to run the application locally. After loading the required libraries, the R Shiny application should open automatically in a new window or in your browser.

The data driving the application's analytics, metrics, and visualization is not automatically updated each time the app is run. Users must run `pull_usefacts_covid_cases.py` to retrieve updated county-level reported case counts and death counts. After pulling the data, users should run `predict_covid_cases.R` to update short-term forecasts.

## File Descriptions

-   `model.py` - contains the short-term forecast model
-   `model_surge.py` - contains the surge scenario planning tool
-   `pull_usafacts_covid_cases.py` - retrieves latest COVID case and
    death counts from USAFacts
-   `predict_covid_cases.py` - performs short-term forecasting
-   `global.R` - R Shiny dashboard (global)
-   `server.R` - R Shiny dashboard (back-end)
-   `ui.R` - R Shiny dashboard (front-end)

## License

Copyright 2020, The MITRE Corporation
Approved for Public Release; Distribution Unlimited. Case Number 20-2049.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
