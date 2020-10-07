import datetime
from scipy.integrate import odeint
import numpy as np
import pandas as pd


def calculate_case_distribution(cases_data, perc_admitted_crit_care=0.2952, perc_crit_care_vent=0.1769,
                                los_non_crit_care=3, los_crit_care=7, los_crit_care_vent=11):
    '''
    Calculate patient severity distribution based on defined proportions and length of stay
    
    Parameters
    ----------
    cases_data : dataframe
        data on covid cases.
    perc_admitted_crit_care : float, optional
        percentage of cases that will require critical care. The default is 0.25.
    perc_crit_care_vent : float, optional
        percentage of cases that will require critical care and mechanical ventilation.
    los_non_crit_care : integer, optional
        number of days a non-critical care patient stays in the hospital. The default is 3.
    los_crit_care : integer, optional
        number of days a critical care patient stays in the hospital. The default is 7.
    los_crit_care_vent : integer, optional
        number of days a critical care patient with mechanical ventilation stays in the hospital. The default is 11.

    Returns
    -------
    case_dist : dict
        contains distribution of patient type by date.

    '''
    
    #Concatenate known data with predicted data
    y = cases_data['y'].to_numpy()
    y_pred = cases_data['y_pred'].to_numpy()
    y = y[y >= 0]

    data = np.concatenate([y, y_pred[len(y):]])
    data = np.round(data)
    data = np.diff(data, prepend=[0])
    data[data < 0] = 0

    # Caluclate number of new cases for each day
    admitted_hospital = np.round(data * 0.0985)
    # admitted_hospital = data

    admitted_crit_care_vent = np.round(admitted_hospital*perc_crit_care_vent)
    admitted_crit_care = np.round(admitted_hospital*perc_admitted_crit_care) - admitted_crit_care_vent
    admitted_non_crit_care = admitted_hospital - admitted_crit_care - admitted_crit_care_vent

    # Calculate population of each patient type
    los_max = np.max([los_non_crit_care, los_crit_care, los_crit_care_vent])
    pop_non_crit_care = np.zeros(len(cases_data['y_pred']) + los_max)
    pop_crit_care = np.zeros(len(cases_data['y_pred']) + los_max)
    pop_crit_care_vent = np.zeros(len(cases_data['y_pred']) + los_max)
    for idx, _case in enumerate(data):
        pop_non_crit_care[idx:idx+los_non_crit_care] += admitted_non_crit_care[idx]
        pop_crit_care[idx:idx+los_crit_care] += admitted_crit_care[idx]
        pop_crit_care_vent[idx:idx+los_crit_care_vent] += admitted_crit_care_vent[idx]

    # Truncate population arrays to forecasted date length
    pop_non_crit_care = pop_non_crit_care[0:len(data)]
    pop_crit_care = pop_crit_care[0:len(data)]
    pop_crit_care_vent = pop_crit_care_vent[0:len(data)]


    case_dist = {
        'admitted_hospital': admitted_hospital,
        'admitted_non_crit_care': admitted_non_crit_care,
        'admitted_crit_care': admitted_crit_care,
        'admitted_crit_care_vent': admitted_crit_care_vent,
        'pop_non_crit_care': pop_non_crit_care,
        'pop_crit_care': pop_crit_care,
        'pop_crit_care_vent': pop_crit_care_vent
        }
    return case_dist

def calculate_ppe(data, ppe_set, sets, reuse_policy, reuse=False):
    '''
    Calculate PPE usage based on defined parameters


    Parameters
    ----------
    data : dict
        contains distribution of patient population.
    ppe_set : dict
        ppe usage coefficients per set of ppe.
    sets : dict
        sets of ppe used by healthcare worker type.
    reuse_policy : dict
        coefficients for ppe reuse.
    reuse : bool, optional
        flag if ppe should be reused. The default is False.

    Returns
    -------
    ppe : dict
        ppe usage by item.

    '''
    
    ppe = {}
    for idx, patient_type in enumerate(ppe_set):
        key = 'pop_' + patient_type
        pop = data[key]
        for item in ppe_set[patient_type]:
            reuse_factor = 1
            if reuse:
                reuse_factor = np.max([reuse_policy[item], 1]) #prevent division by zero
            if idx == 0:
                ppe[item] = np.round(pop * ppe_set[patient_type][item] * sets[patient_type] / reuse_factor)
            else:
                ppe[item] += np.round(pop * ppe_set[patient_type][item] * sets[patient_type]/ reuse_factor)
    return ppe


def model(county_data, parameters, reuse=False):
    '''
    Create model to calculate patient distribution and PPE demand

    Parameters
    ----------
    county_data : dataframe
        contains covid cases for a set of counties or states.
    parameters : dict
        contains coefficients for calculating ppe usage.
    reuse : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    df : dataframe
        contains case data and ppe usage.

    '''
    
    
    # Initialize variables
    ppe_set = parameters['ppe_set']
    reuse_policy = parameters['reuse']
    low_estimate = parameters['estimates']['low_estimate']
    high_estimate = parameters['estimates']['high_estimate']
    mean_estimate = parameters['estimates']['mean_estimate']

    # Stratify patient by severity
    case_distribution = calculate_case_distribution(county_data)
    temp = county_data.copy()
    temp['y'] = temp['low']
    case_distribution_low = calculate_case_distribution(temp)
    temp['y'] = temp['high']
    case_distribution_high = calculate_case_distribution(temp)
    # Estimate PPE
    low = pd.DataFrame(calculate_ppe(case_distribution_low, ppe_set, low_estimate, reuse_policy, reuse=reuse))
    high = pd.DataFrame(calculate_ppe(case_distribution_high, ppe_set, high_estimate, reuse_policy, reuse=reuse))
    mean = pd.DataFrame(calculate_ppe(case_distribution, ppe_set, mean_estimate, reuse_policy, reuse=reuse))
    low.columns = [str(col) + '_low' for col in low.columns]
    high.columns = [str(col) + '_high' for col in high.columns]
    mean.columns = [str(col) + '_mean' for col in mean.columns]

    # Convert case_distribution to a pandas dataframe
    for key in case_distribution:
        case_distribution[key] = case_distribution[key].tolist()
    case_distribution = pd.DataFrame(case_distribution)

    col_names = []
    for key in case_distribution_high:
        case_distribution_high[key] = case_distribution_high[key].tolist()
        col_names.append(key+'_high')
    case_distribution_high = pd.DataFrame(case_distribution_high)
    case_distribution_high.columns = col_names

    col_names = []
    for key in case_distribution_low:
        case_distribution_low[key] = case_distribution_low[key].tolist()
        col_names.append(key+'_low')
    case_distribution_low = pd.DataFrame(case_distribution_low)
    case_distribution_low.columns = col_names

    df = pd.concat([county_data, case_distribution, case_distribution_low, case_distribution_high, low, mean, high], axis=1)

    ## Calculate Pharmaceuticals
    # Load Pharma Coefficients
    pharm = pd.read_csv('./data/Pharmaceutical Data - Simplified (with Actemra).csv')
    pharm = pharm[~(pharm['Model Size'].isna())]
    medication = pharm['Medication'].unique().tolist()
    covid_population = df['pop_non_crit_care'] + df['pop_crit_care'] + df['pop_crit_care_vent']
    covid_population_low = df['pop_non_crit_care_low'] + df['pop_crit_care_low'] + df['pop_crit_care_vent_low']
    covid_population_high = df['pop_non_crit_care_high'] + df['pop_crit_care_high'] + df['pop_crit_care_vent_high']

    large_model = (county_data['NUM_LICENSED_BEDS'].to_numpy()[0] > 7500)

    for med in medication:
        if large_model:
            # Large Model
            param = pharm[pharm['Model Size'] == 'Over 7500 Beds'][pharm['Medication'] == med]
        else:
            # Small Model
            param = pharm[pharm['Model Size'] == 'Under 7500 Beds'][pharm['Medication'] == med]
            if len(param) == 0:
                param = pharm[pharm['Model Size'] == 'Over 7500 Beds'][pharm['Medication'] == med]

        perc_covid_patient = param['Covid 1: % of Covid+ Patients on Medication - Mean'].values[0]
        perc_covid_patient = float(perc_covid_patient.strip('%'))/100
        dosage_covid_patient = param['Covid 1: Use per Covid+ patient - Mean'].values[0]
        std_dosage_covid_patient = param['Covid 1: Use per Covid+  patient - Standard Deviation'].values[0]
        df[med +'_mean'] = covid_population * perc_covid_patient * dosage_covid_patient
        df[med +'_high'] = covid_population_high * perc_covid_patient * (dosage_covid_patient + std_dosage_covid_patient)
        df[med +'_low'] = covid_population_low * perc_covid_patient * (dosage_covid_patient - std_dosage_covid_patient)



    return df


def define_ppe_set():
    '''
    Parameters for PPE used by a healthcare professional
    '''
    core = {
        'isolation_mask': 1,
        'isolation_gown': 1,
        'n95_respirator': 1,
        'face_shield': 1,
        'goggles': 1,
        'sterile_exam_gloves':0,
        'non-sterile_exam_gloves': 1,
        'bouffant': 1,
        'shoe_covers': 1,
        }

    non_crit = core.copy()
    non_crit['sterile_exam_gloves'] = 0
    crit_care = core.copy()
    crit_care_vent = core.copy()

    output = {
        'non_crit_care': non_crit,
        'crit_care': crit_care,
        'crit_care_vent': crit_care_vent,
        }

    return output


def define_reuse_policy():
    '''
    Parameters for number of times PPE is reused
    '''
    core = {
        'isolation_mask': 1,
        'isolation_gown': 6,
        'n95_respirator': 12,
        'face_shield': 50,
        'goggles': 50,
        'sterile_exam_gloves': 1,
        'non-sterile_exam_gloves': 1,
        'bouffant': 1,
        'shoe_covers': 1
        }

    return core

def define_sets_used():
    '''
    Parameters for sets of PPE used in a given day when treating a COVID patient
    '''
    low_estimate = {
        'non_crit_care': 16,
        'crit_care': 28,
        'crit_care_vent': 28
        }

    high_estimate = {
        'non_crit_care': 20,
        'crit_care': 32,
        'crit_care_vent': 32,
        }

    mean_estimate = {}
    for key in low_estimate:
        mean_estimate[key] = (low_estimate[key] + high_estimate[key])/2

    output = {
        'low_estimate': low_estimate,
        'mean_estimate': mean_estimate,
        'high_estimate': high_estimate
        }
    return output


def format_data(county_output, items):
    '''
    Aggregate data into weekly basis


    Parameters
    ----------
    county_output : list
        list of dataframes containing data to be agggregated.
    items : list
        items to be aggregated.

    Returns
    -------
    output : dataframe
        contains weekly aggregated values.

    '''


    for county_idx, data in enumerate(county_output):
        if isinstance(data, str):
            continue
        data['date'] = pd.to_datetime(data['dates_str']) - pd.to_timedelta(7, unit='d')
        display_names = {
            'isolation_mask': 'Surg/Proc. Mask',
            'n95_respirator': 'N95 Respirator',
            'isolation_gown': 'Isolation Gown',
            'face_shield': 'Face Shield',
            'goggles': 'Goggles',
            'sterile_exam_gloves': 'Sterile Exam Gloves',
            'non-sterile_exam_gloves': 'Exam Gloves',
            'bouffant': 'Bouffant',
            'shoe_covers': 'Shoe Covers',
            }

        for idx, item in enumerate(items):
            select = ['date']
            replace = {}
            for estimate in ['low', 'mean', 'high']:
                select.append(item+'_'+estimate)
                replace[item+'_'+estimate] = estimate
            temp = data[select].rename(columns=replace)
            temp['item'] = display_names[item]

            if idx == 0:
                df = temp
            else:
                df = pd.concat([df, temp])

        # Load Pharma Coefficients
        pharm = pd.read_csv('./data/Pharmaceutical Data - Simplified (with Actemra).csv')
        pharm = pharm[~(pharm['Model Size'].isna())]
        pharm_names = pharm['Medication'].unique().tolist()

        for idx, item in enumerate(pharm_names):
            select = ['date']
            replace = {}
            for estimate in ['low', 'mean', 'high']:
                select.append(item+'_'+estimate)
                replace[item+'_'+estimate] = estimate
            temp = data[select].rename(columns=replace)
            temp['item'] = item
            df = pd.concat([df, temp])

        df_grouped = df.groupby(['item', pd.Grouper(key='date', freq='W-MON')])['low', 'mean', 'high'].sum()
        df_grouped = df_grouped.reset_index().sort_values(['item', 'date'])
        df_grouped['state'] = data['State'][0]
        df_grouped['county'] = data['County Name'][0]
        df_grouped['countyFIPS'] = data['countyFIPS'][0]

        if county_idx == 0:
            output = df_grouped
        else:
            output = pd.concat([output, df_grouped])
    output = output.reset_index(drop=True)
    output['date'] = output['date'].dt.strftime('%m/%d/%Y')
    return output


def get_county_data(fips, parameters, reuse=False):
    '''
    Calculate predictions for a given county or state


    Parameters
    ----------
    fips : list
        county FIPS codes or state abbreviations.
    parameters : dict
        coefficients used in ppe calculations.
    reuse : bool, optional
        flag to determine if ppe is reused. The default is False.

    Returns
    -------
    output : dict
        contains covid cases and ppe usage.

    '''
    
    data = pd.read_csv('./data/predicted_cases.csv', converters={'countyFIPS': str})
    data['low'] = data['y_pred']
    data['high'] = data['y_pred']
    hospital_beds = pd.read_csv('./data/hospital_beds_by_county.csv', converters={'countyFIPS': str})
    data['countyFIPS'] = data['countyFIPS'].apply(lambda x: x.zfill(5))
    hospital_beds['countyFIPS'] = hospital_beds['countyFIPS'].apply(lambda x: x.zfill(5))
    data = pd.merge(data, hospital_beds, on='countyFIPS', how='left')
    data.fillna(0)
    county_output = []
    data_fips = pd.unique(data['countyFIPS']).tolist()
    for code in fips:
        code = str(code)

        if len(code) == 2:
            # State Prediction
            data_state = data[data['County Name'] == code].reset_index(drop=True)
            # data_state_pharm['countyFIPS'] = data_state_pharm['County Name']
            # data_state = data_state.groupby('dates_str').sum().reset_index()
            data_state['State'] = code
            data_state['County Name'] = code
            data_state['countyFIPS'] = code
            result = model(data_state, parameters, reuse=reuse)
        else:
            # County Predict
            if code in data_fips:
                county_data = data[data['countyFIPS'] == code].reset_index(drop=True)
                result = model(county_data, parameters, reuse=reuse)
            else:
                result = 'FIPS not found: ' + str(code)
        county_output.append(result)

    df_agg = 'No Data'
    for idx, df in enumerate(county_output):
        try:
            if idx == 0:
                df_agg = df
            else:
                df_agg = df.add(df_agg, df, fill_value=0)
                df_agg['dates_str'] = df['dates_str']
                df_agg['County Name'] = ''
                df_agg['State'] = ''
                df_agg['countyFIPS'] = ''
        except:
            continue

    weekly_agg = format_data([df_agg], list(parameters['ppe_set']['non_crit_care'].keys()))
    weekly_data = format_data(county_output, list(parameters['ppe_set']['non_crit_care'].keys()))
    output = {
        'weekly_data': weekly_data,
        'daily_data': county_output,
        'daily_agg': df_agg,
        'weekly_agg': weekly_agg
        }
    return output

def create_summary_table(fips, parameters, reuse=False):
    '''
    Format data into tables for visualization


    Parameters
    ----------
    fips : list
        county FIPS codes or state abbreviations.
    parameters : dict
        coefficients used in ppe calculations.
    reuse : bool, optional
        flag to determine if ppe is reused. The default is False.

    Returns
    -------
    output : dict
        formatted tables for cases and ppe usage.

    '''
    
    df = get_county_data(fips, parameters=parameters, reuse=reuse)

    # Determine the start of each week to display
    today = datetime.date.today()
    last_monday = (today - datetime.timedelta(days=today.weekday())).strftime("%m/%d/%Y")
    coming_monday = (today + datetime.timedelta(days=-today.weekday(), weeks=1)).strftime("%m/%d/%Y")
    next_coming_monday = (today + datetime.timedelta(days=-today.weekday(), weeks=2)).strftime("%m/%d/%Y")
    final_monday = (today + datetime.timedelta(days=-today.weekday(), weeks=3)).strftime("%m/%d/%Y")

    # Filter out weeks not displayed
    weekly_agg = df['weekly_agg'][['item', 'date', 'mean']].copy()
    weekly_agg['date'] = pd.to_datetime(weekly_agg['date'])
    weekly_agg = weekly_agg[weekly_agg['date'] >= last_monday]
    weekly_agg['date'] = weekly_agg['date'].dt.strftime("%m/%d/%Y")

    # Convert dataframe from long to wide
    temp = pd.pivot(weekly_agg, index='item', columns='date', values='mean').reset_index()

    # Keep covid-related hospital PPE usage
    covid_hospital_demand = temp[['item', last_monday, coming_monday, next_coming_monday, final_monday]]
    covid_hospital_demand['total'] = covid_hospital_demand[last_monday] + covid_hospital_demand[coming_monday] + covid_hospital_demand[next_coming_monday] + covid_hospital_demand[final_monday]
    covid_hospital = temp[['item', last_monday]]
    covid_hospital.columns = ['item', 'covid_hospital']

    ## Load demand data for other sources
    other_source = pd.read_csv('./data/NON_COVID_DEMAND.csv', converters={'countyFIPS': str})
    other_source['countyFIPS'] = other_source['countyFIPS'].apply(lambda x: x.zfill(5))
    non_covid_pharm = pd.read_csv('./data/non_covid_pharmaceutical_demand.csv', converters={'countyFIPS': str})
    non_covid_pharm['countyFIPS'] = non_covid_pharm['countyFIPS'].apply(lambda x: x.zfill(5))

    # Filter data for selected FIPS/States
    code = str(fips[0])
    if len(code) == 2:
        # State
        record = other_source[other_source['State'].isin(fips)].groupby('State').sum().reset_index()
        record = record.rename(columns={'State': 'countyFIPS'})
        pharm_record = non_covid_pharm[non_covid_pharm['State'].isin(fips)].groupby('State').sum().reset_index()
        pharm_record = pharm_record.rename(columns={'State': 'countyFIPS'})

    else:
        # County
        record = other_source[other_source['countyFIPS'].isin(fips)]
        pharm_record = non_covid_pharm[non_covid_pharm['countyFIPS'].isin(fips)]

    other_source_filter_agg = record

    # Define core items for calculations (should match elsewhere)
    core = {
        'isolation_mask': 0,
        'isolation_gown': 0,
        'n95_respirator': 0,
        'face_shield': 0,
        'goggles': 0,
        'sterile_exam_gloves': 0,
        'non-sterile_exam_gloves': 0,
        'bouffant': 0,
        'shoe_covers': 0,
        }

    # Define pretty display names for each item
    display_names = {
        'isolation_mask': 'Surg/Proc. Mask',
        'n95_respirator': 'N95 Respirator',
        'isolation_gown': 'Isolation Gown',
        'face_shield': 'Face Shield',
        'goggles': 'Goggles',
        'sterile_exam_gloves': 'Sterile Exam Gloves',
        'non-sterile_exam_gloves': 'Exam Gloves',
        'bouffant': 'Bouffant',
        'shoe_covers': 'Shoe Covers',
        }

    # Aggregate PPE demand sources into higher groupings
    for item_idx, item in enumerate(core):
        temp = record[['countyFIPS']]
        police_key = 'police_' + item
        fire_key = 'fire_' + item
        emt_key = 'emt_' + item
        long_term_care_key = 'long_term_care_' + item
        home_health_key = 'home_health_' + item
        hospital_staff = 'hospital_staff_' + item
        hospital_patients = 'hospital_patients_' + item
        temp['item'] = display_names[item]
        temp['first_responders'] = other_source_filter_agg[police_key] + other_source_filter_agg[fire_key] + other_source_filter_agg[emt_key]
        temp['long_term_care'] = other_source_filter_agg[long_term_care_key]
        temp['home_health'] = other_source_filter_agg[home_health_key]
        temp['non_covid_hospital'] = other_source_filter_agg[hospital_staff] + other_source_filter_agg[hospital_patients]

        if item_idx == 0:
            other_source_df = temp
        else:
            other_source_df = pd.concat([other_source_df, temp])

    other_source_df = pd.DataFrame(other_source_df)

    # Multiply by factor of 7 to get weekly demand
    other_source_df[other_source_df.select_dtypes(include=['number']).columns] *= 7

    # Add non-covid related pharmaceutical demand in the hospital
    medications = pharm_record.columns.to_list()[5:]
    for med in medications:
        temp = pharm_record[['countyFIPS']]
        temp['item'] = med
        temp['first_responders'] = 0
        temp['long_term_care'] = 0
        temp['home_health'] = 0
        temp['non_covid_hospital'] = pharm_record[med]
        other_source_df = pd.concat([other_source_df, temp])
    other_demand_df = other_source_df.copy()
    other_source_df = other_source_df.groupby('item').sum().reset_index()
    # Merge dataframes to get weekly PPE demand from each source
    weekly_df = pd.merge(covid_hospital, other_source_df, on='item')

    # Calculate demand for weeks 1 to 3
    other_source_df['Weekly'] = other_source_df['first_responders'] + other_source_df['long_term_care'] + other_source_df['home_health'] + other_source_df['non_covid_hospital']
    temp = pd.merge(weekly_agg, other_source_df[['item', 'Weekly']], on='item')
    temp['total_demand'] = temp['mean'] + temp['Weekly']
    temp = temp[['item', 'date', 'total_demand']]
    temp = pd.pivot(temp, index='item', columns='date', values='total_demand').reset_index()
    temp = temp[['item', last_monday, coming_monday, next_coming_monday, final_monday]]

    #Create summary data
    summary = pd.merge(weekly_df, temp, on='item')
    summary[summary.select_dtypes(include=['number']).columns] = summary[summary.select_dtypes(include=['number']).columns].round()
    summary = summary[['item', last_monday, coming_monday, next_coming_monday, final_monday]]
    summary['total'] = summary[last_monday] + summary[coming_monday] + summary[next_coming_monday] + summary[final_monday]

    # Order columns in other_df
    other_source_df = other_source_df[['item', 'non_covid_hospital', 'first_responders', 'long_term_care', 'home_health', 'Weekly']]
    other_source_df = other_source_df.reset_index(drop=True)
    summary = summary.reset_index(drop=True)
    covid_hospital_demand = covid_hospital_demand.reset_index(drop=True)
    weekly_data = df['weekly_data']
    weekly_data['demand_source'] = 'covid_hospital'
    other_demand_df['other_demand_total'] = other_demand_df['first_responders'] + other_demand_df['long_term_care'] + other_demand_df['home_health'] + other_demand_df['non_covid_hospital']
    temp = pd.melt(other_demand_df,
                   id_vars=["item", 'countyFIPS'],
                   value_vars=['first_responders', 'long_term_care', 'home_health', 'non_covid_hospital', 'other_demand_total'])

    temp = temp.rename(columns={'variable': 'demand_source', 'value': 'mean'})
    temp['high'] = temp['mean'] * 1.25
    temp['low'] = temp['mean'] * 0.75
    temp = pd.merge(weekly_data[['item', 'date', 'countyFIPS', 'state', 'county']], temp, on=['item', 'countyFIPS'])
    weekly_data = pd.concat([weekly_data, temp])
    weekly_data = weekly_data.sort_values(by=['demand_source', 'item', 'state', 'countyFIPS', 'date'])
    weekly_data = weekly_data.reset_index(drop=True)
    allocation_table = weekly_data
    allocation_table = allocation_table[allocation_table['date'].isin([last_monday, coming_monday, next_coming_monday, final_monday])]
    allocation_table = allocation_table.replace({last_monday: 'Week 1', coming_monday: 'Week 2', next_coming_monday: 'Week 3', final_monday: 'Week 4'})
    output = {
        'summary': summary,
        'covid_hospital': covid_hospital_demand,
        'other_source_df': other_source_df,
        'allocation_table': allocation_table,
        'weekly_data': weekly_data,
        'daily_data': df['daily_data'],
        'daily_agg': df['daily_agg'],
        'weekly_agg': df['weekly_agg']
        }
    return output

####
#Execute Model
####
parameters = {
    'ppe_set': define_ppe_set(),
    'reuse': define_reuse_policy(),
    'estimates': define_sets_used()
    }


# Get county data for two states
# summary = create_summary_table(fips=['36081', '02270'], parameters=parameters, reuse=False)
summary = create_summary_table(fips=['VA'], parameters=parameters, reuse=False)
