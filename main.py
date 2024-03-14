
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.neighbors import BallTree
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from statsmodels.graphics.api import abline_plot

from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

st.set_page_config(page_title="SASEF Health App", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

pdata = pd.read_csv("psychdata.csv")
pdata = pdata.rename(columns={"Unnamed: 0": "Level", "V1": "OR"})
pdata = pdata.sort_values(by=['OR'],axis=0)
pdata['Inputs'] = pdata.reset_index().index + 1
pdata = pdata.drop(['Level'],axis=1)

p_inputs = pdata['Inputs']
p_inputs = p_inputs.to_numpy()
p_outputs = pdata['OR']
p_outputs = p_outputs.to_numpy()

X = p_inputs
y = p_outputs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train = sm.add_constant(X_train)

glm_model = sm.GLM(y_train, X_train, family=sm.families.Gaussian())
glm_results = glm_model.fit()

edata = pd.read_csv("envdata.csv")
edata = edata.rename(columns={"Unnamed: 0": "Level", "V1": "OR"})
edata = edata.sort_values(by=['OR'],axis=0)
edata['Inputs'] = edata.reset_index().index + 1
edata = edata.drop(['Level'],axis=1)

e_inputs = edata['Inputs']
e_inputs = e_inputs.to_numpy()
e_outputs = edata['OR']
e_outputs = e_outputs.to_numpy()

X_e = e_inputs
y_e = e_outputs

X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(X_e, y_e, test_size=0.2, random_state=42)
X_e_train = sm.add_constant(X_e_train)

glm_e_model = sm.GLM(y_e_train, X_e_train, family=sm.families.Gaussian())
glm_e_results = glm_e_model.fit()

final_or = 1

st.title("Your Child's Health Outcome")
st.write("Enter some basic information about yourself to determine the likelihood of your child developing allergic diseases in adolescence.")

st.write("Select the answer that applies. For questions relating to mental state, select yes if you felt your health was impacted by distress.")

time = st.radio(
    "Has your child been born?",
    ["Yes", "No"],index=None)

if time == 'Yes':
    prenatal = False
    postnatal = True
elif time == "No":
    prenatal = True
    postnatal = False
else:
    st.write(" ")
    prenatal=False
    postnatal = False

if prenatal:
    pre_stress_input = st.radio("Have you experienced stress or depression during your pregnancy?", ["I have experienced stress, depression, and/or negative life events that have impacted my mental health during pregnancy.","I have not experienced stress, depression, and/or negative life events that have impacted my mental health during pregnancy."],index=None)
    if pre_stress_input == "I have experienced stress, depression, and/or negative life events that have impacted my mental health during pregnancy.":
        pre_stress = st.slider('Describe the level of stress you experienced during pregnancy.', 0.0, 100.0)
        pre_stress_val = pre_stress * 25
        X_p_val = np.array([1, pre_stress_val,])
        X_p_val = sm.add_constant(X_p_val)
        pre_p_or = glm_results.predict(X_p_val)
        final_or = final_or * pre_p_or[1]
    if pre_stress_input == "I have not experienced stress, depression, and/or negative life events that have impacted my mental health during pregnancy.":
        final_or = final_or * 1
    else:
        st.write(" ")
    
if postnatal:
    pre_stress_input_post = st.radio("Did you experience stress or depression during your pregnancy?",["I experienced stress, depression, and/or negative life events during pregnancy.","I did not experience stress, depression, and/or negative life events during pregnancy."],index=None)
    post_stress_input = st.radio("Have you experienced stress or depression since your child was born?", ["I have experienced stress, depression, and/or negative life events since my child was born.","I have not experienced stress, depression, and/or negative life events since my child was born."], index=None)
    if pre_stress_input_post == "I experienced stress, depression, and/or negative life events during pregnancy.":
        pre_stress = st.slider('Describe the level of stress you experienced during pregnancy.', 0.0, 100.0)
        pre_stress_val = pre_stress * 25
        X_p_val = np.array([1, pre_stress_val,])
        X_p_val = sm.add_constant(X_p_val)
        pre_p_or = glm_results.predict(X_p_val)
        final_or = final_or * pre_p_or[1]
    if pre_stress_input_post == "I did not experience stress, depression, and/or negative life events during pregnancy.":
        final_or = final_or * 1
    if post_stress_input == "I have experienced stress, depression, and/or negative life events since my child was born.":
        post_stress = st.slider('Describe the level of stress you experienced since your child was born.', 0.0, 100.0)
        post_stress_val = post_stress * 25
        X_po_val = np.array([1, post_stress_val,])
        X_po_val = sm.add_constant(X_po_val)
        post_p_or = glm_results.predict(X_po_val)
        final_or = final_or * post_p_or[1]
    if post_stress_input == "I have not experienced stress, depression, and/or negative life events since my child was born.":
        final_or = final_or * 1
    else:
        st.write(" ")
    
else:
    st.write(" ")

aq_data = pd.read_csv("uscities_airquality.csv")
aq_data = aq_data.ffill(axis=0)
aq_data = aq_data.drop([1042,1041,1040,1039,1038,1037,1036,1035,1034,1033])

locs = pd.read_csv("cbsa_coordinates.csv")
locs = locs.drop(['CBSA_TYPE',"ALAND","AWATER","ALAND_SQMI","AWATER_SQMI",'NAME','CSAFP'],axis=1)
locs = locs.rename(columns={"INTPTLAT": "Lat", "INTPTLONG": "Long"})
locs = locs.set_index("GEOID")
locs.Lat = locs.Lat.apply(np.radians)
locs.Long = locs.Long.apply(np.radians)

zip_cbsa = pd.read_csv("zip_to_cbsa.csv",dtype={'ZIP': str})
zip_cbsa = zip_cbsa.drop(["RES_RATIO","BUS_RATIO",'OTH_RATIO',"TOT_RATIO"],axis=1)

res_data = [{"Resource":"National Maternal Mental Health Hotline", "Link":"https://mchb.hrsa.gov/national-maternal-mental-health-hotline","Description":"Free, 24/7 confidential support during all stages of pregnancy"},{"Resource":"Postpartum Support International", "Link":"https://www.postpartum.net/","Description":"Information and professional online support groups for all types of new parents"},{"Resource":"CDC Maternal and Infant Health", "Link":"https://www.cdc.gov/reproductivehealth/maternalinfanthealth/index.html", "Description":"Information from experts for many pre- and postnatal conditions"},{"Resource":"Maternal Mental Health Leadership Alliance", "Link":"https://www.mmhla.org/","Description":"Futher resources for mental health, advocacy for improving maternal care"},{"Resource":"Substance Abuse and Mental Health Services Administration", "Link":"https://www.samhsa.gov/ ","Description":"Resources for parents struggling with substance abuse/mental health issues"},{"Resource":"Air Pollution and Pregnancy (EPA)", "Link":"https://www.epa.gov/children/promoting-good-prenatal-health-air-pollution-and-pregnancy-january-2010","Description":"Information about environmental exposure, avoidance steps for new/expecting mothers"},{"Resource": "Air Pollution (UNICEF)", "Link":"https://www.unicef.org/parenting/air-pollution","Description":"Information about the effects of air pollution specifically on children"},{"Resource":"Protecting Children’s Environmental Health (EPA)", "Link":"https://www.epa.gov/children","Description":"Resources for healthcare providers, teachers, educators, and caregivers"},{"Resource":"Moms Clean Air Force", "Link":"https://www.momscleanairforce.org/resources/topic/babies/","Description":"Resources about the health impacts of air pollution on young children"}] 
resources = pd.DataFrame(res_data)
# st.dataframe(resources,column_config={"Link": st.column_config.LinkColumn("Link")},hide_index=True)

tree = BallTree(locs[['Lat', 'Long']].values, leaf_size=2, metric='haversine')

get_locs = st.text_input(label="Enter your zip code")
zip_to_geoid = zip_cbsa[zip_cbsa['ZIP'] == get_locs]
get_cbsa = zip_to_geoid["CBSA"]

if len(get_locs) == 0:
    st.text(" ")

if len(get_cbsa) == 1:
    geoid = int(get_cbsa.iloc[0])
    query_point = locs.loc[geoid][["Lat", "Long"]].values
    distances, indices = tree.query([query_point], k=10)
    result_df = locs.iloc[indices[0]]
elif len(get_cbsa) > 1:
    dfs=[]
    for loc in get_cbsa:
        geoid=int(loc)
        query_point = locs.loc[geoid][["Lat", "Long"]].values
        distances, indices = tree.query([query_point], k=10)
        result_part_df = locs.iloc[indices[0]]
        dfs.append(result_part_df)
    result_df = pd.concat(dfs) 
if len(get_locs) > 0:
    if len(get_cbsa)== 0:
        st.text("Sorry, we were unable to located a Core Based Statistical Area associated with that zip code. Try entering a different zip code near your area.")
        result_df = pd.DataFrame()

try:
    result_df = result_df.reset_index()
    new_lst = result_df["GEOID"]
    new_lst = new_lst.drop_duplicates()
    aq_data["CBSA"] = aq_data["CBSA"].astype(int)
    filtered_aqs = aq_data[aq_data['CBSA'].isin(new_lst)]
    filtered_aqs = filtered_aqs.drop(['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'],axis=1)
  
    co_df = filtered_aqs[filtered_aqs['Pollutant'] == "CO"]
    no2_df = filtered_aqs[filtered_aqs['Pollutant'] == "NO2"]
    o3_df = filtered_aqs[filtered_aqs['Pollutant'] == "O3"]
    pm10_df = filtered_aqs[filtered_aqs['Pollutant'] == "PM10"]
    pm25_df = filtered_aqs[filtered_aqs['Pollutant'] == "PM2.5"]
    so2_df = filtered_aqs[filtered_aqs['Pollutant'] == "SO2"]
    indiv_dfs = [co_df,no2_df,o3_df,pm10_df,pm25_df,so2_df]

    pre_lims_above = []
    pre_lims_below = []
    post_lims_above = []
    post_lims_below = []
    pre_sums_above = 0
    pre_sums_below = 0
    post_sums_above = 0
    post_sums_below = 0
    frac_above_lim = 0
    X_e_val = 0
    pre_e_or = 0
    post_e_or = 0

    def naaq_pre_lims(df):
        poll_vals = df['2022']
        base_or = 1
        name = "blank"
        for index, row in df.iterrows():
            if row['Pollutant'] == "O3":
                limit = 0.07
                name = "O3"
            elif row['Pollutant'] == "CO":
                limit = 9.0
                name = "CO"
            elif row['Pollutant'] == "NO2":
                name = "NO2"
                if row['Trend Statistic'] == 'Annual Mean':
                    limit = 53.0
                elif row['Trend Statistic'] == '98th Percentile':
                    limit = 100.0
            elif row['Pollutant'] == "PM10":
                name = "PM10"
                limit = 150.0
            elif row['Pollutant'] == "PM2.5":
                name = "PM2.5"
                if row['Trend Statistic'] == 'Weighted Annual Mean':
                    limit = 12.0
                elif row['Trend Statistic'] == '98th Percentile':
                    limit = 35.0
            elif row['Pollutant'] == "SO2":
                name = "SO2"
                limit = 75.0   
                
        counts_above = 0
        counts_below = 0
            
        for val in poll_vals:
            if float(val) >= limit:
                counts_above = counts_above + 1
            else:
                counts_below = counts_below + 1   
                
        pre_lims_above.append(counts_above)
        pre_lims_below.append(counts_below)


    def naaq_post_lims(df):
        poll_vals = df['2022']
        base_or = 1
        name = "blank"
        for index, row in df.iterrows():
            if row['Pollutant'] == "O3":
                limit = 0.07
                name = "O3"
            elif row['Pollutant'] == "CO":
                limit = 9.0
                name = "CO"
            elif row['Pollutant'] == "NO2":
                name = "NO2"
                if row['Trend Statistic'] == 'Annual Mean':
                    limit = 53.0
                elif row['Trend Statistic'] == '98th Percentile':
                    limit = 100.0
            elif row['Pollutant'] == "PM10":
                name = "PM10"
                limit = 150.0
            elif row['Pollutant'] == "PM2.5":
                name = "PM2.5"
                if row['Trend Statistic'] == 'Weighted Annual Mean':
                    limit = 12.0
                elif row['Trend Statistic'] == '98th Percentile':
                    limit = 35.0
            elif row['Pollutant'] == "SO2":
                name = "SO2"
                limit = 75.0
                
        counts_above = 0
        counts_below = 0
            
        for val in poll_vals:
            if float(val) >= limit:
                counts_above = counts_above + 1
            else:
                counts_below = counts_below + 1
                
        post_lims_above.append(counts_above)
        post_lims_below.append(counts_below)     


    if prenatal:        
        for df in indiv_dfs:
            if len(df) > 0:
                naaq_pre_lims(df)
        for val in pre_lims_above:
            pre_sums_above = pre_sums_above + val
        for val in pre_lims_below:
            pre_sums_below = pre_sums_below + val

        frac_above_lim = 2500*(pre_sums_above/pre_sums_below)
        
        X_e_val = np.array([1, frac_above_lim,])
        if frac_above_lim != 0:
            X_e_val = sm.add_constant(X_e_val)
            pre_e_or = glm_e_results.predict(X_e_val)
            final_or = final_or * pre_e_or[1]
        else:
            final_or = final_or * 1
        
        rounded_final = str(round((final_or-1)*100))
        if int(rounded_final) > 0:
            st.write("Based on exposure to psychological distress and environmental pollution, there is a " + rounded_final + "% increase in the odds of your child developing allergies. There are also many other possible factors that may contribute to atopy, such as family history of allergic disease. Explore the resources below and consult with your doctor for more information.")
            st.dataframe(resources,column_config={"Link": st.column_config.LinkColumn("Link")},hide_index=True)
        elif int(rounded_final) == 0:
            st.write("Based on exposure to psychological distress and environmental pollution, your child likely will not have an increased risk of atopy. However, there are many other possible factors that may contribute to atopy, such as family history of allergic disease. If answers to the above questions change, explore the resources below and consult with your doctor for more information.")
            st.dataframe(resources,column_config={"Link": st.column_config.LinkColumn("Link")},hide_index=True)
        else:
            st.write("something is wrong prenatal")
        
    if postnatal:           
        for df in indiv_dfs:
            if len(df) > 0:
                naaq_post_lims(df)
        for val in post_lims_above:
            post_sums_above = post_sums_above + val
        for val in post_lims_below:
            post_sums_below = post_sums_below + val

        frac_above_lim = 2500*(post_sums_above/post_sums_below)
        
        X_e_val = np.array([1, frac_above_lim,])
        if frac_above_lim != 0:
            X_e_val = sm.add_constant(X_e_val)
            post_e_or = glm_e_results.predict(X_e_val)
            final_or = final_or * post_e_or[1]
        else:
            final_or = final_or * 1
            
        rounded_final = str(round((final_or-1)*100))
        
        if int(rounded_final) > 0:
            st.write("Based on exposure to psychological distress and environmental pollution, there is a " + rounded_final + "% increase in the odds of your child developing allergies. There are also many other possible factors that may contribute to atopy, such as family history of allergic disease. Explore the resources below and consult with your doctor for more information.")
            st.dataframe(resources,column_config={"Link": st.column_config.LinkColumn("Link")},hide_index=True)
        elif int(rounded_final) == 0:
            st.write("Based on exposure to psychological distress and environmental pollution, your child likely will not have an increased risk of atopy. However, there are many other possible factors that may contribute to atopy, such as family history of allergic disease. If answers to the above questions change, explore the resources below and consult with your doctor for more information.")
            st.dataframe(resources,column_config={"Link": st.column_config.LinkColumn("Link")},hide_index=True)
        else:
            st.write("something is wrong postnatal")


except:
    st.text(" ")
