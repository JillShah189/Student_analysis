import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Load the data from the two CSV files
file1 = pd.read_csv('student_answers_part1.csv')
file2 = pd.read_csv('student_answers_part2.csv')

# Merge the data
merged_data = pd.concat([file1, file2])

# Save the merged data to a new CSV file for convenience
merged_data.to_csv('merged_data.csv', index=False)

# Load the merged data
merged_data = pd.read_csv('merged_data.csv')

# Check the structure of merged_data
print("Columns in merged_data:", merged_data.columns)

# Check for missing values
print("Missing values in critical columns:")
print(merged_data[['ClassId', 'SectionId', 'Score']].isnull().sum())

# Ensure correct data types
merged_data['Score'] = pd.to_numeric(merged_data['Score'], errors='coerce')
merged_data['ClassId'] = pd.to_numeric(merged_data['ClassId'], errors='coerce')
merged_data['SectionId'] = pd.to_numeric(merged_data['SectionId'], errors='coerce')

# Handle missing values (optional)
merged_data.dropna(subset=['ClassId', 'SectionId', 'Score'], inplace=True)

# Load the competency level data
competency_data = pd.read_csv('student_competency_levels (1).csv')

# Compute overall competency level mappings
student_competency = competency_data[['StudentProfileId','Score','CompetencyLevel']]
student_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)

# Check the structure of student_competency
print("Columns in student_competency:", student_competency.columns)

# Correct class competency computation
class_competency = merged_data.groupby(['ClassId', 'SectionId'])['Score'].mean().reset_index()
class_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)

# Debugging information
print("Class Competency Data:")
print(class_competency)

school_competency = merged_data.groupby('SchCd')['Score'].mean().reset_index()
school_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)

student_competency = student_competency[['StudentProfileId', 'AverageScore']].copy()

# Define competency level function
def get_competency_level(score):
    if score <= 0.4:
        return 'Beginner'
    elif score <= 0.65:
        return 'Learner'
    elif score <= 0.8:
        return 'Skilled'
    elif score <= 0.95:
        return 'Expert'
    else:
        return 'Master'

school_competency['CompetencyLevel'] = school_competency['AverageScore'].apply(get_competency_level)

# Define question categories
question_categories = {
    1: 'Integers', 2: 'Integers', 3: 'Fractions and Decimals', 4: 'Comparing Quantities',
    5: 'Lines and Angles', 6: 'Integers', 7: 'Integers', 8: 'Integers', 9: 'Data Handling',
    10: 'Equations', 11: 'Fractions and Decimals', 12: 'Lines and Angles', 13: 'Equations',
    14: 'Comparing Quantities', 15: 'Integers', 16: 'Data Handling', 17: 'Equations',
    18: 'Fractions and Decimals', 19: 'The Triangle and Its Properties', 20: 'Integers',
    21: 'Integers', 22: 'Perimeter and Area', 23: 'Perimeter and Area', 24: 'Perimeter and Area',
    25: 'Equations', 26: 'Integers', 27: 'Equations', 28: 'Integers', 29: 'Integers', 30: 'Integers'
}

# Add question categories to the merged data
merged_data['Category'] = merged_data['QuestionNumber'].map(question_categories)

# Define the Streamlit app
st.title('Student Performance Analysis Dashboard')

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Go to', ['Home', 'Overall Competency', 'Staff Performance', 'Question Analysis', 'Category Analysis', 'School-wise Clustering'])

# Home Page
if options == 'Home':
    st.write("""
    ## Welcome to the Student Performance Analysis Dashboard
    This dashboard provides insights into student performance across various dimensions such as overall competency, staff performance, and more.
    Use the navigation sidebar to explore different analyses.
    """)

# Overall Competency Level Mapping
if options == 'Overall Competency':
    st.header('Overall Competency Level Mapping')

    st.subheader('Average Score per Student')
    st.dataframe(student_competency)

    st.subheader('Count of Students Based on Competency Level')
    competency_count = student_competency['CompetencyLevel'].value_counts().reset_index()
    competency_count.columns = ['CompetencyLevel', 'Count']
    fig2 = px.bar(competency_count, x='CompetencyLevel', y='Count', title='Count of Students Based on Competency Level')
    st.plotly_chart(fig2)

    st.subheader('Average Score per Class')
    st.dataframe(class_competency)

    st.subheader('Average Score per School')
    st.dataframe(school_competency)

    st.subheader('Count of Students Based on Competency Level per School')
    competency_count_school = school_competency['CompetencyLevel'].value_counts().reset_index()
    competency_count_school.columns = ['CompetencyLevel', 'Count']
    fig5 = px.bar(competency_count_school, x='CompetencyLevel', y='Count', title='Count of Students Based on Competency Level per School')
    st.plotly_chart(fig5)

# Staff Performance Analysis
if options == 'Staff Performance':
    st.header('Staff Performance Analysis')
    staff_performance = merged_data.groupby('StaffID')['Score'].mean().reset_index()
    staff_performance.rename(columns={'Score': 'AverageScore'}, inplace=True)
    staff_performance['CompetencyLevel'] = staff_performance['AverageScore'].apply(get_competency_level)

    st.subheader('Average Score per Staff Member')
    st.dataframe(staff_performance)

    st.subheader('Count of Students Based on Competency Level per Staff')
    competency_count_staff = staff_performance['CompetencyLevel'].value_counts().reset_index()
    competency_count_staff.columns = ['CompetencyLevel', 'Count']
    fig5 = px.bar(competency_count_staff, x='CompetencyLevel', y='Count', title='Count of Students Based on Competency Level per Staff')
    st.plotly_chart(fig5)

    staff_count = merged_data.groupby('SchCd')['StaffID'].nunique().reset_index()
    staff_count.rename(columns={'StaffID': 'StaffCount'}, inplace=True)

    st.subheader('Count of Staff Members per School')
    st.dataframe(staff_count)

    # Calculate competency level count per school
    staff_competency_counts = merged_data.groupby(['SchCd', 'StaffID'])['Score'].mean().reset_index()
    staff_competency_counts['CompetencyLevel'] = staff_competency_counts['Score'].apply(get_competency_level)
    competency_counts_per_school = staff_competency_counts.groupby(['SchCd', 'CompetencyLevel'])['StaffID'].count().reset_index()
    competency_counts_per_school.rename(columns={'StaffID': 'Count'}, inplace=True)

    st.subheader('Competency Level Counts per School')
    fig6 = px.bar(competency_counts_per_school, x='SchCd', y='Count', color='CompetencyLevel', barmode='group', title='Competency Level Counts per School')
    st.plotly_chart(fig6)

# Question Analysis
if options == 'Question Analysis':
    st.header('Question Analysis')
    question_performance = merged_data.groupby('QuestionNumber')['Score'].mean().reset_index()
    question_performance.rename(columns={'Score': 'AverageScore'}, inplace=True)
    question_performance['CompetencyLevel'] = question_performance['AverageScore'].apply(get_competency_level)

    st.subheader('Average Score per Question')
    st.dataframe(question_performance)

    st.subheader('Competency Level Distribution per Question')
    competency_count_question = question_performance['CompetencyLevel'].value_counts().reset_index()
    competency_count_question.columns = ['CompetencyLevel', 'Count']
    fig7 = px.bar(competency_count_question, x='CompetencyLevel', y='Count', title='Competency Level Distribution per Question')
    st.plotly_chart(fig7)

    question_count = merged_data.groupby('QuestionNumber')['StudentProfileId'].nunique().reset_index()
    question_count.rename(columns={'StudentProfileId': 'StudentCount'}, inplace=True)

    st.subheader('Count of Students Answering Each Question')
    st.dataframe(question_count)

# Category Analysis
if options == 'Category Analysis':
    st.header('Category Analysis')
    category_performance = merged_data.groupby('Category')['Score'].mean().reset_index()
    category_performance.rename(columns={'Score': 'AverageScore'}, inplace=True)
    category_performance['CompetencyLevel'] = category_performance['AverageScore'].apply(get_competency_level)

    st.subheader('Average Score per Category')
    st.dataframe(category_performance)

    st.subheader('Competency Level Distribution per Category')
    competency_count_category = category_performance['CompetencyLevel'].value_counts().reset_index()
    competency_count_category.columns = ['CompetencyLevel', 'Count']
    fig8 = px.bar(competency_count_category, x='CompetencyLevel', y='Count', title='Competency Level Distribution per Category')
    st.plotly_chart(fig8)

    category_count = merged_data.groupby('Category')['StudentProfileId'].nunique().reset_index()
    category_count.rename(columns={'StudentProfileId': 'StudentCount'}, inplace=True)

    st.subheader('Count of Students Answering Each Category')
    st.dataframe(category_count)

# School-wise Clustering
if options == 'School-wise Clustering':
    st.header('School-wise Clustering')
    X = school_competency[['AverageScore']]

    # Use KMeans for clustering
    kmeans = KMeans(n_clusters=3)
    school_competency['Cluster'] = kmeans.fit_predict(X)

    st.subheader('Clustered School Competency')
    fig9 = px.scatter(school_competency, x='SchCd', y='AverageScore', color='Cluster', title='School Competency Clustering')
    st.plotly_chart(fig9)

    st.subheader('Clustered Data')
    st.dataframe(school_competency[['SchCd', 'AverageScore', 'Cluster']])

