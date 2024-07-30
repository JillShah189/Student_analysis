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

# Load the competency level data
competency_data = pd.read_csv('student_competency_levels (1).csv')

# Compute overall competency level mappings
student_competency = competency_data[['StudentProfileId','Score','CompetencyLevel']].copy()

# Rename 'Score' to 'AverageScore'
student_competency = student_competency.rename(columns={'Score': 'AverageScore'})

# Create a copy for class competency calculations
class_competency_copy = merged_data.copy()
class_competency = class_competency_copy.groupby(['ClassId', 'SectionId'])['Score'].mean().reset_index()
class_competency = class_competency.rename(columns={'Score': 'AverageScore'})

# Create a copy for school competency calculations
school_competency_copy = merged_data.copy()
school_competency = school_competency_copy.groupby('SchCd')['Score'].mean().reset_index()
school_competency = school_competency.rename(columns={'Score': 'AverageScore'})

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

# Add question categories to a new copy of merged data
merged_data_copy = merged_data.copy()
merged_data_copy['Category'] = merged_data_copy['QuestionNumber'].map(question_categories)

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
    # Create a copy for staff performance calculations
    staff_performance_copy = merged_data.copy()
    staff_performance = staff_performance_copy.groupby('StaffID')['Score'].mean().reset_index()
    staff_performance = staff_performance.rename(columns={'Score': 'AverageScore'})
    staff_performance['CompetencyLevel'] = staff_performance['AverageScore'].apply(get_competency_level)

    st.subheader('Average Score per Staff Member')
    st.dataframe(staff_performance)

    st.subheader('Count of Students Based on Competency Level per Staff')
    competency_count_staff = staff_performance['CompetencyLevel'].value_counts().reset_index()
    competency_count_staff.columns = ['CompetencyLevel', 'Count']
    fig5 = px.bar(competency_count_staff, x='CompetencyLevel', y='Count', title='Count of Students Based on Competency Level per Staff')
    st.plotly_chart(fig5)

    staff_count = staff_performance_copy.groupby('SchCd')['StaffID'].nunique().reset_index()
    staff_count = staff_count.rename(columns={'StaffID': 'StaffCount'})

    st.subheader('Count of Staff Members per School')
    st.dataframe(staff_count)

    # Calculate competency level count per school
    staff_competency_counts = staff_performance_copy.groupby(['SchCd', 'StaffID'])['Score'].mean().reset_index()
    staff_competency_counts['CompetencyLevel'] = staff_competency_counts['Score'].apply(get_competency_level)
    competency_counts_per_school = staff_competency_counts.groupby(['SchCd', 'CompetencyLevel'])['StaffID'].count().reset_index()
    competency_counts_per_school = competency_counts_per_school.rename(columns={'StaffID': 'Count'})

    st.subheader('Count of Competency Levels Among Staff Members per School')

    # Create an understandable graph
    fig7 = px.bar(competency_counts_per_school, x='SchCd', y='Count', color='CompetencyLevel', title='Count of Competency Levels Among Staff Members per School',
                 labels={'SchCd': 'School Code', 'Count': 'Number of Staff', 'CompetencyLevel': 'Competency Level'})

    fig7.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig7)
    st.dataframe(competency_counts_per_school)


# Question Analysis
if options == 'Question Analysis':
    st.header('Question Analysis')
    # Create a copy for question difficulty calculations
    question_difficulty_copy = merged_data.copy()
    question_difficulty = question_difficulty_copy.groupby('QuestionNumber')['Score'].mean().reset_index()
    question_difficulty = question_difficulty.rename(columns={'Score': 'AverageScore'})

    st.subheader('Difficulty Level of Each Question')
    fig6 = px.bar(question_difficulty, x='QuestionNumber', y='AverageScore', title='Difficulty Level of Each Question')
    st.plotly_chart(fig6)

# Category Analysis
if options == 'Category Analysis':
    st.header('Question Analysis by Category')

    st.subheader('Average Score per Category')
    category_scores_copy = merged_data_copy.copy()
    category_scores = category_scores_copy.groupby('Category')['Score'].mean().reset_index()
    category_scores = category_scores.rename(columns={'Score': 'AverageScore'})
    fig8 = px.bar(category_scores, x='Category', y='AverageScore', title='Average Score per Category')
    st.plotly_chart(fig8)

    st.subheader('Count of Students per Category')
    student_category_count = category_scores_copy['Category'].value_counts().reset_index()
    student_category_count.columns = ['Category', 'Count']
    fig9 = px.pie(student_category_count, names='Category', values='Count', title='Count of Students per Category')
    st.plotly_chart(fig9)

# School-wise KMeans Clustering
if options == 'School-wise Clustering':
    st.header('School-wise Clustering of Student Competency Levels')
    school_data_copy = merged_data.copy()

    # Impute missing values if any
    imputer = SimpleImputer(strategy='mean')
    school_data_copy['Score'] = imputer.fit_transform(school_data_copy[['Score']])

    # Prepare data for clustering
    X = school_data_copy[['SchCd', 'Score']].values

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    school_data_copy['Cluster'] = kmeans.fit_predict(X)

    st.subheader('KMeans Clustering of Schools based on Student Scores')
    fig10 = px.scatter(school_data_copy, x='SchCd', y='Score', color='Cluster', title='KMeans Clustering of Schools based on Student Scores')
    st.plotly_chart(fig10)
