'''import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Load the data from the two CSV files
file1 = pd.read_csv('c:/Users/admin/Downloads/student_answers_part1.csv')
file2 = pd.read_csv('c:/Users/admin/Downloads/student_answers_part2.csv')

# Merge the data
merged_data = pd.concat([file1, file2])

# Save the merged data to a new CSV file for convenience
merged_data.to_csv('merged_data.csv', index=False)

# Load the merged data
merged_data = pd.read_csv('merged_data.csv')

# Load the competency level data
competency_data = pd.read_csv('c:/Users/admin/Downloads/student_competency_levels (1).csv')

# Compute overall competency level mappings
student_competency = competency_data[['StudentProfileId','Score','CompetencyLevel']]
#student_competency = merged_data.groupby('StudentProfileId')['Score'].mean().reset_index()
student_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)
class_competency = merged_data.groupby(['ClassId', 'SectionId'])['Score'].mean().reset_index()
class_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)
school_competency = merged_data.groupby('SchCd')['Score'].mean().reset_index()
school_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)

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
# Define the Streamlit app
st.title('Student Performance Analysis Dashboard')

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Go to', ['Home', 'Overall Competency', 'Staff Performance', 'Question Analysis', 'School-wise Clustering'])

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

    st.subheader('Count of Students Based on Competency Level per staff')
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
    question_difficulty = merged_data.groupby('QuestionNumber')['Score'].mean().reset_index()
    question_difficulty.rename(columns={'Score': 'AverageScore'}, inplace=True)

    st.subheader('Difficulty Level of Each Question')
    fig6 = px.bar(question_difficulty, x='QuestionNumber', y='AverageScore', title='Difficulty Level of Each Question')
    st.plotly_chart(fig6)


# School-wise Clustering
if options == 'School-wise Clustering':
    st.header('School-wise Clustering')
    school_competency = merged_data.groupby('SchCd')['Score'].mean().reset_index()
    school_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)

    kmeans = KMeans(n_clusters=3)
    school_competency['Cluster'] = kmeans.fit_predict(school_competency[['AverageScore']])

    st.subheader('Clustering of Schools Based on Performance')
    fig10 = px.scatter(school_competency, x='SchCd', y='AverageScore', color='Cluster', title='Clustering of Schools Based on Performance')
    st.plotly_chart(fig10)

    st.write("### Cluster Centers")
    st.write(kmeans.cluster_centers_)'''

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
student_competency = competency_data[['StudentProfileId','Score','CompetencyLevel']]
student_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)
class_competency = merged_data.groupby(['ClassId', 'SectionId'])['Score'].mean().reset_index()
class_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)
school_competency = merged_data.groupby('SchCd')['Score'].mean().reset_index()
student_competency = student_competency[['Score']].copy()
school_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)

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
    question_difficulty = merged_data.groupby('QuestionNumber')['Score'].mean().reset_index()
    question_difficulty.rename(columns={'Score': 'AverageScore'}, inplace=True)

    st.subheader('Difficulty Level of Each Question')
    fig6 = px.bar(question_difficulty, x='QuestionNumber', y='AverageScore', title='Difficulty Level of Each Question')
    st.plotly_chart(fig6)

# Category Analysis
if options == 'Category Analysis':
    st.header('Question Analysis by Category')

    st.subheader('Average Score per Category')
    category_scores = merged_data.groupby('Category')['Score'].mean().reset_index()
    category_scores.rename(columns={'Score': 'AverageScore'}, inplace=True)
    fig8 = px.bar(category_scores, x='AverageScore', y='Category', title='Average Score per Category')
    st.plotly_chart(fig8)

    st.subheader('Count of Questions per Category')
    category_counts = merged_data['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    fig9 = px.bar(category_counts, x='Count', y='Category', title='Count of Questions per Category')
    st.plotly_chart(fig9)

    st.subheader('Count of Questions per Category')
    category_counts = merged_data['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    fig9 = px.pie(category_counts, values='Count', names='Category', title='Count of Questions per Category')
    st.plotly_chart(fig9)

    # Allow user to select the threshold score using a slider
    threshold_score = st.slider('Select Threshold Score', min_value=0.0, max_value=1.0, value=0.4, step=0.1)

    try:
        # Step 1: Calculate the average score of each school per category
        school_category_scores = merged_data.groupby(['SchCd', 'Category'])['Score'].mean().reset_index()
        school_category_scores.rename(columns={'Score': 'AverageScore'}, inplace=True)
        
        # Step 2: Filter out schools with an average score above the threshold
        best_performing_schools = school_category_scores[school_category_scores['AverageScore'] >= threshold_score]
        
        # Step 3: Count the number of schools that meet the threshold for each category
        category_counts = best_performing_schools['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'BestSchoolsCount']
        
        # Verify the intermediate DataFrames
        st.subheader('Average Score of each School per Category')
        st.write(school_category_scores)
        st.subheader(f'The Better Performing Schools(threshold > {threshold_score})')
        st.write(best_performing_schools)
        st.subheader('Count of schools')
        st.write(category_counts)

        # Step 4: Check if category_counts is not empty before plotting
        if not category_counts.empty:
            # Plotting a pie chart to show the percentage of schools that performed best in each category
            fig14 = px.pie(category_counts, values='BestSchoolsCount', names='Category',
                        title=f'Percentage of Schools Performing Best in Each Category (Threshold Score >= {threshold_score})',
                        hole=0.3)  # Adding a hole in the middle for better visualization

            # Display the pie chart in Streamlit
            st.plotly_chart(fig14)
        else:
            st.write("No data to display")
    except Exception as e:
        st.write(f"An error occurred: {e}")

    try:
        # Step 1: Calculate the average score of each school per category
        staff_category_scores = merged_data.groupby(['StaffID', 'Category'])['Score'].mean().reset_index()
        staff_category_scores.rename(columns={'Score': 'AverageScore'}, inplace=True)
        
        # Step 2: Filter out schools with an average score above the threshold
        best_performing_staffs = staff_category_scores[staff_category_scores['AverageScore'] >= threshold_score]
        
        # Step 3: Count the number of schools that meet the threshold for each category
        category_counts = best_performing_staffs['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'BestStaffCount']
        
        # Verify the intermediate DataFrames
        st.subheader('Average Score of each Staff per Category')
        st.write(staff_category_scores)
        st.subheader(f'The Better Performing Staffs(threshold > {threshold_score})')
        st.write(best_performing_staffs)
        st.subheader('Count of schools')
        st.write(category_counts)

        # Step 4: Check if category_counts is not empty before plotting
        if not category_counts.empty:
            # Plotting a pie chart to show the percentage of schools that performed best in each category
            fig14 = px.pie(category_counts, values='BestStaffCount', names='Category',
                        title=f'Percentage of Staffs Performing Best in Each Category (Threshold Score >= {threshold_score})',
                        hole=0.3)  # Adding a hole in the middle for better visualization

            # Display the pie chart in Streamlit
            st.plotly_chart(fig14)
        else:
            st.write("No data to display")
    except Exception as e:
        st.write(f"An error occurred: {e}")

# School-wise Clustering
if options == 'School-wise Clustering':
    st.header('School-wise Clustering')
    school_competency = merged_data.groupby('SchCd')['Score'].mean().reset_index()
    school_competency.rename(columns={'Score': 'AverageScore'}, inplace=True)

    kmeans = KMeans(n_clusters=3)
    school_competency['Cluster'] = kmeans.fit_predict(school_competency[['AverageScore']])

    st.subheader('Clustering of Schools Based on Performance')
    fig10 = px.scatter(school_competency, x='SchCd', y='AverageScore', color='Cluster', title='Clustering of Schools Based on Performance')
    st.plotly_chart(fig10)

    st.write("### Cluster Centers")
    st.write(kmeans.cluster_centers_)

