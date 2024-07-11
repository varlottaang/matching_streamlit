import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud
import seaborn as sns
from collections import defaultdict
import random

# Initialize the sentence transformer model (the engine)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize session state for storing uploaded files and results
# Session state precente to lose files

for state in ['mentors_df', 'mentees_df', 'all_matches_df', 'groups_df', 'reassigned_groups_df', 'best_matches_df', 'updated_groups_df']:
    if state not in st.session_state:
        st.session_state[state] = None

######## load files
def load_files():
    st.sidebar.header('Upload your CSV files')
    uploaded_mentors = st.sidebar.file_uploader("Upload Mentors CSV", type=['csv'], key="mentors_csv")
    uploaded_mentees = st.sidebar.file_uploader("Upload Mentees CSV", type=['csv'], key="mentees_csv")
    
    if uploaded_mentors is not None and uploaded_mentees is not None:
        mentors_df = pd.read_csv(uploaded_mentors)
        mentees_df = pd.read_csv(uploaded_mentees)
        
        st.session_state['mentors_df'] = mentors_df
        st.session_state['mentees_df'] = mentees_df
        return mentors_df, mentees_df
    else:
        return st.session_state['mentors_df'], st.session_state['mentees_df']

#####################
#####################
# EMBEDDINGS

def prepare_mentee_embeddings(mentees_df):
    def adjust_and_combine_keywords(row):
        weighted_preferences = f"{row['preference1']} " * 1 + f"{row['preference2']} " * 1 + f"{row['preference3']}" * 1
        return weighted_preferences.strip()
    mentees_df['weighted_preferences'] = mentees_df.apply(adjust_and_combine_keywords, axis=1)
    return mentees_df

############
###########
# PART 1 MATCHING

def part1(mentors_df, mentees_df):
    mentees_df = prepare_mentee_embeddings(mentees_df)

    if 'keywords' not in mentors_df.columns or 'Senior' not in mentors_df.columns or 'Senior' not in mentees_df.columns:
        st.error("The 'keywords' or 'Senior' column is missing in the dataframes.")
        return

    mentors_df['Senior'] = mentors_df['Senior'].apply(lambda x: x.strip().lower() == 'senior')
    mentees_df['Senior'] = mentees_df['Senior'].apply(lambda x: x.strip().lower() == 'senior')

    mentor_embeddings = model.encode(mentors_df['keywords'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    mentee_weighted_embeddings = model.encode(mentees_df['weighted_preferences'].tolist(), convert_to_tensor=True, show_progress_bar=True)

    all_matches_df = pd.DataFrame(columns=['Mentee', 'Mentor', 'Timezone', 'Similarity Score'])

    for timezone in mentors_df['timezone'].unique():
        filtered_mentors = mentors_df[mentors_df['timezone'] == timezone]
        filtered_mentees = mentees_df[mentees_df['timezone'] == timezone]

        if filtered_mentors.empty or filtered_mentees.empty:
            continue

        filtered_mentor_embeddings = model.encode(filtered_mentors['keywords'].tolist(), convert_to_tensor=True)
        filtered_mentee_embeddings = model.encode(filtered_mentees['weighted_preferences'].tolist(), convert_to_tensor=True)
        mentor_assignments = np.zeros(len(filtered_mentors))
        matches_with_scores = []

        for mentee_idx, mentee_embedding in enumerate(filtered_mentee_embeddings):
            similarities = util.cos_sim(mentee_embedding, filtered_mentor_embeddings)[0]
            mentors_priority = sorted(
                range(len(filtered_mentors)),
                key=lambda x: (mentor_assignments[x], -similarities[x].item())
            )

            for mentor_idx in mentors_priority:
                if mentor_assignments[mentor_idx] < np.ceil(len(filtered_mentees) / len(filtered_mentors)):
                    mentor_name = filtered_mentors.iloc[mentor_idx]['name_id']
                    mentee_name = filtered_mentees.iloc[mentee_idx]['name_id']
                    similarity_score = similarities[mentor_idx].item()

                    matches_with_scores.append((mentee_name, mentor_name, timezone, similarity_score))
                    mentor_assignments[mentor_idx] += 1
                    break

        current_matches_df = pd.DataFrame(matches_with_scores, columns=['Mentee', 'Mentor', 'Timezone', 'Similarity Score'])
        all_matches_df = pd.concat([all_matches_df, current_matches_df], ignore_index=True)

    if all_matches_df.empty:
        st.warning("No matches found. Please check your data and try again.")
        return

    # Check for seniority mismatches
    mismatches = []
    for _, row in all_matches_df.iterrows():
        mentee_name = row['Mentee']
        mentor_name = row['Mentor']
        mentee_is_senior = mentees_df[mentees_df['name_id'] == mentee_name]['Senior'].values[0]
        mentor_is_senior = mentors_df[mentors_df['name_id'] == mentor_name]['Senior'].values[0]

        if mentee_is_senior and not mentor_is_senior:
            mismatches.append((mentee_name, row['Timezone']))

    # Reassign mismatched mentees with similarity score consideration
    random.seed(42)
    mentee_assignments = defaultdict(list)
    mentor_assignments = defaultdict(list)

    for _, row in all_matches_df.iterrows():
        mentor_name = row['Mentor']
        mentee_name = row['Mentee']
        mentor_assignments[mentor_name].append((mentee_name, row['Similarity Score']))

    for mentee, timezone in mismatches:
        mentee_embedding = model.encode([mentees_df[mentees_df['name_id'] == mentee]['weighted_preferences'].values[0]], convert_to_tensor=True)
        available_mentors = mentors_df[(mentors_df['timezone'] == timezone) & (mentors_df['Senior'] == True)]

        if not available_mentors.empty:
            available_mentor_embeddings = model.encode(available_mentors['keywords'].tolist(), convert_to_tensor=True)
            similarities = util.cos_sim(mentee_embedding, available_mentor_embeddings)[0]
            best_mentor_idx = similarities.argmax().item()
            best_mentor = available_mentors.iloc[best_mentor_idx]['name_id']
            similarity_score = similarities[best_mentor_idx].item()

            # Find the current mentor
            current_mentor = next((mentor for mentor, mentees in mentor_assignments.items() if mentee in [m[0] for m in mentees]), None)
            if current_mentor:
                mentor_assignments[current_mentor] = [m for m in mentor_assignments[current_mentor] if m[0] != mentee]
            mentor_assignments[best_mentor].append((mentee, similarity_score))

    new_mentor_mentee_groups = []
    for mentor, mentees in mentor_assignments.items():
        timezone = mentors_df[mentors_df['name_id'] == mentor]['timezone'].values[0]
        mentees_with_scores = [{'Mentee': m[0], 'Similarity Score': m[1]} for m in mentees]
        new_mentor_mentee_groups.append({'Mentor': mentor, 'Timezone': timezone, 'Matched Mentees': mentees_with_scores})

    new_mentor_mentee_groups_df = pd.DataFrame(new_mentor_mentee_groups)

    new_mentor_mentee_groups_df['Matched Mentees'] = new_mentor_mentee_groups_df['Matched Mentees'].apply(str)
    new_mentor_mentee_groups_df.to_csv('mentor_mentee_groups.csv', index=False)

    st.session_state['all_matches_df'] = all_matches_df
    st.session_state['groups_df'] = new_mentor_mentee_groups_df

    st.success("Matching complete. Results saved to 'mentee_mentor_matches.csv' and 'mentor_mentee_groups.csv'.")
    st.dataframe(all_matches_df)
    st.dataframe(new_mentor_mentee_groups_df)
    st.download_button(label="Download Matches CSV", data=all_matches_df.to_csv(index=False), file_name='mentee_mentor_matches.csv', key="download_matches_1")
    st.download_button(label="Download Groups CSV", data=new_mentor_mentee_groups_df.to_csv(index=False), file_name='mentor_mentee_groups.csv', key="download_groups_1")






#####################################################
##################
#################
# PART 2 REASSIGNMENT

def part2(mentors_df, mentees_df, mentor_mentee_groups_df, leaving_mentors):
    st.write("mentor_mentee_groups_df columns:", mentor_mentee_groups_df.columns.tolist())

    mentees_df = prepare_mentee_embeddings(mentees_df)
    leaving_mentors = [mentor.strip() for mentor in leaving_mentors.split(',')]

    mentees_to_reassign = []
    for _, row in mentor_mentee_groups_df.iterrows():
        mentor = row['Mentor']
        if mentor in leaving_mentors:
            mentees_to_reassign.extend(eval(row['Matched Mentees']))

    all_matches_df = pd.DataFrame(columns=['Mentee', 'Mentor', 'Timezone', 'Similarity Score'])

    for mentee in mentees_to_reassign:
        mentee_info = mentees_df[mentees_df['name_id'] == mentee]
        if not mentee_info.empty:
            mentee_info = mentee_info.iloc[0]
            mentee_embedding = model.encode([mentee_info['weighted_preferences']], convert_to_tensor=True)
            mentee_timezone = mentee_info['timezone']
            mentee_is_senior = mentee_info['Senior'].strip().lower() == 'senior'

            available_mentors = mentors_df[(mentors_df['timezone'] == mentee_timezone) & 
                                           (~mentors_df['name_id'].isin(leaving_mentors))]
            if mentee_is_senior:
                available_mentors = available_mentors[available_mentors['Senior'].str.strip().str.lower() == 'senior']

            if not available_mentors.empty:
                available_mentor_embeddings = model.encode(available_mentors['keywords'].tolist(), convert_to_tensor=True)
                similarities = util.cos_sim(mentee_embedding, available_mentor_embeddings)[0]

                mentors_priority = sorted(
                    range(len(available_mentors)),
                    key=lambda x: -similarities[x].item()
                )

                best_mentor_idx = mentors_priority[0]
                best_mentor = available_mentors.iloc[best_mentor_idx]['name_id']
                similarity_score = similarities[best_mentor_idx].item()

                new_match = pd.DataFrame({
                    'Mentee': [mentee],
                    'Mentor': [best_mentor],
                    'Timezone': [mentee_timezone],
                    'Similarity Score': [similarity_score]
                })
                all_matches_df = pd.concat([all_matches_df, new_match], ignore_index=True)

    mentor_columns = ['name_id', 'email', 'First Name', 'Last Name']
    mentee_columns = ['name_id', 'email']

    all_matches_df = all_matches_df.merge(
        mentors_df[mentor_columns], left_on='Mentor', right_on='name_id', suffixes=('', '_mentor')
    ).drop('name_id', axis=1)

    all_matches_df = all_matches_df.merge(
        mentees_df[mentee_columns], left_on='Mentee', right_on='name_id', suffixes=('', '_mentee')
    ).drop('name_id', axis=1)

    groupby_columns = ['Mentor', 'email', 'Timezone', 'First Name', 'Last Name']
    aggregate_columns = ['Mentee', 'email_mentee']

    groups_df = all_matches_df.groupby(groupby_columns)[aggregate_columns].agg(list).reset_index()
    groups_df.rename(columns={'email': 'Mentor Email', 'email_mentee': 'Mentees Email', 'Mentee': 'Matched Mentees'}, inplace=True)
    groups_df.to_csv('reassigned_mentor_mentee_groups.csv', index=False)
    st.session_state['reassigned_groups_df'] = groups_df

    st.success("Reassignment complete. Results saved to 'reassigned_mentor_mentee_groups.csv'.")
    st.dataframe(groups_df)
    st.download_button(label="Download Reassigned Groups CSV", data=groups_df.to_csv(index=False), file_name='reassigned_mentor_mentee_groups.csv', key="download_reassigned_groups_1")

###################################################################################################################
########################
# PART 3 FIND BEST MATCH

def part3(mentors_df, mentees_df, leaving_mentors_input):
    mentees_df = prepare_mentee_embeddings(mentees_df)
    leaving_mentors = [mentor.strip() for mentor in leaving_mentors_input.split(',')] if leaving_mentors_input.strip().lower() != 'none' else []

    if 'keywords' not in mentors_df.columns:
        st.error("The 'keywords' column is missing in mentors_df.")
        return

    mentor_embeddings = model.encode(mentors_df['keywords'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    mentee_weighted_embeddings = model.encode(mentees_df['weighted_preferences'].tolist(), convert_to_tensor=True, show_progress_bar=True)

    def find_best_match(mentee_embedding, filtered_mentors, filtered_mentor_embeddings):
        similarities = util.cos_sim(mentee_embedding, filtered_mentor_embeddings)[0]
        best_mentor_idx = similarities.argmax().item()
        best_match = filtered_mentors.iloc[best_mentor_idx]
        return {
            'Mentor ID': best_match['name_id'],
            'Mentor First Name': best_match['First Name'] if 'First Name' in best_match else '',
            'Mentor Last Name': best_match['Last Name'] if 'Last Name' in best_match else '',
            'Mentor Email': best_match['email'],
            'Similarity Score': similarities[best_mentor_idx].item()
        }

    final_matches = []
    for i, mentee_row in mentees_df.iterrows():
        mentee_name = mentee_row['name_id']
        mentee_embedding = model.encode([mentee_row['weighted_preferences']], convert_to_tensor=True)
        mentee_timezone = mentee_row['timezone']
        mentee_is_senior = mentee_row['Senior'].strip().lower() == 'senior'

        filtered_mentors = mentors_df[(mentors_df['timezone'] == mentee_timezone) & (~mentors_df['name_id'].isin(leaving_mentors))]
        if mentee_is_senior:
            senior_mentors = filtered_mentors[filtered_mentors['Senior'].str.strip().str.lower() == 'senior']
            if not senior_mentors.empty:
                filtered_mentors = senior_mentors

        filtered_mentor_embeddings = mentor_embeddings[[mentors_df.index.get_loc(index) for index in filtered_mentors.index]]
        best_match = find_best_match(mentee_embedding, filtered_mentors, filtered_mentor_embeddings)
        final_matches.append({
            'Mentee ID': mentee_name,
            'Mentee First Name': mentee_row['first_name'] if 'first_name' in mentee_row else '',
            'Mentee Last Name': mentee_row['last_name'] if 'last_name' in mentee_row else '',
            'Mentee Email': mentee_row['email'],
            'Mentor ID': best_match['Mentor ID'],
            'Mentor First Name': best_match['Mentor First Name'],
            'Mentor Last Name': best_match['Mentor Last Name'],
            'Mentor Email': best_match['Mentor Email'],
            'Similarity Score': best_match['Similarity Score'],
            'Timezone': mentee_timezone
        })

    final_matches_df = pd.DataFrame(final_matches)
    final_matches_df.to_csv("best_mentee_mentor_matches.csv", index=False)
    st.session_state['best_matches_df'] = final_matches_df

    st.success("Best matches found. Results saved to 'best_mentee_mentor_matches.csv'.")
    st.dataframe(final_matches_df)
    st.download_button(label="Download Best Matches CSV", data=final_matches_df.to_csv(index=False), file_name='best_mentee_mentor_matches.csv', key="download_best_matches_2")


###################################################################################################################
########################
# PART 4: ADD NEW MENTEES TO EXISTING GROUPS

def part4(mentors_df, new_mentees_df, mentor_mentee_groups_df):
    new_mentees_df = prepare_mentee_embeddings(new_mentees_df)

    if 'keywords' not in mentors_df.columns or 'Senior' not in mentors_df.columns or 'Senior' not in new_mentees_df.columns:
        st.error("The 'keywords' or 'Senior' column is missing in the dataframes.")
        return

    mentors_df['Senior'] = mentors_df['Senior'].apply(lambda x: x.strip().lower() == 'senior')
    new_mentees_df['Senior'] = new_mentees_df['Senior'].apply(lambda x: x.strip().lower() == 'senior')

    mentor_embeddings = model.encode(mentors_df['keywords'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    new_mentee_embeddings = model.encode(new_mentees_df['weighted_preferences'].tolist(), convert_to_tensor=True, show_progress_bar=True)

    all_new_matches_df = pd.DataFrame(columns=['Mentee', 'Mentor', 'Timezone', 'Similarity Score'])

    for timezone in mentors_df['timezone'].unique():
        filtered_mentors = mentors_df[mentors_df['timezone'] == timezone]
        filtered_new_mentees = new_mentees_df[new_mentees_df['timezone'] == timezone]

        if filtered_mentors.empty or filtered_new_mentees.empty:
            continue

        filtered_mentor_embeddings = model.encode(filtered_mentors['keywords'].tolist(), convert_to_tensor=True)
        filtered_new_mentee_embeddings = model.encode(filtered_new_mentees['weighted_preferences'].tolist(), convert_to_tensor=True)
        mentor_assignments = np.zeros(len(filtered_mentors))
        matches_with_scores = []

        for mentee_idx, mentee_embedding in enumerate(filtered_new_mentee_embeddings):
            mentee_is_senior = filtered_new_mentees.iloc[mentee_idx]['Senior']
            eligible_mentors = filtered_mentors if not mentee_is_senior else filtered_mentors[filtered_mentors['Senior']]
            if eligible_mentors.empty:
                continue

            eligible_mentor_embeddings = model.encode(eligible_mentors['keywords'].tolist(), convert_to_tensor=True)
            similarities = util.cos_sim(mentee_embedding, eligible_mentor_embeddings)[0]
            mentors_priority = sorted(
                range(len(eligible_mentors)),
                key=lambda x: (mentor_assignments[eligible_mentors.index.get_loc(eligible_mentors.index[x])], -similarities[x].item())
            )

            for mentor_idx in mentors_priority:
                mentor_loc = eligible_mentors.index.get_loc(eligible_mentors.index[mentor_idx])
                if mentor_assignments[mentor_loc] < np.ceil(len(filtered_new_mentees) / len(filtered_mentors)):
                    mentor_name = eligible_mentors.iloc[mentor_idx]['name_id']
                    mentee_name = filtered_new_mentees.iloc[mentee_idx]['name_id']
                    similarity_score = similarities[mentor_idx].item()

                    matches_with_scores.append((mentee_name, mentor_name, timezone, similarity_score))
                    mentor_assignments[mentor_loc] += 1
                    break

        current_matches_df = pd.DataFrame(matches_with_scores, columns=['Mentee', 'Mentor', 'Timezone', 'Similarity Score'])
        all_new_matches_df = pd.concat([all_new_matches_df, current_matches_df], ignore_index=True)

    # Merge additional mentor and mentee details for better visibility
    mentor_columns = ['name_id', 'email', 'First Name', 'Last Name']
    mentee_columns = ['name_id', 'email']

    all_new_matches_df = all_new_matches_df.merge(
        mentors_df[mentor_columns], left_on='Mentor', right_on='name_id', suffixes=('', '_mentor')
    ).drop('name_id', axis=1)

    all_new_matches_df = all_new_matches_df.merge(
        new_mentees_df[mentee_columns], left_on='Mentee', right_on='name_id', suffixes=('', '_mentee')
    ).drop('name_id', axis=1)

    # Append new matches to the existing groups and track updated groups
    mentor_mentee_groups_df['Matched Mentees'] = mentor_mentee_groups_df['Matched Mentees'].apply(eval)
    mentor_mentee_groups_df['Mentees Email'] = mentor_mentee_groups_df['Mentees Email'].apply(eval)
    updated_groups = mentor_mentee_groups_df.copy()

    for _, match in all_new_matches_df.iterrows():
        mentor = match['Mentor']
        mentee = match['Mentee']
        mentee_email = match['email_mentee']
        mentor_mentee_groups_df.loc[mentor_mentee_groups_df['Mentor'] == mentor, 'Matched Mentees'].apply(lambda x: x.append(mentee))
        mentor_mentee_groups_df.loc[mentor_mentee_groups_df['Mentor'] == mentor, 'Mentees Email'].apply(lambda x: x.append(mentee_email))

    # Identify the groups where new mentees have been added
    updated_groups['New Mentees Added'] = updated_groups.apply(lambda row: any(mentee in row['Matched Mentees'] for mentee in all_new_matches_df['Mentee'].tolist()), axis=1)
    new_mentees_added_df = updated_groups[updated_groups['New Mentees Added']].drop(columns=['New Mentees Added'])

    mentor_mentee_groups_df['Matched Mentees'] = mentor_mentee_groups_df['Matched Mentees'].apply(str)
    mentor_mentee_groups_df['Mentees Email'] = mentor_mentee_groups_df['Mentees Email'].apply(str)
    new_mentees_added_df['Matched Mentees'] = new_mentees_added_df['Matched Mentees'].apply(str)
    new_mentees_added_df['Mentees Email'] = new_mentees_added_df['Mentees Email'].apply(str)

    mentor_mentee_groups_df.to_csv('updated_mentor_mentee_groups.csv', index=False)
    new_mentees_added_df.to_csv('new_mentees_added_groups.csv', index=False)
    st.session_state['updated_groups_df'] = mentor_mentee_groups_df
    st.session_state['new_mentees_added_groups_df'] = new_mentees_added_df

    st.success("New mentees added. Updated groups saved to 'updated_mentor_mentee_groups.csv'. Groups with new mentees saved to 'new_mentees_added_groups.csv'.")
    st.dataframe(mentor_mentee_groups_df)
    st.dataframe(new_mentees_added_df)
    st.download_button(label="Download Updated Groups CSV", data=mentor_mentee_groups_df.to_csv(index=False), file_name='updated_mentor_mentee_groups.csv', key="download_updated_groups_1")
    st.download_button(label="Download New Mentees Added Groups CSV", data=new_mentees_added_df.to_csv(index=False), file_name='new_mentees_added_groups.csv', key="download_new_mentees_added_groups")


###################################################################################################################
########################
# PART 5: DATA VISUALIZATION

# Function to save plots to a PDF
def save_plots_to_pdf(pdf_path, mentor_words, mentee_words, combined_words, timezone_counts):
    with PdfPages(pdf_path) as pdf:
        # Mentor word cloud
        mentor_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(mentor_words)
        plt.figure(figsize=(10, 5))
        plt.title('Mentor Word Cloud')
        plt.imshow(mentor_wordcloud, interpolation='bilinear')
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # Mentee word cloud
        mentee_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(mentee_words)
        plt.figure(figsize=(10, 5))
        plt.title('Mentee Word Cloud')
        plt.imshow(mentee_wordcloud, interpolation='bilinear')
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # Combined word cloud
        combined_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_words)
        plt.figure(figsize=(10, 5))
        plt.title('Combined Mentor and Mentee Word Cloud')
        plt.imshow(combined_wordcloud, interpolation='bilinear')
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # Percentage of mentors and mentees across timezones
        plt.figure(figsize=(10, 10))
        plt.title('Percentage of Mentors and Mentees across Timezones')
        plt.pie(timezone_counts, labels=timezone_counts.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        pdf.savefig()
        plt.close()

        if 'groups_df' in st.session_state:
            groups_df = st.session_state['groups_df']
            # General Mentor/Mentees average group size
            avg_group_size = groups_df['Matched Mentees'].apply(len).mean()
            plt.figure()
            plt.title('Average Group Size')
            plt.text(0.5, 0.5, f"Average Group Size: {avg_group_size:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            # Average group size for each timezone
            avg_group_size_tz = groups_df.groupby('Timezone')['Matched Mentees'].apply(lambda x: np.mean([len(i) for i in x]))
            plt.figure()
            avg_group_size_tz.plot(kind='bar')
            plt.title('Average Group Size by Timezone')
            plt.ylabel('Average Group Size')
            pdf.savefig()
            plt.close()

            # Biggest group size for each timezone
            max_group_size_tz = groups_df.groupby('Timezone')['Matched Mentees'].apply(lambda x: max([len(i) for i in x]))
            plt.figure()
            max_group_size_tz.plot(kind='bar')
            plt.title('Biggest Group Size by Timezone')
            plt.ylabel('Group Size')
            pdf.savefig()
            plt.close()

            # Smallest group size for each timezone
            min_group_size_tz = groups_df.groupby('Timezone')['Matched Mentees'].apply(lambda x: min([len(i) for i in x]))
            plt.figure()
            min_group_size_tz.plot(kind='bar')
            plt.title('Smallest Group Size by Timezone')
            plt.ylabel('Group Size')
            pdf.savefig()
            plt.close()

### PART 5 ...
def part5(mentors_df, mentees_df, groups_df=None):
    st.header("Data Visualization")

    # Optional: Request to upload the groups CSV
    st.subheader("Upload Groups CSV (optional)")
    groups_file = st.file_uploader("", type=['csv'], key="groups_csv")
    if groups_file is not None:
        groups_df = pd.read_csv(groups_file)
        st.session_state['groups_df'] = groups_df

    if mentors_df is not None and mentees_df is not None:
        # 5. Mentor word cloud
        st.subheader("Mentor Word Cloud")
        mentor_words = ' '.join(mentors_df['keywords'].astype(str))
        mentor_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(mentor_words)
        plt.figure(figsize=(10, 5))
        plt.title('Mentor Word Cloud')
        plt.imshow(mentor_wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # 6. Mentees word cloud
        st.subheader("Mentee Word Cloud")
        mentee_words = ' '.join(mentees_df['preference1'].astype(str) + ' ' +
                                mentees_df['preference2'].astype(str) + ' ' +
                                mentees_df['preference3'].astype(str))
        mentee_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(mentee_words)
        plt.figure(figsize=(10, 5))
        plt.title('Mentee Word Cloud')
        plt.imshow(mentee_wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # 7. Combined mentors and mentees word cloud
        st.subheader("Combined Mentor and Mentee Word Cloud")
        combined_words = mentor_words + ' ' + mentee_words
        combined_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_words)
        plt.figure(figsize=(10, 5))
        plt.title('Combined Mentor and Mentee Word Cloud')
        plt.imshow(combined_wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # 8. Percentage of mentors and mentees across timezones
        st.subheader("Percentage of Mentors and Mentees across Timezones")
        timezone_counts = pd.concat([mentors_df['timezone'], mentees_df['timezone']], axis=0).value_counts(normalize=True) * 100
        plt.figure(figsize=(10, 10))
        plt.title('Percentage of Mentors and Mentees across Timezones')
        plt.pie(timezone_counts, labels=timezone_counts.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        st.pyplot(plt)

        if groups_df is not None:
            # Convert 'Matched Mentees' from string representation of lists to actual lists
            groups_df['Matched Mentees'] = groups_df['Matched Mentees'].apply(eval)

            # 1. General Mentor/Mentees average group size
            avg_group_size = groups_df['Matched Mentees'].apply(len).mean()
            st.subheader(f"Average Group Size: {avg_group_size:.2f}")

            # 2. Average group size for each timezone
            avg_group_size_tz = groups_df.groupby('Timezone')['Matched Mentees'].apply(lambda x: np.mean([len(i) for i in x]))
            st.subheader("Average Group Size by Timezone")
            st.dataframe(avg_group_size_tz)

            # 3. Biggest group size for each timezone
            max_group_size_tz = groups_df.groupby('Timezone')['Matched Mentees'].apply(lambda x: max([len(i) for i in x]))
            st.subheader("Biggest Group Size by Timezone")
            st.dataframe(max_group_size_tz)

            # 4. Smallest group size for each timezone
            min_group_size_tz = groups_df.groupby('Timezone')['Matched Mentees'].apply(lambda x: min([len(i) for i in x]))
            st.subheader("Smallest Group Size by Timezone")
            st.dataframe(min_group_size_tz)

        # Save visualizations to PDF
        pdf_path = '/tmp/visualizations.pdf'
        save_plots_to_pdf(pdf_path, mentor_words, mentee_words, combined_words, timezone_counts)
        with open(pdf_path, "rb") as f:
            st.download_button(label="Download Visualizations as PDF", data=f, file_name="visualizations.pdf", mime="application/pdf")
    else:
        st.warning("Please upload both mentors and mentees CSV files first.")


################################################################################################################
###############
### WEB INTERFACE

# Interface setup
st.set_page_config(page_title="Mentorship Matching", layout="wide")
st.title("Mentorship Matching App 🧑🏿‍🏫👨🏻‍🏫👩‍🏫🧑🏻‍🎓👨‍🎓👩🏽‍🎓")
mentors_df, mentees_df = load_files()

st.sidebar.markdown("[Breakout 🎶](https://yewtu.be/watch?v=IIOJdMdS56k)")

if mentors_df is not None and mentees_df is not None:
    st.sidebar.success("Files successfully uploaded.")
    page = st.sidebar.selectbox("Choose a part to proceed with", ["Part 1: Initial Matching", "Part 2: Mentees Reassignment", "Part 3: Best Matches", "Part 4: Add New Mentees to Existing Groups", "Part 5: Data Visualization"])

    if page == "Part 1: Initial Matching":
        st.header("Part 1: Initial Matching")
        if st.button("Run Initial Matching"):
            part1(mentors_df, mentees_df)
        if st.session_state['all_matches_df'] is not None and st.session_state['groups_df'] is not None:
            st.dataframe(st.session_state['all_matches_df'])
            st.dataframe(st.session_state['groups_df'])
            st.download_button(label="Download Matches CSV", data=st.session_state['all_matches_df'].to_csv(index=False), file_name='mentee_mentor_matches.csv', key="download_matches_2")
            st.download_button(label="Download Groups CSV", data=st.session_state['groups_df'].to_csv(index=False), file_name='mentor_mentee_groups.csv', key="download_groups_2")

    elif page == "Part 2: Mentees Reassignment":
        st.header("Part 2: Mentees Reassignment")
        groups_file = st.file_uploader("Upload Initial Groups CSV", type=['csv'], key="initial_groups_csv")
        if groups_file is not None:
            mentor_mentee_groups_df = pd.read_csv(groups_file)
            leaving_mentors = st.text_input("Enter leaving mentors' name_id, separated by commas")
            if st.button("Reassign Mentees"):
                part2(mentors_df, mentees_df, mentor_mentee_groups_df, leaving_mentors)
        if st.session_state['reassigned_groups_df'] is not None:
            st.dataframe(st.session_state['reassigned_groups_df'])
            st.download_button(label="Download Reassigned Groups CSV", data=st.session_state['reassigned_groups_df'].to_csv(index=False), file_name='reassigned_mentor_mentee_groups.csv', key="download_reassigned_groups_2")

    elif page == "Part 3: Best Matches":
        st.header("Part 3: Best Matches")
        leaving_mentors_input = st.text_input("Enter leaving mentors' name_id, separated by commas (or 'None')")
        if st.button("Find Best Matches"):
            part3(mentors_df, mentees_df, leaving_mentors_input)
        if st.session_state['best_matches_df'] is not None:
            st.dataframe(st.session_state['best_matches_df'])
            st.download_button(label="Download Best Matches CSV", data=st.session_state['best_matches_df'].to_csv(index=False), file_name='best_mentee_mentor_matches.csv', key="download_best_matches_2")

    elif page == "Part 4: Add New Mentees to Existing Groups":
        st.header("Part 4: Add New Mentees to Existing Groups")
        new_mentees_file = st.file_uploader("Upload New Mentees CSV", type=['csv'], key="new_mentees_csv")
        groups_file = st.file_uploader("Upload Existing Groups CSV", type=['csv'], key="existing_groups_csv")
        if new_mentees_file is not None and groups_file is not None:
            new_mentees_df = pd.read_csv(new_mentees_file)
            mentor_mentee_groups_df = pd.read_csv(groups_file)
            if st.button("Add New Mentees"):
                part4(mentors_df, new_mentees_df, mentor_mentee_groups_df)
        if st.session_state['updated_groups_df'] is not None:
            st.dataframe(st.session_state['updated_groups_df'])
            st.download_button(label="Download Updated Groups CSV", data=st.session_state['updated_groups_df'].to_csv(index=False), file_name='updated_mentor_mentee_groups.csv', key="download_updated_groups")

    elif page == "Part 5: Data Visualization":
        st.header("Part 5: Data Visualization")
        part5(mentors_df, mentees_df)

else:
    st.warning("Please upload both mentors and mentees CSV files.")


### 
# End? 