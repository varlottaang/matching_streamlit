
# Mentorship Matching App

This Streamlit application facilitates the matching of mentors and mentees based on their preferences and keywords. It provides various functionalities for initial matching, reassignment, best match finding, adding new mentees to existing groups, and visualizing the results.

DEMO [Streamlit](https://complete-revised.streamlit.app/)

## Features

1. **Initial Matching**: Match mentors and mentees based on their preferences and keywords.
2. **Mentees Reassignment**: Reassign mentees if their mentors leave.
3. **Best Matches**: Find the best matches for mentees.
4. **Add New Mentees**: Add new mentees to existing mentor-mentee groups.
5. **Data Visualization**: Visualize group sizes, word clouds, and distribution across timezones.

## How to Use

### Step 1: Upload CSV Files
Upload the `mentors` and `mentees` CSV files using the sidebar.

### Step 2: Choose a Functionality
Select the desired part to proceed with:
- Part 1: Initial Matching
- Part 2: Mentees Reassignment
- Part 3: Best Matches
- Part 4: Add New Mentees to Existing Groups
- Part 5: Data Visualization

### Part 1: Initial Matching
1. Click "Run Initial Matching".
2. View and download the resulting matches and groups.

### Part 2: Mentees Reassignment
1. Upload the initial groups CSV file.
2. Enter the leaving mentors' `name_id` separated by commas.
3. Click "Reassign Mentees".
4. View and download the reassigned groups.

### Part 3: Best Matches
1. Enter the leaving mentors' `name_id` separated by commas (or 'None').
2. Click "Find Best Matches".
3. View and download the best matches.

### Part 4: Add New Mentees to Existing Groups
1. Upload the new mentees CSV file.
2. Upload the existing groups CSV file.
3. Click "Add New Mentees".
4. View and download the updated groups.

### Part 5: Data Visualization
1. Upload the groups CSV file.
2. View various visualizations including average group sizes, biggest and smallest group sizes by timezone, word clouds, and percentage distribution across timezones.

## Dependencies

- Streamlit
- pandas
- numpy
- sentence-transformers
- matplotlib
- wordcloud
- seaborn

## Running the App

To run the app locally, execute the following command:

```sh
pip install -r requirements.txt

```sh
streamlit run complete_revised.py

## License
The project is licensed under the MIT License.
