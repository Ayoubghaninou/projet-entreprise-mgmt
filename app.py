import streamlit as st
import scrap
import json
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import spacy

# Set up the layout
# st.sidebar.title('Menu')
# operation = st.sidebar.radio(
#     "Go to", ['Home', 'Scrape Reviews', 'View Data', 'Analyse Data'])

home, scrap_reviews, view_data, analyse_data = st.tabs(['Home', 'Scrape Reviews', 'View Data', 'Analyse Data'])

with open('config.json') as json_file:
    config = json.load(json_file)

# Define actions for each operation
with home:
    st.title("Welcome to the Google Maps Reviews Scrapper and Analyser!")
    st.write("Use the sidebar to navigate between operations.")

with scrap_reviews:
    st.title("Scrape Google Maps Reviews")

    with open('config.json') as json_file:
        config = json.load(json_file)

    url = st.text_input(
        'Enter the URL of the Google Maps location', config['URL'])
    driver_path = st.text_input(
        'Enter the path to the Chrome Driver', config['DriverLocation'])

    if st.button('Scrape Reviews'):
        config['URL'] = url
        config['DriverLocation'] = driver_path

        with open('config.json', 'w') as json_file:
            json.dump(config, json_file)

        result = scrap.main_scrap()

        if result == "Success":
            st.success("Scraping completed successfully!")
        else:
            st.error(f"Scraping failed. Error: {result}")


with view_data:
    st.title("View Data")
    if st.button('Load Data'):
        try:
            data = pd.read_csv('out.csv')
            st.dataframe(data)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

with analyse_data:
    st.title("Analyse Data")
    if st.button('Analyse Data'):
        try:

            # 1. Loading the Data
            data = pd.read_csv('out.csv')

            # 2. Data Cleaning
            data['date'] = pd.to_datetime(data['date'])
            data.drop_duplicates(inplace=True)
            # Replace NaNs with empty strings
            data['comment'].fillna('', inplace=True)

            # 3. Quantitative Analysis
            # Descriptive statistics
            st.subheader("Descriptive statistics of ratings:")
            st.write(data['rating'].describe())

            # Trend over time
            st.subheader("Average rating over time:")
            plt.figure(figsize=(12, 6))
            data.set_index('date', inplace=True)
            data.resample('M')['rating'].mean().plot()
            plt.title('Average rating over time')
            plt.ylabel('Average rating')
            st.pyplot(plt)

            # 4. Qualitative Analysis
            # Sentiment analysis
            st.subheader("Average sentiment polarity and subjectivity:")
            # Ensure all comments are processed as strings
            data['polarity'] = data['comment'].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity)
            data['subjectivity'] = data['comment'].apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity)
            polarity = round(data['polarity'].mean(), 2)
            subjectivity = round(data['subjectivity'].mean(), 2)

            st.write(f"Average sentiment polarity is {polarity}. This score is slightly towards the positive side, suggesting that the reviews are generally neutral to slightly positive.")
            st.write(f"Average sentiment subjectivity is {subjectivity}. This low score suggests that the reviews are generally more objective (fact-based) rather than opinion-based.")

            st.write(data[['polarity', 'subjectivity']].mean())


            # Create a WordCloud object
            wordcloud = WordCloud(background_color="white", stopwords=STOPWORDS, max_words=100, contour_color='steelblue')
            # Generate a word cloud
            wordcloud.generate(' '.join(data['comment'].astype(str)))
            # Visualize the word cloud
            st.subheader("Word Cloud:")
            st.image(wordcloud.to_array())
            
            # 5. Data Visualization
            # Histogram of the ratings
            st.subheader("Histogram of Ratings:")
            plt.figure(figsize=(12, 6))
            plt.hist(data['rating'], bins=np.arange(
                1, 7) - 0.5, edgecolor='black')
            plt.title('Histogram of Ratings')
            plt.xlabel('Rating')
            plt.ylabel('Frequency')
            plt.xticks(range(1, 6))
            st.pyplot(plt)
            
            # Download stopwords and punkt tokenizer
            nltk.download('stopwords')
            nltk.download('punkt')

            # Define the set of stopwords for French
            stop_words = set(stopwords.words('french'))

            # Concatenate all the comments:
            all_comments = ' '.join(data['comment'].dropna().astype(str))

            # Tokenize the comments into words
            words = word_tokenize(all_comments)

            # Filter out stop words, punctuation marks, and single-letter words
            filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha() and len(word) > 1]

            # Count the frequency of each word
            counter = Counter(filtered_words)

            # Get the most frequent words
            most_occur = counter.most_common(10)

            st.subheader("Most Frequent Words:")
            chart, table = st.tabs(['Chart', 'Table'])
            word_freq = [(word, freq) for word, freq in most_occur]
            df_word_freq = pd.DataFrame(word_freq, columns=["Word", "Frequency"])

            chart.subheader("Most Frequent Words bar chart")
            chart.bar_chart(df_word_freq, x = "Word", y = "Frequency")

            table.subheader("Most Frequent Words table")
            table.table(df_word_freq)

            st.subheader("Monthly Reviews Count:")
            data.resample('M').size().plot()
            st.pyplot(plt)

            # Rating Distribution Over Time
            st.subheader("Rating Distribution Over Time:")
            monthly_ratings = data.groupby(pd.Grouper(freq='M'))['rating'].value_counts().unstack(fill_value=0)

            fig, ax = plt.subplots(figsize=(10, 6))
            monthly_ratings.plot(kind='bar', ax=ax)
            ax.set_xlabel('Month')
            ax.set_ylabel('Count')
            ax.set_title('Rating Distribution Over Time')
            st.pyplot(fig)

            # Load the pre-trained French language model
            nlp = spacy.load('fr_core_news_sm')

            # Process the comments to extract named entities
            doc = nlp(all_comments)

            # Extract named entities and count their frequencies
            named_entities = Counter([ent.text for ent in doc.ents if ent.label_ in ['PER', 'ORG', 'LOC', 'DATE']])

            # Display the extracted named entities and their frequencies in the Streamlit app
            st.subheader("Named Entities:")
            for entity, count in named_entities.most_common(10):
                st.write(f"{entity}: {count}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
