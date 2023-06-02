import streamlit as st
import scrap
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from textblob import TextBlob
import seaborn as sns
import numpy as np
import random
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import spacy
from spacy import displacy
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models
import pyLDAvis
# from PIL import Image
# import io
# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPM
import fr_core_news_sm
from IPython.core.display import display, HTML


home, view_data, analyse_data = st.tabs(['Home', 'View Data', 'Analyse Data'])



# nlp = spacy.load('fr_core_news_sm')
nlp = fr_core_news_sm.load()

# Define actions for each operation
with home:
    st.title("Welcome to the Google Maps Reviews Scrapper and Analyser!")
    st.write("Use the tabs to navigate between pages.")

# with scrap_reviews:
#     st.title("Scrape Google Maps Reviews")

#     with open('config.json') as json_file:
#         config = json.load(json_file)

#     url = st.text_input(
#         'Enter the URL of the Google Maps location', config['URL'])
#     driver_path = st.text_input(
#         'Enter the path to the Chrome Driver', config['DriverLocation'])

#     if st.button('Scrape Reviews'):
#         config['URL'] = url
#         config['DriverLocation'] = driver_path

#         with open('config.json', 'w') as json_file:
#             json.dump(config, json_file)

#         result = scrap.main_scrap()

#         if result == "Success":
#             st.success("Scraping completed successfully!")
#         else:
#             st.error(f"Scraping failed. Error: {result}")



with view_data:
    st.title("View Data")
    if st.button('Load Data'):
        try:
            data = pd.read_csv('data.csv')
            st.dataframe(data)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

with analyse_data:
    st.title("Analyse Data")
    if st.button('Analyse Data'):
        try:

            # 1. Loading the Data
            data = pd.read_csv('data.csv')

            # 2. Data Cleaning
            # Convert the index back to a column
            data.reset_index(inplace=True)
            # Create a new column to hold just the year-month information
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S.%f') 
            data['year_month'] = data['date'].dt.to_period('M')
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

            st.subheader("Individual ratings over time:")
            plt.figure(figsize=(12, 6))
            # Plot each individual review
            plt.scatter(data['year_month'].dt.to_timestamp(), data['rating'], alpha=0.5)

            plt.title('Individual ratings over time')
            plt.ylabel('Rating')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # set x-axis intervals to every 6 months
            plt.gcf().autofmt_xdate()  # Rotate date labels for readability

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

            nltk.download('stopwords')
            french_stopwords = set(stopwords.words('french'))
            custom_stopwords = {"c'est", "très", "fois","si"}  # add any custom stopwords here
            all_stopwords = STOPWORDS.union(french_stopwords).union(custom_stopwords) # Union of English and French stopwords

            wordcloud = WordCloud(background_color="white", stopwords=all_stopwords, max_words=100, contour_color='steelblue')
            wordcloud.generate(' '.join(data['comment'].astype(str)))

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
            
            city_counts = data['city'].value_counts()
            region_counts = data['region'].value_counts()

            st.subheader("Frequency of Reviews by City:")
            st.bar_chart(city_counts)

            st.subheader("Frequency of Reviews by Region:")
            st.bar_chart(region_counts)


            city_avg_ratings = data.groupby('city')['rating'].mean()
            region_avg_ratings = data.groupby('region')['rating'].mean()

            st.subheader("Average Rating by City:")
            st.bar_chart(city_avg_ratings)

            st.subheader("Average Rating by Region:")
            st.bar_chart(region_avg_ratings)


            city_polarity = data.groupby('city')['polarity'].mean().reset_index()
            city_subjectivity = data.groupby('city')['subjectivity'].mean().reset_index()
            region_polarity = data.groupby('region')['polarity'].mean().reset_index()
            region_subjectivity = data.groupby('region')['subjectivity'].mean().reset_index()

            st.subheader("Average Sentiment Polarity by City:")
            st.bar_chart(city_polarity.set_index('city'))

            st.subheader("Average Sentiment Subjectivity by City:")
            st.bar_chart(city_subjectivity.set_index('city'))

            st.subheader("Average Sentiment Polarity by Region:")
            st.bar_chart(region_polarity.set_index('region'))

            st.subheader("Average Sentiment Subjectivity by Region:")
            st.bar_chart(region_subjectivity.set_index('region'))


            # Download stopwords and punkt tokenizer
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
            
            np.random.seed(42)
            random.seed(42)
            # Preprocessing
            processed_data = data['comment'].map(word_tokenize)
            dictionary = corpora.Dictionary(processed_data)
            corpus = [dictionary.doc2bow(text) for text in processed_data]

            # Apply LDA
            num_topics = 5  # Set this to the number of topics you want to identify
            lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

            # Prepare the visualization
            vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
            # Convert the prepared visualization to HTML
            vis_html = pyLDAvis.prepared_data_to_html(vis)

            # Save the visualization HTML to a file
            with open('lda.html', 'w') as f:
                f.write(vis_html)

            # Display the HTML in Streamlit
            st.markdown('## Topic Modeling Visualization:')
            st.components.v1.html(vis_html, height=1000, width=1200)


            data['comment'].dropna().apply(lambda x: [(ent.text, ent.label_) for ent in nlp(x).ents])
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
            plt.xticks(rotation=45, ha='right')

            st.pyplot(fig)

            # Process the comments to extract named entities
            doc = nlp(all_comments)
            exclusion_list = ['l’', 's’', 'j’', 'm’', 'y’', 'qu’', 'd’', 'n’']

            # Extract named entities and count their frequencies
            named_entities = Counter([ent.text for ent in doc.ents if ent.label_ in ['PER', 'ORG', 'LOC', 'DATE'] and ent.text not in exclusion_list])

            # Convert named entities to a pandas dataframe
            df_entities = pd.DataFrame(named_entities.most_common(10), columns=['Entity', 'Count'])
            st.table(df_entities)

            # https://huggingface.co/mrm8488/camembert2camembert_shared-finetuned-french-summarization
            
            # for review in data['comment']:
            #     doc = nlp(review) 
            #     png_image = render_displacy(doc) 
            #     st.image(png_image, format='PNG')    
            #     for token in doc: 
            #         st.write(f'{token.text} <--{token.dep_}-- {token.head.text}')

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
