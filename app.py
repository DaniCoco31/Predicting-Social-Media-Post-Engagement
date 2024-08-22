import streamlit as st
import pandas as pd
import numpy as np
import re
import ollama as ol
import pickle
import sys
import sklearn
from scipy.stats import boxcox
import plotly.graph_objects as go
from streamlit.components.v1 import html
from openai import OpenAI, AuthenticationError
import base64
from io import BytesIO
import io

# Ensure compatibility between versions
print("Scikit-Learn Version:", sklearn.__version__)

sys.path.append('../Notebooks/')
st.set_page_config(layout="wide", page_title="IG Interaction Predictor")

# Updated Custom CSS with white background and black text
custom_css = """
<style>
    body {
        font-family: 'Helvetica', sans-serif;
        font-size: 14px;
        color: #262626;
        background: #ffffff;
    }
    .stApp {
        max-width: 100%;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .logo {
        width: 50px;
        height: 50px;
    }
    h1 {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #405de6, #5851db, #833ab4, #c13584, #e1306c, #fd1d1d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        background: linear-gradient(45deg, #405de6, #5851db, #833ab4, #c13584, #e1306c, #fd1d1d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border: 1px solid;
        border-image-slice: 1;
        border-image-source: linear-gradient(45deg, #405de6, #5851db, #833ab4, #c13584, #e1306c, #fd1d1d);
        border-radius: 4px;
        background-color: #ffffff;
        color: #262626;
    }
    .stButton > button {
        background: linear-gradient(45deg, #405de6, #5851db, #833ab4, #c13584, #e1306c, #fd1d1d);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 4px;
        cursor: pointer;
        transition: transform 0.2s ease-out;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    .output-box {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid;
        border-image-slice: 1;
        border-image-source: linear-gradient(45deg, #405de6, #5851db, #833ab4, #c13584, #e1306c, #fd1d1d);
        color: #262626;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Instagram logo
st.markdown("""
    <div class="logo-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/132px-Instagram_logo_2016.svg.png" class="logo" alt="Instagram Logo">
    </div>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = 'https://github.com/DaniCoco31/Predicting-Social-Media-Post-Engagement/blob/main/Notebooks/model.pkl'
    scaler_path = 'https://github.com/DaniCoco31/Predicting-Social-Media-Post-Engagement/blob/main/Notebooks/scaler.pkl'

    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please check the file paths.")
        st.stop()

model, scaler = load_model_and_scaler()

# Define mappings
category_mapping = {
    'Arts & Culture': 'category_arts_&_culture',
    'Business & Entrepreneurs': 'category_business_&_entrepreneurs',
    'Celebrity & Pop Culture': 'category_celebrity_&_pop_culture',
    'Diaries & Daily Life': 'category_diaries_&_daily_life',
    'Family': 'category_family',
    'Fashion & Style': 'category_fashion_&_style',
    'Film TV & Video': 'category_film_tv_&_video',
    'Fitness & Health': 'category_fitness_&_health',
    'Food & Dining': 'category_food_&_dining',
    'Gaming': 'category_gaming',
    'Learning & Educational': 'category_learning_&_educational',
    'Music': 'category_music',
    'News & Social Concern': 'category_news_&_social_concern',
    'Other Hobbies': 'category_other_hobbies',
    'Relationships': 'category_relationships',
    'Science & Technology': 'category_science_&_technology',
    'Sports': 'category_sports',
    'Travel & Adventure': 'category_travel_&_adventure',
    'Youth & Student Life': 'category_youth_&_student_life'
}

day_mapping = {
    'Monday': 'day_of_week_Monday',
    'Tuesday': 'day_of_week_Tuesday',
    'Wednesday': 'day_of_week_Wednesday',
    'Thursday': 'day_of_week_Thursday',
    'Friday': 'day_of_week_Friday',
    'Saturday': 'day_of_week_Saturday',
    'Sunday': 'day_of_week_Sunday'
}

time_mapping = {
    'Early morning': 'time_of_day_early_morning',
    'Morning': 'time_of_day_morning',
    'Afternoon': 'time_of_day_afternoon',
    'Night': 'time_of_day_night'
}

def get_caption_suggestion(caption, api_key):
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in social media marketing. Your task is to suggest improvements to Instagram captions to increase engagement."},
                {"role": "user", "content": f"Please suggest an improved version of this Instagram caption to increase engagement:\n{caption}"}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except AuthenticationError:
        st.error("Authentication failed. Please check your OpenAI API key.")
        return "Unable to generate caption suggestion due to authentication failure."
    except Exception as e:
        st.error(f"An error occurred while generating the caption suggestion: {str(e)}")
        return "Unable to generate caption suggestion due to an unexpected error."

# Optimized prediction function
@st.cache_data
def predict_interactions(input_data):
    try:
        caption = ' '.join(re.findall(r'[a-zA-Z]+', input_data['caption'].lower()))
        
        data = {
            "following": [input_data['following']],
            "followers": [input_data['followers']],
            "num_posts": [input_data['posts']],
            "is_business_account": [int(input_data['businessAccount'])],
        }
        
        embeddings = ol.embeddings(model='mxbai-embed-large', prompt=caption)['embedding']
        df_embed = pd.DataFrame({f"embedded_{i}": [embed] for i, embed in enumerate(embeddings)})
        
        extra = {
            'description_length': len(caption),
            'followers_trans': np.log1p([input_data['followers']])[0] if input_data['followers'] > 0 else 0,
            'num_posts_trans': np.log1p([input_data['posts']])[0] if input_data['posts'] > 0 else 0,
            'description_length_trans': len(caption)
        }
        df_extra = pd.DataFrame([extra])
        
        category = category_mapping[input_data['category']]
        day = day_mapping[input_data['day']]
        time = time_mapping[input_data['time']]
        
        category_dict = {cat: 1 if cat == category else 0 for cat in category_mapping.values()}
        day_dict = {d: 1 if d == day else 0 for d in day_mapping.values()}
        time_dict = {t: 1 if t == time else 0 for t in time_mapping.values()}
        
        final_df = pd.concat([pd.DataFrame(data), pd.DataFrame([category_dict]), pd.DataFrame([day_dict]), pd.DataFrame([time_dict]), df_extra, df_embed], axis=1)
        
        column_order = [
            'following', 'followers', 'num_posts', 'is_business_account',
        ] + list(category_mapping.values()) + list(day_mapping.values()) + list(time_mapping.values()) + [
            'description_length', 'followers_trans', 'num_posts_trans', 'description_length_trans'
        ] + [f'embedded_{i}' for i in range(1024)]
        
        final_df = final_df.reindex(columns=column_order, fill_value=0)
        
        scaled_features = scaler.transform(final_df)
        predicted_interactions = model.predict(scaled_features)[0]
        interactions_per_follower = predicted_interactions / input_data['followers'] if input_data['followers'] > 0 else 0
        
        return interactions_per_follower, predicted_interactions
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return 0, 0

def improve_caption(input_data, api_key, original_interactions):
    max_attempts = 5
    for attempt in range(max_attempts):
        suggested_caption = get_caption_suggestion(input_data['caption'], api_key)
        if suggested_caption.startswith("Unable to generate caption suggestion"):
            return None, None, None
        
        suggested_input_data = input_data.copy()
        suggested_input_data['caption'] = suggested_caption
        _, suggested_interactions = predict_interactions(suggested_input_data)
        
        if suggested_interactions > original_interactions:
            return suggested_caption, suggested_interactions, attempt + 1
    
    return None, None, max_attempts

# Function to generate heatmap data
@st.cache_data
def generate_heatmap_data(input_data):
    heatmap_data = [
        [day, time, predict_interactions({**input_data, 'day': day, 'time': time})[1]]
        for day in day_mapping.keys()
        for time in time_mapping.keys()
    ]
    return heatmap_data

# Function to create interactive heatmap
@st.cache_data
def create_heatmap(heatmap_data):
    df = pd.DataFrame(heatmap_data, columns=['Day', 'Time', 'Interactions'])
    df_pivot = df.pivot(index='Time', columns='Day', values='Interactions')
    time_order = ['Early morning', 'Morning', 'Afternoon', 'Night']
    df_pivot = df_pivot.reindex(time_order)

    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns,
        y=df_pivot.index,
        colorscale=[
            [0, "#405de6"],      # Instagram blue
            [0.2, "#5851db"],    # Instagram purple
            [0.4, "#833ab4"],    # Instagram darker purple
            [0.6, "#c13584"],    # Instagram pinkish purple
            [0.8, "#e1306c"],    # Instagram pink
            [1, "#fd1d1d"]       # Instagram red
        ],
        hovertemplate='Day: %{x}<br>Time: %{y}<br>Interactions: %{z:.0f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Predicted Interactions by Day and Time',
            'font': {'color': 'black', 'size': 24}
        },
        xaxis_title={'text': 'Day of Week', 'font': {'color': 'black', 'size': 18}},
        yaxis_title={'text': 'Time of Day', 'font': {'color': 'black', 'size': 18}},
        font=dict(color='black'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        height=500,
        width=800,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_xaxes(side='top', tickangle=45, tickfont=dict(color='black', size=14))
    fig.update_yaxes(autorange='reversed', tickfont=dict(color='black', size=14))

    return fig


# New function to get download link for text content
def get_text_download_link(content, filename, link_text):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}">{link_text}</a>'

def get_plot_download_link(fig, filename, text):
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    encoded = base64.b64encode(html_bytes).decode()
    href = f'<a href="data:text/html;base64,{encoded}" download="{filename}">{text}</a>'
    return href

# Streamlit app
def main():
    st.title("IG Interaction Predictor")

    if 'page' not in st.session_state:
        st.session_state.page = 'input'

    # Option to use API
    use_api = st.checkbox("Use API for caption improvement", value=False)

    if use_api:
        # API Key input
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        if not api_key:
            st.warning("Please enter your OpenAI API Key to use the caption improvement feature.")

    if st.session_state.page == 'input':
        with st.form("input_form"):
            st.header("Account Information")
            caption = st.text_area("Caption", height=100, help="Enter your Instagram post caption here.")
            followers = st.number_input("Followers", min_value=1, step=1, value=1)
            following = st.number_input("Following", min_value=0, step=1, value=0)
            posts = st.number_input("Number of posts", min_value=0, step=1, value=0)
            business_account = st.checkbox("Business Account")

            st.header("Post Details")
            category = st.selectbox("Category", list(category_mapping.keys()), index=0)
            day = st.selectbox("Day of posting", list(day_mapping.keys()), index=0)
            time = st.selectbox("Time of posting", list(time_mapping.keys()), index=0)

            submitted = st.form_submit_button("Predict")

        if submitted:
            if not caption:
                st.error("Please fill in the caption field.")
            else:
                st.session_state.input_data = {
                    'caption': caption,
                    'followers': followers,
                    'following': following,
                    'posts': posts,
                    'businessAccount': business_account,
                    'category': category,
                    'day': day,
                    'time': time
                }
                st.session_state.page = 'output'
                st.rerun()

    elif st.session_state.page == 'output':
        original_interactions_per_follower, original_total_interactions = predict_interactions(st.session_state.input_data)

        st.header("Original Prediction Results")
        st.markdown(f"""
        <div class="output-box">
            <p><strong>Original Interactions:</strong> {original_total_interactions:.0f}</p>
            <p><strong>Original Interactions / Follower:</strong> {original_interactions_per_follower:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Generate and display the original heatmap
        original_heatmap_data = generate_heatmap_data(st.session_state.input_data)
        original_fig = create_heatmap(original_heatmap_data)
        st.subheader("Original Heatmap")
        st.plotly_chart(original_fig, use_container_width=True)

        # Download link for original heatmap
        st.markdown(get_plot_download_link(original_fig, "original_heatmap.html", "Download Original Heatmap"), unsafe_allow_html=True)

        if use_api and api_key:
            # Improve caption and compare results
            improved_caption, improved_interactions, attempts = improve_caption(st.session_state.input_data, api_key, original_total_interactions)

            if improved_caption:
                improved_interactions_per_follower = improved_interactions / st.session_state.input_data['followers']
                
                st.header("Improved Caption Results")
                st.markdown(f"""
                <div class="output-box">
                    <p><strong>Improved Caption:</strong> {improved_caption}</p>
                    <p><strong>Improved Interactions:</strong> {improved_interactions:.0f}</p>
                    <p><strong>Improved Interactions / Follower:</strong> {improved_interactions_per_follower:.2f}</p>
                    <p><strong>Improvement Attempts:</strong> {attempts}</p>
                </div>
                """, unsafe_allow_html=True)

                # Generate and display the improved heatmap
                improved_input_data = st.session_state.input_data.copy()
                improved_input_data['caption'] = improved_caption
                improved_heatmap_data = generate_heatmap_data(improved_input_data)
                improved_fig = create_heatmap(improved_heatmap_data)
                st.subheader("Improved Caption Heatmap")
                st.plotly_chart(improved_fig, use_container_width=True)

                # Download link for improved caption
                st.markdown(get_text_download_link(improved_caption, "improved_caption.txt", "Download Improved Caption"), unsafe_allow_html=True)

                st.subheader("Comparison of Original vs Improved")
                comparison_data = {
                    "Metric": ["Total Interactions", "Interactions per Follower"],
                    "Original": [original_total_interactions, original_interactions_per_follower],
                    "Improved": [improved_interactions, improved_interactions_per_follower],
                    "Improvement": [
                        f"{(improved_interactions - original_total_interactions) / original_total_interactions * 100:.2f}%",
                        f"{(improved_interactions_per_follower - original_interactions_per_follower) / original_interactions_per_follower * 100:.2f}%"
                    ]
                }
                st.table(pd.DataFrame(comparison_data))
            else:
                st.warning("Unable to generate an improved caption that increases interactions after multiple attempts.")

                # Convert comparison table to CSV for download
            comparison_df = pd.DataFrame(comparison_data)
            csv = comparison_df.to_csv(index=False)
            st.markdown(get_text_download_link(csv, "comparison_table.csv", "Download Comparison Table"), unsafe_allow_html=True)

        if st.button("Go back"):
            st.session_state.page = 'input'
            st.rerun()

if __name__ == "__main__":
    main()
