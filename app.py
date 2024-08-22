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
from openai import OpenAI

# Ensure compatibility between versions
print("Scikit-Learn Version:", sklearn.__version__)

sys.path.append('../Notebooks/')
st.set_page_config(layout="wide", page_title="IG Interaction Predictor")

# Custom CSS with Instagram-like colors and gradients
custom_css = """
<style>
    body {
        font-family: 'Helvetica', sans-serif;
        font-size: 14px;
        color: #262626;
        background: #121212;
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
        background-color: #1e1e1e;
        color: #ffffff;
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
        background-color: #1e1e1e;
        padding: 2rem;
        border-radius: 8px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid;
        border-image-slice: 1;
        border-image-source: linear-gradient(45deg, #405de6, #5851db, #833ab4, #c13584, #e1306c, #fd1d1d);
        color: #ffffff;
    }
</style>
"""

# JavaScript for input validation
js_code = """
<script>
document.addEventListener('DOMContentLoaded', (event) => {
    const form = document.querySelector('form');
    form.addEventListener('submit', (e) => {
        const inputs = form.querySelectorAll('input[type="number"]');
        let isValid = true;
        inputs.forEach(input => {
            if (input.value === '' || parseInt(input.value) < 0) {
                isValid = false;
                input.style.borderColor = 'red';
            } else {
                input.style.borderColor = '';
            }
        });
        if (!isValid) {
            e.preventDefault();
            alert('Please fill all fields with non-negative numbers.');
        }
    });
});
</script>
"""

st.markdown(custom_css, unsafe_allow_html=True)
html(js_code)

# Instagram logo
st.markdown("""
    <div class="logo-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/132px-Instagram_logo_2016.svg.png" class="logo" alt="Instagram Logo">
    </div>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = '/home/danicoco/Escritorio/IronHack-DataAnalysis/8. week-eight/project/Notebooks/model.pkl'
    scaler_path = '/home/danicoco/Escritorio/IronHack-DataAnalysis/8. week-eight/project/Notebooks/scaler.pkl'

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

# Function to read API key from file
def get_api_key():
    api_key_path = '.env'  # Change this to the path of your file containing the API key
    try:
        with open(api_key_path, 'r') as file:
            api_key = file.read().strip()
        if not api_key.startswith('sk-'):
            raise ValueError("The API key doesn't seem to be in the correct format.")
        return api_key
    except FileNotFoundError:
        st.error(f"API key file not found at {api_key_path}. Please ensure the file exists and contains your API key.")
    except ValueError as e:
        st.error(f"Invalid API key format: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred while reading the API key: {str(e)}")
    return None

def get_caption_suggestion(caption):
    api_key = get_api_key()
    if not api_key:
        return "Unable to generate caption suggestion due to API key issues."
    
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


# Function to get caption suggestion
def get_caption_suggestion(caption):
    api_key = get_api_key()
    if not api_key:
        return "Unable to generate caption suggestion due to missing API key."
    
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
        colorscale='YlOrRd',
        hovertemplate='Day: %{x}<br>Time: %{y}<br>Interactions: %{z:.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='Predicted Interactions by Day and Time',
        xaxis_title='Day of Week',
        yaxis_title='Time of Day',
        font=dict(color='white'),
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        height=500,
        width=800,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_xaxes(side='top', tickangle=45)
    fig.update_yaxes(autorange='reversed')

    return fig

# Streamlit app
def main():
    st.title("IG Interaction Predictor")

    if 'page' not in st.session_state:
        st.session_state.page = 'input'

    if st.session_state.page == 'input':
        with st.form("input_form"):
            st.header("Account Information")
            caption = st.text_input("Caption")
            followers = st.number_input("Followers", min_value=0, step=1)
            following = st.number_input("Following", min_value=0, step=1)
            posts = st.number_input("Number of posts", min_value=0, step=1)
            business_account = st.checkbox("Business Account")

            st.header("Post Details")
            category = st.selectbox("Category", list(category_mapping.keys()))
            day = st.selectbox("Day of posting", list(day_mapping.keys()))
            time = st.selectbox("Time of posting", list(time_mapping.keys()))

            submitted = st.form_submit_button("Predict")

        if submitted:
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

        # Get caption suggestion
        suggested_caption = get_caption_suggestion(st.session_state.input_data['caption'])
        st.subheader("Suggested Caption Improvement")
        st.write(suggested_caption)

        if not suggested_caption.startswith("Unable to generate caption suggestion"):
            

           # Predict interactions with the suggested caption
            suggested_input_data = st.session_state.input_data.copy()
            suggested_input_data['caption'] = suggested_caption
            suggested_interactions_per_follower, suggested_total_interactions = predict_interactions(suggested_input_data)

            st.header("Suggested Caption Prediction Results")
            st.markdown(f"""
            <div class="output-box">
                <p><strong>Suggested Interactions:</strong> {suggested_total_interactions:.0f}</p>
                <p><strong>Suggested Interactions / Follower:</strong> {suggested_interactions_per_follower:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

            # Generate and display the suggested heatmap
            suggested_heatmap_data = generate_heatmap_data(suggested_input_data)
            suggested_fig = create_heatmap(suggested_heatmap_data)
            st.subheader("Suggested Caption Heatmap")
            st.plotly_chart(suggested_fig, use_container_width=True)

            st.subheader("Comparison of Original vs Suggested")
            comparison_data = {
                "Metric": ["Total Interactions", "Interactions per Follower"],
                "Original": [original_total_interactions, original_interactions_per_follower],
                "Suggested": [suggested_total_interactions, suggested_interactions_per_follower],
                "Improvement": [
                    f"{(suggested_total_interactions - original_total_interactions) / original_total_interactions * 100:.2f}%",
                    f"{(suggested_interactions_per_follower - original_interactions_per_follower) / original_interactions_per_follower * 100:.2f}%"
                ]
            }
            st.table(pd.DataFrame(comparison_data))
        else:
            st.warning("Caption suggestion feature is unavailable due to missing API key.")

        if st.button("Go back"):
            st.session_state.page = 'input'
            st.rerun()

if __name__ == "__main__":
    main()