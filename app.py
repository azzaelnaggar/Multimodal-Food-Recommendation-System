import streamlit as st
import weaviate
import base64
from io import BytesIO
from PIL import Image
import os
from typing import List, Optional

COHERE_API_KEY = os.getenv("COHERE_API_KEY", <YOUR KEY>)###############33

# Constants
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
COLLECTION_NAME = "FoodsMultiModal"
MAX_IMAGE_SIZE = (400, 400)
TOP_RESULTS = 3

class WeaviateConnectionError(Exception):
    """Custom exception for Weaviate connection issues"""
    pass

@st.cache_resource
def get_weaviate_client():
    """Get Weaviate client with error handling"""
    try:
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
            headers={"X-Cohere-Api-Key": COHERE_API_KEY}
        )
        
        # Test connection
        if not client.is_ready():
            raise WeaviateConnectionError("Weaviate is not ready")
        
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {str(e)}")
        st.info("Make sure Docker containers are running: `docker-compose up -d`")
        raise WeaviateConnectionError(f"Connection failed: {str(e)}")

def base64_to_image(base64_str: str) -> Optional[Image.Image]:
    """Convert base64 string to PIL Image with error handling"""
    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        st.warning(f"Failed to decode image: {str(e)}")
        return None

def image_to_base64(image_file) -> Optional[str]:
    """Convert uploaded file to base64 with validation"""
    try:
        img = Image.open(image_file)
        
        # Validate image
        if img.size[0] > 5000 or img.size[1] > 5000:
            st.warning("Image is too large. Resizing...")
        
        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        img = img.resize(MAX_IMAGE_SIZE)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        st.error(f"Failed to process image: {str(e)}")
        return None

def search_foods_by_text(query: str, limit: int = 10) -> Optional[List]:
    """Search foods by text query with error handling"""
    if not query or len(query.strip()) < 2:
        st.warning("Please enter at least 2 characters")
        return None
    
    try:
        client = get_weaviate_client()
        foods_collection = client.collections.get(COLLECTION_NAME)
        
        response = foods_collection.query.near_text(
            query=query.strip(),
            limit=limit,
            target_vector="text_vector",
            return_properties=["name", "description", "price", "calories", "image"],
            return_metadata=["distance", "certainty"]
        )
        
        return response.objects if response.objects else []
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return None

def search_foods_by_image(image_base64: str, limit: int = 10) -> Optional[List]:
    """Search foods by image with error handling"""
    if not image_base64:
        st.error("Invalid image data")
        return None
    
    try:
        client = get_weaviate_client()
        foods_collection = client.collections.get(COLLECTION_NAME)
        
        response = foods_collection.query.near_image(
            near_image=image_base64,
            limit=limit,
            target_vector="image_vector",
            return_properties=["name", "description", "price", "calories", "image"],
            return_metadata=["distance", "certainty"]
        )
        
        return response.objects if response.objects else []
    except Exception as e:
        st.error(f"Image search failed: {str(e)}")
        return None

def display_food_card_compact(food, show_score: bool = True):
    """Display food card with lazy image loading"""
    try:
        # Display image
        if food.properties.get('image'):
            img = base64_to_image(food.properties['image'])
            if img:
                st.image(img, use_container_width=True)
        
        # Display info
        st.markdown(f"### {food.properties.get('name', 'Unknown')}")
        
        # Show match score
        if show_score and hasattr(food.metadata, 'certainty'):
            certainty_percent = food.metadata.certainty * 100
            st.markdown(f"**Match: {certainty_percent:.1f}%**")
        
        price = food.properties.get('price', 0)
        calories = food.properties.get('calories', 0)
        
        st.write(f" **${price:.2f}**")
        st.write(f" **{calories} kcal**")
        
        # Expandable description
        description = food.properties.get('description', 'No description available')
        with st.expander(" Description"):
            st.write(description)
    except Exception as e:
        st.error(f"Failed to display food card: {str(e)}")

def display_results(results: List, top_n: int = TOP_RESULTS):
    """Display search results in organized layout"""
    if not results:
        st.warning("No results found")
        return
    
    # Top results
    st.subheader(f" Top {min(top_n, len(results))} Matches")
    cols = st.columns(top_n)
    for idx, food in enumerate(results[:top_n]):
        with cols[idx]:
            display_food_card_compact(food)
    
    # Remaining results
    if len(results) > top_n:
        st.divider()
        st.subheader("📋 Other Results")
        
        remaining = results[top_n:]
        for row_start in range(0, len(remaining), 3):
            cols = st.columns(3)
            for idx, food in enumerate(remaining[row_start:row_start+3]):
                with cols[idx]:
                    display_food_card_compact(food)

# Page config
st.set_page_config(
    page_title="Food Search App",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .uploadedFile {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("🍕 Food Search & Discovery")
st.write("Search for your favorite foods by text or image!")

# Tabs
tab1, tab2 = st.tabs(["🔍 Search by Text", "📸 Search by Image"])

# Text Search Tab
with tab1:
    with st.form(key="text_search_form"):
        search_query = st.text_input(
            "Search for food",
            placeholder="e.g., pizza, pasta, dessert, spicy...",
            help="Enter keywords to search for food items"
        )
        search_button = st.form_submit_button("Search", type="primary")
    
    if search_button and search_query:
        with st.spinner("Searching..."):
            results = search_foods_by_text(search_query, limit=10)
        
        if results:
            display_results(results)
        elif results is not None:
            st.info("No results found. Try different keywords.")

# Image Search Tab
with tab2:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a food image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of food for best results"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if uploaded_file is not None:
            if st.button("Search Similar Foods", type="primary", use_container_width=True):
                with st.spinner("Analyzing image and searching..."):
                    image_base64 = image_to_base64(uploaded_file)
                    
                    if image_base64:
                        results = search_foods_by_image(image_base64, limit=10)
                        
                        if results:
                            st.success(f"Found {len(results)} similar dishes!")
                            st.session_state['image_results'] = results
                        elif results is not None:
                            st.info("No similar foods found.")
    
    # Display image search results
    if 'image_results' in st.session_state:
        st.divider()
        display_results(st.session_state['image_results'])
