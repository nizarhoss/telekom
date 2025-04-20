import streamlit as st
import os
import sys
from llama_index.core import StorageContext, load_index_from_storage

# Set page configuration
st.set_page_config(
    page_title="Telecom 10-K Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stTextArea label {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .info-header {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-subheader {
        font-size: 1.3rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #888888;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Set environment variables to avoid tiktoken cache permission issues
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/tiktoken_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["XDG_CACHE_HOME"] = "/tmp/xdg_cache"

# Get API key from Streamlit secrets or environment variables
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set it in the Streamlit secrets or as an environment variable.")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Cache the index loading so it only happens once
@st.cache_resource
def load_vector_index():
    """Load the LlamaIndex from storage"""
    try:
        # Check if index directory exists
        if not os.path.exists("index_10k_storage"):
            st.error("The index_10k_storage directory doesn't exist. Please make sure it's correctly uploaded.")
            return None
            
        # Print the contents of the directory for debugging
        st.write("Index directory contents:")
        st.write(os.listdir("index_10k_storage"))
        
        # Load the index from storage
        storage_context = StorageContext.from_defaults(persist_dir="index_10k_storage")
        index = load_index_from_storage(storage_context)
        
        return index
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None

# Function to query the index
# For more precise control
def query_index(index, query_text):
    if not index:
        return "Error: Index not loaded properly."
    
    try:
        query_engine = index.as_query_engine()
        response_obj = query_engine.query(query_text)
        
        # Extract only the text response
        if hasattr(response_obj, 'response'):
            clean_response = response_obj.response
        else:
            clean_response = str(response_obj)
            
        return clean_response
    except Exception as e:
        return f"Error processing query: {str(e)}"
    
# Main app function
def main():
    # App header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Try to load logo, but don't fail if it doesn't exist
        try:
            st.image("logo.png", width=150)
        except:
            st.write("📊")  # Fallback emoji if no logo
            
        st.markdown('<h1 class="main-header">Telecom 10-K Analyzer</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Explore and analyze insights from telecom company annual reports</p>', unsafe_allow_html=True)
    
    # Load the index
    with st.spinner("Loading index..."):
        index = load_vector_index()
    
    # Display a warning if the index isn't loaded
    if not index:
        st.warning("⚠️ The index could not be loaded. Please check your setup.")
    
    # Query input section
    st.write("### Ask your question")
    query = st.text_area(
        "Enter your question about telecom 10-K reports:",
        height=100,
        placeholder="For example: What are the main risk factors faced by T-Mobile?",
        help="Ask any question about the telecom companies' 10-K filings."
    )
    
    # Submit button
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        submit_button = st.button("Submit Query", type="primary", use_container_width=True)
    
    # Process the query when the button is clicked
    if submit_button:
        if not query:
            st.error("Please enter a query.")
        else:
            with st.spinner("Processing your query..."):
                # Query the index
                response = query_index(index, query)
                
                # Display the response
                st.write("### Response")
                st.write(response)
    
    # About This Project section
    st.markdown('<h4 class="info-header">About This Project</h2>', unsafe_allow_html=True)
    
    
    st.markdown('<h5 class="info-subheader">Purpose</h3>', unsafe_allow_html=True)
    st.write("""
    I wanted to try vectorizing and indexing from my own subset of information. This can be scaled up in leveraging different bodies of text.
    """)
    
    st.markdown('<h5 class="info-subheader">How It Works</h3>', unsafe_allow_html=True)
    st.write("""
    The application uses LlamaIndex to create a searchable knowledge base from the text of telecom companies'  
    10-K filings. When you ask a question, the system retrieves the most relevant information from these filings 
    and uses OpenAI's language models to generate a human-readable response.
    """)
    
    st.markdown('<h5 class="info-subheader">Data Sources</h3>', unsafe_allow_html=True)
    st.write("""
    Using the SEC EDGAR API, I pulled information from 10K reports for companies in the Telecom industry.
    This demo includes 10-K filings from major telecom companies including T-Mobile (TMUS), AT&T (T), 
    Verizon (VZ), and EchoStar (SATS).
    """)
    
    # Footer
    st.markdown('<div class="footer">Created by Nizar Hoss - © 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()