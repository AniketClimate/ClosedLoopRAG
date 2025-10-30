
import streamlit as st
import io
from datetime import datetime
from openai import OpenAI

# Try to import document processing libraries
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Document Insights with Perplexity",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #20808d;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #f0f9ff;
        border-left: 5px solid #20808d;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .citation-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .api-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .perplexity-badge {
        background-color: #20808d;
        color: white;
    }
    .openai-badge {
        background-color: #10a37f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📄 Document Insights Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Perplexity AI & OpenAI</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Provider Selection
    api_provider = st.radio(
        "Select AI Provider",
        ["Perplexity AI", "OpenAI"],
        help="Choose which API to use for analysis"
    )

    st.markdown("---")

    # API Key input based on provider
    if api_provider == "Perplexity AI":
        st.info("🔍 **Perplexity AI**: Search-augmented responses with automatic citations")
        api_key = st.text_input(
            "Perplexity API Key",
            type="password",
            help="Get your API key from https://www.perplexity.ai/settings/api"
        )
        st.markdown("[Get Perplexity API Key](https://www.perplexity.ai/settings/api)")
        st.markdown("*Requires Perplexity Pro subscription ($20/month includes $5 API credits)*")

        # Model selection for Perplexity
        model = st.selectbox(
            "Perplexity Model",
            [
                "sonar-pro",
                "sonar",
                "sonar-deep-research"
            ],
            help="sonar-pro: Premier with citations | sonar: Fast and efficient | sonar-deep-research: Exhaustive research"
        )
    else:
        st.info("🤖 **OpenAI**: Powerful general-purpose AI models")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        st.markdown("[Get OpenAI API Key](https://platform.openai.com/api-keys)")

        # Model selection for OpenAI
        model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            help="GPT-3.5: Faster and cheaper | GPT-4: More accurate"
        )

    st.markdown("---")

    # Temperature setting
    temperature = st.slider(
        "Creativity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )

    st.markdown("---")

    # Instructions
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Choose your AI provider above
    2. Enter your API key
    3. Upload your document
    4. Extract text
    5. Ask a question
    6. Get AI-powered insights!
    """)

    st.markdown("---")

    # Comparison table
    with st.expander("📊 API Comparison"):
        st.markdown("""
        | Feature | Perplexity | OpenAI |
        |---------|-----------|--------|
        | **Citations** | ✅ Auto | ❌ No |
        | **Web Search** | ✅ Yes | ❌ No |
        | **Speed** | Fast | Fast |
        | **Cost** | Lower | Higher |
        | **Best For** | Research | General |
        """)

# Helper Functions
def extract_text_from_pdf(file_bytes):
    """Extract text from PDF file"""
    try:
        if not PDF_AVAILABLE:
            return "PDF support not available. Please install PyPDF2."

        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX file"""
    try:
        if not DOCX_AVAILABLE:
            return "DOCX support not available. Please install python-docx."

        doc = DocxDocument(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"

def extract_text_from_txt(file_bytes):
    """Extract text from TXT file"""
    try:
        return file_bytes.decode('utf-8', errors='ignore').strip()
    except Exception as e:
        return f"Error extracting TXT: {str(e)}"

def extract_text(uploaded_file):
    """Extract text based on file type"""
    file_bytes = uploaded_file.read()
    file_type = uploaded_file.type

    if file_type == "application/pdf" or uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.name.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    elif file_type == "text/plain" or uploaded_file.name.endswith('.txt'):
        return extract_text_from_txt(file_bytes)
    else:
        return "Unsupported file format. Please upload PDF, DOCX, or TXT files."

def generate_insights(document_text, question, api_key, model_name, temp, provider):
    """Generate insights using selected API provider"""
    try:
        # Initialize client based on provider
        if provider == "Perplexity AI":
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )
        else:
            client = OpenAI(api_key=api_key)

        # Truncate if too long
        max_chars = 10000
        if len(document_text) > max_chars:
            document_text = document_text[:max_chars] + "\n\n[Document truncated for processing...]"

        # Create prompt
        prompt = f"""You are a helpful AI assistant that analyzes documents and provides clear, detailed insights.

Read the following document carefully and answer this question: {question}

Document:
{document_text}

Please provide a comprehensive answer based on the information in the document. If you use any external sources or knowledge, clearly cite them."""

        # Call API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear, detailed document analysis with citations when available."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=temp
        )

        answer = response.choices[0].message.content

        # Extract citations if Perplexity
        citations = []
        if provider == "Perplexity AI" and hasattr(response, 'citations'):
            citations = response.citations if response.citations else []

        return {
            "answer": answer,
            "citations": citations,
            "model": model_name,
            "provider": provider
        }

    except Exception as e:
        error_msg = str(e)

        # Handle common errors
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return {
                "answer": f"❌ Invalid API key for {provider}. Please check your API key and try again.",
                "citations": [],
                "model": model_name,
                "provider": provider
            }
        elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            return {
                "answer": f"❌ Rate limit exceeded for {provider}. Please check your quota or try again later.",
                "citations": [],
                "model": model_name,
                "provider": provider
            }
        elif "insufficient" in error_msg.lower():
            return {
                "answer": f"❌ Insufficient credits in your {provider} account. Please add credits.",
                "citations": [],
                "model": model_name,
                "provider": provider
            }
        else:
            return {
                "answer": f"❌ Error: {error_msg}",
                "citations": [],
                "model": model_name,
                "provider": provider
            }

# Initialize session state
if 'document_text' not in st.session_state:
    st.session_state.document_text = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'question' not in st.session_state:
    st.session_state.question = None

# Main content area
st.markdown("## 📤 Upload Your Document")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['pdf', 'docx', 'txt'],
    help="Upload PDF, Word, or Text files (max 200MB)"
)

if uploaded_file:
    st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
    st.session_state.filename = uploaded_file.name

    # Extract text button
    if st.button("📖 Extract Text from Document"):
        with st.spinner("Extracting text..."):
            document_text = extract_text(uploaded_file)
            st.session_state.document_text = document_text

            if document_text and not document_text.startswith("Error") and not document_text.startswith("Unsupported"):
                st.success(f"✅ Text extracted successfully! ({len(document_text)} characters)")

                # Show preview
                with st.expander("📄 Preview Extracted Text (first 1000 characters)"):
                    preview_text = document_text[:1000]
                    st.text(preview_text + ("..." if len(document_text) > 1000 else ""))
            else:
                st.error(document_text)

# Question input
if st.session_state.document_text:
    st.markdown("---")
    st.markdown("## ❓ Ask Your Question")

    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What are the main points in this document?
        - Summarize the key findings
        - What risks are mentioned?
        - List all recommendations
        - What is the conclusion?
        - Identify challenges discussed
        - Compare different viewpoints mentioned
        """)

    question = st.text_area(
        "Type your question here:",
        height=100,
        placeholder="e.g., What are the main findings in this document?"
    )

    # Generate insights button
    col1, col2 = st.columns([3, 1])

    with col1:
        generate_button = st.button("🤖 Generate Insights", type="primary", use_container_width=True)

    with col2:
        if api_provider == "Perplexity AI":
            st.markdown('<span class="api-badge perplexity-badge">Perplexity</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="api-badge openai-badge">OpenAI</span>', unsafe_allow_html=True)

    if generate_button:
        if not api_key:
            st.error(f"⚠️ Please enter your {api_provider} API key in the sidebar.")
        elif not question:
            st.error("⚠️ Please enter a question.")
        else:
            with st.spinner(f"🤖 {api_provider} is analyzing your document... This may take 10-30 seconds."):
                result = generate_insights(
                    st.session_state.document_text,
                    question,
                    api_key,
                    model,
                    temperature,
                    api_provider
                )
                st.session_state.insights = result
                st.session_state.question = question

# Display insights
if st.session_state.insights:
    st.markdown("---")

    # Display provider badge
    provider = st.session_state.insights.get('provider', 'Unknown')
    model_used = st.session_state.insights.get('model', 'Unknown')

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 🎯 Insights")
    with col2:
        if provider == "Perplexity AI":
            st.markdown(f'<span class="api-badge perplexity-badge">{model_used}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="api-badge openai-badge">{model_used}</span>', unsafe_allow_html=True)

    # Display answer
    answer = st.session_state.insights.get('answer', 'No answer generated')
    st.markdown(f'<div class="insight-box">{answer}</div>', unsafe_allow_html=True)

    # Display citations if available (Perplexity)
    citations = st.session_state.insights.get('citations', [])
    if citations:
        st.markdown("### 📚 Sources & Citations")
        for i, citation in enumerate(citations, 1):
            st.markdown(f'<div class="citation-box">**[{i}]** {citation}</div>', unsafe_allow_html=True)

    # Download options
    st.markdown("### 💾 Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Create downloadable text file
        citations_text = ""
        if citations:
            citations_text = "\n\nCITATIONS:\n" + "\n".join([f"[{i}] {c}" for i, c in enumerate(citations, 1)])

        result_text = f"""Document: {st.session_state.filename}
Question: {st.session_state.question}
AI Provider: {provider}
Model: {model_used}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}

INSIGHTS:

{answer}
{citations_text}

{'='*80}

Document Text (excerpt):
{st.session_state.document_text[:2000]}...
"""

        st.download_button(
            label="📄 Download TXT",
            data=result_text,
            file_name=f"{st.session_state.filename}_insights.txt",
            mime="text/plain"
        )

    with col2:
        # Create markdown version
        citations_md = ""
        if citations:
            citations_md = "\n\n## Sources\n\n" + "\n".join([f"{i}. {c}" for i, c in enumerate(citations, 1)])

        markdown_text = f"""# Document Insights

**Document:** {st.session_state.filename}  
**Question:** {st.session_state.question}  
**AI Provider:** {provider}  
**Model:** {model_used}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Insights

{answer}
{citations_md}

---

*Generated by Document Insights Generator*
"""

        st.download_button(
            label="📝 Download MD",
            data=markdown_text,
            file_name=f"{st.session_state.filename}_insights.md",
            mime="text/markdown"
        )

    with col3:
        # Create JSON version
        import json
        json_data = {
            "document": st.session_state.filename,
            "question": st.session_state.question,
            "provider": provider,
            "model": model_used,
            "timestamp": datetime.now().isoformat(),
            "insights": answer,
            "citations": citations
        }

        st.download_button(
            label="📊 Download JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"{st.session_state.filename}_insights.json",
            mime="application/json"
        )

    # Option to analyze again
    if st.button("🔄 Analyze Another Document"):
        st.session_state.document_text = None
        st.session_state.insights = None
        st.session_state.filename = None
        st.session_state.question = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with ❤️ using Streamlit, Perplexity AI & OpenAI</p>
    <p>Choose between search-augmented insights (Perplexity) or general AI (OpenAI)</p>
    <p>No data is stored - everything happens in your browser session</p>
</div>
""", unsafe_allow_html=True)
