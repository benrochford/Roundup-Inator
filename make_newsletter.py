import streamlit as st
import requests
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import docx
import io
import markdown
import time
import random

# Add this near the top of the file with other constants
DEFAULT_ROUNDUP_PROMPT = """Create a research round-up following this structure:

# ((apparent topic)) Research Round-up: ((apparent date range from papers))

Overview Paragraph with:
- 3-4 major themes identified from the papers
- Brief (1-2 sentence) explanation of why each theme matters
- (Include inline links to papers that are most relevant to each theme)

## Notable Papers
IMPORTANT: You MUST discuss AT LEAST 5 papers in this section, preferably 6-8. Do not stop at 3 papers.
For each paper:
**Paper Title** (with link to Semantic Scholar URL)
Key findings in 2-3 sentences, 1 sentence of why it matters

## Quick Takes
- 8-15 highlights from other interesting papers
- Focus on actionable insights or surprising findings
- Format like "a study by AUTHOR et al. found that..." which is hyperlinked to the semanticscholar url of the paper

## Emerging Trends, Future Directions
in paragraph form:
- 5 or so important emerging trends based on developments from the research                                                     
- 3 sentence discussion of potential implications, future research directions that are coming"""

# Add near the top with other constants
LOADING_MESSAGES = [
    "Reading papers",
    "Highlighting important bits",
    "Taking a snack break",
    "Checking Mastodon for hot takes",
    "Drawing connections",
    "Looking at pictures of national parks",
    "Procrastinating productively",
    "Making coffee",
    "Organizing sticky notes",
    "Double-checking citations",
    "Contemplating the nature of knowledge",
    "Doing desk stretches",
    "Looking at mildly interesting things on Reddit",
    "Petting nearby cats for inspiration",
    "Reorganizing browser bookmarks",
    "Watching one more YouTube video",
]

st.set_page_config(page_title="Research Roundup Generator", layout="wide")

# Add custom CSS for dark mode styling
st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton button {
        background-color: #c71585;
        color: white !important;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #ff4500;
        color: white !important;
    }
    .stButton button:active {
        color: white !important;
    }
    .stTextInput input {
        background-color: #2d2d2d;
        color: #ffffff;
        border-radius: 4px;
        border: 1px solid #404040;
    }
    .stMarkdown {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'papers_df' not in st.session_state:
    st.session_state.papers_df = None
if 'newsletter_content' not in st.session_state:
    st.session_state.newsletter_content = None
if 'custom_prompt' not in st.session_state:
    st.session_state.custom_prompt = DEFAULT_ROUNDUP_PROMPT

def suggest_search_terms(topic_description, openai_api_key):
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")
    prompt = ChatPromptTemplate.from_template("""
    Given this research topic description, suggest 2-4 specific search terms that would be good for finding relevant papers on Semantic Scholar:
    {topic}
    
    Return only the search terms, one per line, nothing else.
    """)
    
    messages = prompt.format_messages(topic=topic_description)
    response = llm.invoke(messages)
    return [term.strip() for term in response.content.split('\n') if term.strip()]

# SemanticScholar API setup
def collect_papers(queries, start_year=None, end_year=None, api_key=None):
    url = "https://api.semanticscholar.org/graph/v1/paper/search/"
    params = {
        "fields": "title,url,abstract,citationCount,authors,publicationTypes,publicationDate,openAccessPdf",
        "limit": 100,
    }
    
    # Add year range filter if provided
    if start_year and end_year:
        params["year"] = f"{start_year}-{end_year}"
    elif start_year:
        params["year"] = f"{start_year}-"
    elif end_year:
        params["year"] = f"-{end_year}"
    
    headers = {"x-api-key": api_key} if api_key else {}
    all_recent_papers = []
    results_by_term = {}  # Track results for each term
    
    for query in queries:
        params["query"] = f'"{query}"'
        with st.spinner(f"Searching for: {query}"):
            response = requests.get(url, params=params, headers=headers)
            papers = response.json().get('data', [])
            results_by_term[query] = len(papers)
            
            for paper in papers:
                paper['search_term'] = query
            
            all_recent_papers.extend(papers)
            # Show immediate results for this term
            st.write(f"📍 Found {len(papers)} papers for '{query}'")
    
    return all_recent_papers, results_by_term

# Clean and process papers
def process_papers(papers):
    # Create DataFrame with basic error handling
    df = pd.DataFrame(papers)
    
    # Define expected columns with default values
    expected_columns = {
        'title': None,
        'url': None,
        'paperId': None,
        'abstract': None,
        'citationCount': 0,
        'authors': None,
        'publicationTypes': None,
        'publicationDate': None,
        'openAccessPdf': None,
        'search_term': None
    }
    
    # Add missing columns with default values
    for col, default in expected_columns.items():
        if col not in df.columns:
            df[col] = default
    
    # Process dates safely
    df['publicationDate'] = pd.to_datetime(df['publicationDate'], errors='coerce')
    df['year'] = df['publicationDate'].dt.year.fillna(-1).astype('Int64')
    df['month'] = df['publicationDate'].dt.month.fillna(-1).astype('Int64')
    
    # Process other columns safely
    df['openAccessUrl'] = df['openAccessPdf'].apply(lambda x: x['url'] if isinstance(x, dict) and 'url' in x else None)
    df['authors'] = df['authors'].apply(lambda x: [author['name'] for author in x] if isinstance(x, list) else None)
    
    # Reorder columns
    columns_order = ['title', 'url', 'paperId', 'abstract', 'citationCount', 'authors', 'publicationTypes', 
                     'publicationDate', 'year', 'month', 'openAccessUrl', 'search_term']
    
    # Only include columns that exist
    existing_columns = [col for col in columns_order if col in df.columns]
    return df[existing_columns].copy()

# Newsletter generation
def generate_newsletter(papers_df, openai_api_key, custom_prompt=None):
    # Sort by both date and citation count (giving more weight to recent papers)
    papers_df['score'] = papers_df['citationCount'].fillna(0) + (1 / (1 + (pd.Timestamp.now() - papers_df['publicationDate']).dt.days))
    
    # Get top 35 papers based on score
    papers_df = papers_df.sort_values('score', ascending=False).head(35)
    
    newsletter_prompt = ChatPromptTemplate.from_template(
        custom_prompt + "\n\nPapers to analyze:\n{topic_analyses}"
    )
    
    # Format papers for prompt
    topic_analyses = []
    for topic in papers_df['search_term'].unique():
        topic_papers = papers_df[papers_df['search_term'] == topic]
        
        papers_text = []
        for _, paper in topic_papers.iterrows():
            authors = ', '.join(author for author in paper['authors']) if isinstance(paper['authors'], list) else 'Unknown'
            paper_text = f"""Title: {paper['title']}
            URL: {paper['url']}
            Authors: {authors}
            Abstract: {paper['abstract']}
            Publication Date: {paper['publicationDate']}
            """
            papers_text.append(paper_text)
        
        topic_text = f"Topic: {topic}\n" + "\n".join(papers_text)
        topic_analyses.append(topic_text)
    
    topic_analyses_text = "\n---\n".join(topic_analyses)
    
    # Generate newsletter
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")
    messages = newsletter_prompt.format_messages(topic_analyses=topic_analyses_text)
    newsletter = llm.invoke(messages)
    
    return newsletter.content


def convert_to_docx(markdown_content):
    doc = docx.Document()
    
    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = docx.shared.Pt(11)
    
    # Customize heading styles
    h1_style = doc.styles['Heading 1']
    h1_style.font.name = 'Calibri'
    h1_style.font.size = docx.shared.Pt(16)
    h1_style.font.bold = True
    
    h2_style = doc.styles['Heading 2']
    h2_style.font.name = 'Calibri'
    h2_style.font.size = docx.shared.Pt(14)
    h2_style.font.bold = True
    
    # Convert markdown to HTML first for better parsing
    html = markdown.markdown(markdown_content)
    
    # Split content by sections
    sections = html.split('<h')
    
    for section in sections:
        if not section.strip():
            continue
            
        # Reconstruct the h tag for processing
        if section[0] != '<':
            section = '<h' + section
            
        # Process headings
        if '<h1>' in section:
            text = section.split('</h1>')[0].replace('<h1>', '')
            doc.add_heading(text, level=1)
            content = section.split('</h1>')[1]
        elif '<h2>' in section:
            text = section.split('</h2>')[0].replace('<h2>', '')
            doc.add_heading(text, level=2)
            content = section.split('</h2>')[1]
        else:
            content = section
            
        # Process paragraphs and lists
        paragraphs = content.split('<p>')
        for p in paragraphs:
            if not p.strip():
                continue
                
            # Handle lists
            if '<ul>' in p:
                items = p.split('<li>')
                for item in items[1:]:  # Skip first empty item
                    item_text = item.split('</li>')[0]
                    # Handle links within list items
                    item_text = process_links(item_text)
                    doc.add_paragraph(item_text, style='List Bullet')
            else:
                # Handle regular paragraphs
                p_text = p.replace('</p>', '').strip()
                if p_text:
                    # Handle links within paragraphs
                    p_text = process_links(p_text)
                    doc.add_paragraph(p_text)
    
    doc_io = io.BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io

def process_links(text):
    # Handle both markdown and HTML links
    import re
    
    # First handle markdown links [text](url)
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    def replace_markdown_link(match):
        text, url = match.groups()
        return f"{text} ({url})"
    
    # Then handle HTML links <a href="url">text</a>
    html_pattern = r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>'
    
    def replace_html_link(match):
        url, text = match.groups()
        return f"{text} ({url})"
    
    # Apply both replacements
    text = re.sub(markdown_pattern, replace_markdown_link, text)
    text = re.sub(html_pattern, replace_html_link, text)
    
    # Clean up any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()


## UI Layout ##

# Logo and title
col1, col2 = st.columns([1, 20])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/Heinz_Doofenshmirtz.png/135px-Heinz_Doofenshmirtz.png", width=50)
with col2:
    st.title("📚 Research Roundup-Inator")

# Sidebar for API keys
with st.sidebar:
    st.header("API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    semanticscholar_api_key = st.text_input("Semantic Scholar API Key (optional)", type="password")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool uses Semantic Scholar's API and GPT-4o to generate research roundups from academic papers. Provide your OpenAI API key above to get started \n\n*(don't worry, your key stays secure on your device)* \n\nMade with ❤️ by [Ben Rochford](https://benrochford.com)")

# Main content
tab1, tab2, tab3 = st.tabs(["Search Papers", "Browse Collected Papers", "Generate Roundup"])

with tab1:
    st.header("Search papers with Semantic Scholar")
    search_method = st.radio("Search Method", ["Provide search terms", "Generate search terms automatically"])
    
    # Replace date inputs with year selection
    col1, col2 = st.columns(2)
    current_year = pd.Timestamp.now().year
    with col1:
        start_year = st.number_input(
            "Start Year",
            min_value=1900,
            max_value=current_year,
            value=None,
            placeholder="Optional",
            help="(in or after this year)"
        )
    with col2:
        end_year = st.number_input(
            "End Year",
            min_value=1900,
            max_value=current_year,
            value=None,
            placeholder="Optional",
            help="(in or before this year)"
        )
    
    if start_year and end_year and start_year > end_year:
        st.error("Start year must be before or equal to end year")
        
    # Initialize queries
    queries = []
    
    if search_method == "Provide search terms":
        search_terms = st.text_area("Enter search terms (one per line)")
        queries = [term.strip() for term in search_terms.split('\n') if term.strip()]
    else:
        topic_description = st.text_area(
            "Describe your research topic",
            placeholder="Example: society centered AI and developments in AI safety",
            help="Provide a brief description of the research area you're interested in. Be specific about any particular aspects you want to focus on."
        )
        
        if not topic_description:
            queries = []
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                suggest_clicked = st.button(
                    "🔍 Suggest Terms",
                    disabled=not openai_api_key,
                    help="Click to generate relevant search terms based on your description"
                )
            with col2:
                if not openai_api_key:
                    st.warning("⚠️ Please enter your OpenAI API key in the sidebar first")
        
            if suggest_clicked:
                with st.spinner("Generating search terms..."):
                    suggested_terms = suggest_search_terms(topic_description, openai_api_key)
                    st.session_state.selected_terms = suggested_terms
                    
            if hasattr(st.session_state, 'selected_terms') and st.session_state.selected_terms:
                st.subheader("Confirm auto search terms:")
                current_selections = []
                for term in st.session_state.selected_terms:
                    if st.checkbox(term, value=True, key=f"term_{term}"):
                        current_selections.append(term)
                
                custom_terms = st.text_area(
                    "Add additional search terms (optional, one per line)",
                    placeholder="Enter additional terms, one per line"
                )
                if custom_terms:
                    for term in custom_terms.split('\n'):
                        term = term.strip()
                        if term and term not in current_selections:
                            current_selections.append(term)
                
                queries = current_selections
    
    # Update search button section to use years instead of dates
    if queries and (not (start_year and end_year) or start_year <= end_year):
        st.write("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            search_clicked = st.button(
                "🚀 Search Semantic Scholar",
                disabled=not openai_api_key,
                help="Click to search for papers on Semantic Scholar using the selected terms"
            )
        with col2:
            if not openai_api_key:
                st.warning("⚠️ Please enter your OpenAI API key in the sidebar first")
        
        if search_clicked:
            with st.spinner("Initializing search..."):
                papers, results_by_term = collect_papers(queries, start_year, end_year, semanticscholar_api_key)
                st.session_state.papers_df = process_papers(papers)
                
                # Success message with tab navigation instruction
                st.success(f"✨ Search completed! Found {len(st.session_state.papers_df)} unique papers")
                st.info("👆 Click the 'Browse Collected Papers' tab above to review the results")

with tab2:
    if st.session_state.papers_df is not None:
        st.header("Browse Collected Papers", anchor="review-collected-papers")
        st.dataframe(
            st.session_state.papers_df,
            column_config={
                "title": st.column_config.TextColumn("Title", width="large"),
                "url": st.column_config.LinkColumn("URL"),
                "abstract": st.column_config.TextColumn("Abstract", width="large"),
            },
            hide_index=True,
        )
    else:
        st.info("No papers collected yet. Use the Search tab to find papers.")

with tab3:
    if st.session_state.papers_df is not None:
        st.header("Generate Roundup")

        # Compute and display top 35 papers with relevancy scores
        st.session_state.papers_df['score'] = st.session_state.papers_df['citationCount'].fillna(0) + \
            100*(1 / (1 + (pd.Timestamp.now() - st.session_state.papers_df['publicationDate']).dt.days))
        
        top_papers_df = st.session_state.papers_df.sort_values('score', ascending=False).head(35)
        top_papers_df = top_papers_df[['score'] + [col for col in top_papers_df.columns if col != 'score']]
        
        with st.expander("📄 Papers Selected for Roundup", expanded=False):
            st.write("The top 35 most relevant papers selected for the roundup, based on recency and citation count")
            st.dataframe(top_papers_df)
        
        # Add prompt customization with expander
        with st.expander("📝 Customize Generation Prompt", expanded=False):
            
            col1, col2 = st.columns([3, 10])
            with col1:
                st.markdown("#### Roundup Prompt")
            with col2:
                if st.button("reset to default", type="secondary", key="reset_prompt"):
                    st.session_state.custom_prompt = DEFAULT_ROUNDUP_PROMPT
            
            custom_prompt = st.text_area(
                "Generation Prompt",
                value=st.session_state.custom_prompt,
                height=400,
                help="Edit this prompt to customize how your research roundup is generated",
                label_visibility="collapsed"
            )
            
            # Save any changes to the prompt
            if custom_prompt != st.session_state.custom_prompt:
                st.session_state.custom_prompt = custom_prompt
        
        if st.button("Generate Research Roundup") and openai_api_key:
            progress_placeholder = st.empty()
            message_placeholder = st.empty()
            
            with progress_placeholder:
                with st.spinner("Generating roundup..."):
                    # Start newsletter generation in background
                    newsletter_content = None
                    start_time = time.time()
                    
                    # Keep showing messages until generation is complete
                    message_index = 0
                    while newsletter_content is None:
                        # Cycle through messages
                        dots = "." * ((message_index % 3) + 1)
                        message = LOADING_MESSAGES[message_index % len(LOADING_MESSAGES)]
                        message_placeholder.markdown(f"*{message}{dots}*")
                        
                        # Generate newsletter on first iteration
                        if message_index == 0:
                            newsletter_content = generate_newsletter(
                                st.session_state.papers_df,
                                openai_api_key,
                                st.session_state.custom_prompt
                            )
                        
                        message_index += 1
                        time.sleep(0.8)
                    
                    st.session_state.newsletter_content = newsletter_content
            
            # Clear the loading message
            message_placeholder.empty()
        if st.session_state.newsletter_content:
            st.markdown("---")
            st.markdown(st.session_state.newsletter_content)
            st.markdown("---")

            # Download buttons
            col1, col2, col3, col4 = st.columns([4,3,3,4])
            with col2:
                st.download_button(
                    "Download as Markdown", 
                    st.session_state.newsletter_content,
                    file_name="research_roundup.md",
                    mime="text/markdown"
                )
            with col3:
                docx_file = convert_to_docx(st.session_state.newsletter_content)
                st.download_button(
                    "Download as Word",
                    docx_file, 
                    file_name="research_roundup.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    else:
        st.info("No papers collected yet. Use the Search tab to find papers.")
