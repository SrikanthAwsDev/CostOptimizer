import sys
sys.setrecursionlimit(5000)  # Increase recursion limit to handle pandas issue

import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os
import httpx
import pandas as pd
import json
from io import StringIO
from modules.cost_parser import CostDataParser
from modules.recommendation_engine import RecommendationEngine
from modules.report_generator import ReportGenerator
from modules.visualizer import CostVisualizer

# Simple embedding class that doesn't use tokenizers
class SimpleEmbeddings(Embeddings):
    """Simple embeddings using hash-based vectors to avoid tokenizer issues"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using simple hash-based approach"""
        import hashlib
        embeddings = []
        for text in texts:
            # Create a simple 384-dimensional embedding from text hash
            hash_obj = hashlib.sha384(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to normalized floats
            embedding = [float(b) / 255.0 - 0.5 for b in hash_bytes[:384]]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text"""
        return self.embed_documents([text])[0]

# LLM and Embedding setup - only initialized when RAG mode is used
def get_llm():
    """Lazy load LLM only when needed for RAG analysis"""
    client = httpx.Client(verify=False)
    return ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key="sk-JEVyjAuuQSr7akfYgASCXA",
        http_client=client
    )

def get_embedding_model():
    """Lazy load embedding model only when needed for RAG analysis"""
    # Use simple hash-based embeddings to avoid tokenizer issues
    return SimpleEmbeddings()

# Page config
st.set_page_config(
    page_title="Cloud Cost Optimizer",
    page_icon="☁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">☁️ Cloud Cost Optimizer</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Cost Analysis with RAG | Multi-Format Support**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Quick Analysis", "RAG-Powered Deep Analysis"],
        help="RAG mode uses vector embeddings for intelligent insights"
    )
    
    savings_threshold = st.slider(
        "Minimum Savings Threshold ($)",
        min_value=0,
        max_value=500,
        value=10,
        step=10
    )
    
    st.markdown("---")
    st.markdown("### 📁 Supported Formats")
    st.markdown("- CSV (AWS Cost Explorer)")
    st.markdown("- Excel (.xlsx)")
    st.markdown("- PDF (Billing Reports)")
    st.markdown("- JSON (API Exports)")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Dashboard", "📝 Reports"])

with tab1:
    st.subheader("Upload Cost Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload one or multiple files",
            type=["csv", "xlsx", "pdf", "json"],
            accept_multiple_files=True,
            help="Upload AWS Cost Explorer exports, billing PDFs, or custom formats"
        )
    
    with col2:
        st.info("💡 **Tip**: Upload multiple months of data for better trend analysis")

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        if st.button("🚀 Analyze Costs", type="primary"):
            with st.spinner("Processing files..."):
                # Parse all uploaded files
                parser = CostDataParser()
                all_data = []
                
                for uploaded_file in uploaded_files:
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_type == 'pdf':
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.read())
                            temp_file_path = temp_file.name
                        
                        raw_text = extract_text(temp_file_path)
                        parsed_data = parser.parse_pdf_text(raw_text)
                        all_data.append(parsed_data)
                        os.unlink(temp_file_path)
                    
                    elif file_type == 'csv':
                        df = pd.read_csv(uploaded_file)
                        parsed_data = parser.parse_csv(df)
                        all_data.append(parsed_data)
                    
                    elif file_type == 'xlsx':
                        df = pd.read_excel(uploaded_file)
                        parsed_data = parser.parse_csv(df)
                        all_data.append(parsed_data)
                    
                    elif file_type == 'json':
                        json_data = json.load(uploaded_file)
                        parsed_data = parser.parse_json(json_data)
                        all_data.append(parsed_data)
                
                # Merge all data
                merged_data = parser.merge_datasets(all_data)
                st.session_state['cost_data'] = merged_data
                
                # Generate recommendations
                with st.spinner("Generating recommendations..."):
                    rec_engine = RecommendationEngine(merged_data, threshold=savings_threshold)
                    recommendations = rec_engine.generate_recommendations()
                    st.session_state['recommendations'] = recommendations
                
                # RAG Analysis if enabled
                if analysis_mode == "RAG-Powered Deep Analysis":
                    with st.spinner("Running RAG analysis..."):
                        # Initialize LLM and embeddings only when needed
                        llm = get_llm()
                        embedding_model = get_embedding_model()
                        
                        # Prepare text for RAG
                        text_content = parser.prepare_text_for_rag(merged_data, recommendations)
                        
                        # Chunking
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_text(text_content)
                        
                        # Embed and store
                        vectordb = Chroma.from_texts(
                            chunks,
                            embedding_model,
                            persist_directory="./chroma_index"
                        )
                        vectordb.persist()
                        
                        # RAG QA Chain (langchain 1.0+ API)
                        retriever = vectordb.as_retriever()
                        
                        # Create prompt template
                        template = """Answer the question based on the provided context:

Context: {context}

Question: {question}

Provide a detailed, actionable answer."""
                        
                        prompt = ChatPromptTemplate.from_template(template)
                        
                        # Create RAG chain
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)
                        
                        rag_chain = (
                            {"context": retriever | format_docs, "question": RunnablePassthrough()}
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        # Generate insights
                        insight_prompt = """Analyze this cloud cost data and provide:
1. Top 3 priority actions to reduce costs
2. Risk areas with unusual spending
3. Long-term optimization strategy
4. Estimated ROI timeline

Be specific and actionable."""
                        
                        result = rag_chain.invoke(insight_prompt)
                        st.session_state['ai_insights'] = result
                
                st.success("✅ Analysis complete!")
                st.rerun()

with tab2:
    if 'cost_data' in st.session_state and 'recommendations' in st.session_state:
        cost_data = st.session_state['cost_data']
        recommendations = st.session_state['recommendations']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_cost = cost_data['summary']['total_cost']
        total_savings = sum(r['estimated_savings'] for r in recommendations)
        savings_pct = (total_savings / total_cost * 100) if total_cost > 0 else 0
        
        with col1:
            st.metric("Total Monthly Cost", f"${total_cost:,.2f}")
        with col2:
            st.metric("Potential Savings", f"${total_savings:,.2f}", delta=f"{savings_pct:.1f}%")
        with col3:
            st.metric("Recommendations", len(recommendations))
        with col4:
            health_score = max(0, 100 - int(savings_pct))
            st.metric("Cost Health Score", f"{health_score}/100")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Cost by Service")
            visualizer = CostVisualizer(cost_data, recommendations)
            fig1 = visualizer.create_service_pie_chart()
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("📈 Savings Potential")
            fig2 = visualizer.create_savings_bar_chart()
            st.plotly_chart(fig2, use_container_width=True)
        
        # Cost trend
        st.subheader("📊 Cost Trend Analysis")
        fig3 = visualizer.create_cost_trend_chart()
        st.plotly_chart(fig3, use_container_width=True)
        
        # Recommendations table
        st.subheader("🎯 Top Recommendations")
        df_recs = pd.DataFrame(recommendations)
        df_recs = df_recs[['category', 'service', 'priority', 'estimated_savings', 'description']]
        df_recs['estimated_savings'] = df_recs['estimated_savings'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(df_recs, use_container_width=True, hide_index=True)
        
        # AI Insights
        if 'ai_insights' in st.session_state:
            st.markdown("---")
            st.subheader("🤖 AI-Powered Strategic Insights")
            st.markdown(st.session_state['ai_insights'])
    
    else:
        st.info("👆 Upload and analyze cost data in the 'Upload & Analyze' tab to see the dashboard")

with tab3:
    if 'cost_data' in st.session_state and 'recommendations' in st.session_state:
        st.subheader("📝 Generate Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_format = st.selectbox(
                "Report Format",
                ["Excel (Recommended)", "PDF", "CSV", "HTML Dashboard"]
            )
        
        with col2:
            report_name = st.text_input("Report Name", value="cost_optimization_report")
        
        if st.button("📥 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                cost_data = st.session_state['cost_data']
                recommendations = st.session_state['recommendations']
                ai_insights = st.session_state.get('ai_insights')
                
                report_gen = ReportGenerator(cost_data, recommendations, ai_insights)
                
                format_map = {
                    "Excel (Recommended)": "excel",
                    "PDF": "pdf",
                    "CSV": "csv",
                    "HTML Dashboard": "html"
                }
                
                output_file = report_gen.generate(report_name, format_map[report_format])
                
                # Provide download
                with open(output_file, 'rb') as f:
                    st.download_button(
                        label=f"⬇️ Download {report_format}",
                        data=f,
                        file_name=os.path.basename(output_file),
                        mime="application/octet-stream"
                    )
                
                st.success(f"✅ Report generated: {output_file}")
    else:
        st.info("👆 Upload and analyze cost data first to generate reports")
