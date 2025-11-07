"""
Test RAG-powered deep analysis functionality
"""

import sys
sys.setrecursionlimit(5000)

import pandas as pd
from modules.cost_parser import CostDataParser
from modules.recommendation_engine import RecommendationEngine
from app import SimpleEmbeddings, get_llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("=" * 60)
print("Testing RAG-Powered Deep Analysis")
print("=" * 60)

# Step 1: Parse data
print("\n1. Parsing cost data...")
parser = CostDataParser()
df = pd.read_csv('sample_data/aws_cost_export.csv')
cost_data = parser.parse_csv(df)
print(f"   ✅ Parsed {len(cost_data['raw_data'])} records")

# Step 2: Generate recommendations
print("\n2. Generating recommendations...")
rec_engine = RecommendationEngine(cost_data, threshold=10.0)
recommendations = rec_engine.generate_recommendations()
print(f"   ✅ Generated {len(recommendations)} recommendations")

# Step 3: Prepare text for RAG
print("\n3. Preparing text for RAG...")
text_content = parser.prepare_text_for_rag(cost_data, recommendations)
print(f"   ✅ Prepared {len(text_content)} characters of text")

# Step 4: Chunking
print("\n4. Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_text(text_content)
print(f"   ✅ Created {len(chunks)} chunks")

# Step 5: Create embeddings
print("\n5. Creating embeddings...")
embedding_model = SimpleEmbeddings()
test_embedding = embedding_model.embed_documents(["test"])
print(f"   ✅ Embeddings working (dimension: {len(test_embedding[0])})")

# Step 6: Create vector store
print("\n6. Creating vector store...")
try:
    vectordb = Chroma.from_texts(
        chunks,
        embedding_model,
        persist_directory="./test_chroma_index"
    )
    print(f"   ✅ Vector store created with {len(chunks)} documents")
except Exception as e:
    print(f"   ❌ Vector store failed: {e}")
    sys.exit(1)

# Step 7: Test retrieval
print("\n7. Testing retrieval...")
try:
    retriever = vectordb.as_retriever()
    docs = retriever.invoke("What are the top cost savings?")
    print(f"   ✅ Retrieved {len(docs)} relevant documents")
except Exception as e:
    print(f"   ❌ Retrieval failed: {e}")
    sys.exit(1)

# Step 8: Test LLM (without actual API call)
print("\n8. Testing LLM initialization...")
try:
    llm = get_llm()
    print(f"   ✅ LLM initialized: {llm.model_name}")
except Exception as e:
    print(f"   ⚠️  LLM initialization: {e}")
    print("   (This is OK if API is not accessible)")

# Step 9: Test RAG chain construction
print("\n9. Testing RAG chain construction...")
try:
    template = """Answer based on context:

Context: {context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("   ✅ RAG chain constructed successfully")
    print("   ✅ Chain components: retriever -> prompt -> LLM -> parser")
    
except Exception as e:
    print(f"   ❌ RAG chain construction failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ RAG-Powered Deep Analysis: ALL TESTS PASSED!")
print("=" * 60)
print("\nRAG functionality is working correctly!")
print("Note: Actual LLM API call not tested (requires network access)")
print("\nComponents verified:")
print("  ✅ Data parsing")
print("  ✅ Text chunking")
print("  ✅ Simple embeddings")
print("  ✅ Vector store (ChromaDB)")
print("  ✅ Document retrieval")
print("  ✅ RAG chain construction")
