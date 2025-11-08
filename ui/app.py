"""
Streamlit UI for RAG AI Agent.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.main import RAGPipeline
from src.eval.metrics import evaluate_answer_quality

# Page configuration
st.set_page_config(
    page_title="RAG AI Agent",
    page_icon=None,
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    if st.session_state.pipeline is None:
        with st.spinner("Initializing RAG pipeline..."):
            try:
                st.session_state.pipeline = RAGPipeline()
                st.success("Pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing pipeline: {e}")
                return False
    return True


def main():
    """Main Streamlit app."""
    st.title("RAG AI Agent - Q&A System")
    st.markdown("Ask questions and get answers based on the knowledge base.")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Initialize pipeline button
        if st.button("Initialize Pipeline", use_container_width=True):
            st.session_state.pipeline = None
            initialize_pipeline()
        
        st.divider()
        
        st.header("About")
        st.markdown("""
        This RAG (Retrieval-Augmented Generation) agent:
        - **Plans** whether retrieval is needed
        - **Retrieves** relevant documents
        - **Answers** using LLM with context
        - **Reflects** on answer quality
        
        Built with LangChain, ChromaDB, and Streamlit.
        """)
        
        st.divider()
        
        # Clear conversation history
        if st.button("Clear History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Initialize pipeline
    if not initialize_pipeline():
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask a Question")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is RAG? How does it work?",
            key="question_input"
        )
        
        # Submit button
        if st.button("Get Answer", type="primary", use_container_width=True):
            if question:
                with st.spinner("Processing your question..."):
                    try:
                        # Run pipeline
                        result = st.session_state.pipeline.run(question)
                        
                        # Store in conversation history
                        st.session_state.conversation_history.append({
                            "question": question,
                            "result": result
                        })
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {e}")
            else:
                st.warning("Please enter a question.")
        
        st.divider()
        
        # Display conversation history
        st.header("Conversation History")
        
        if st.session_state.conversation_history:
            for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q: {conv['question']}", expanded=(i == 0)):
                    result = conv['result']
                    
                    if result.get('success'):
                        # Answer
                        st.markdown("### Answer")
                        st.write(result.get('answer', {}).get('answer', ''))
                        
                        # Reflection
                        reflection = result.get('reflection', {})
                        st.markdown("### Reflection")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Verdict", reflection.get('verdict', 'unknown'))
                        with col_b:
                            st.metric("Score", f"{reflection.get('score', 0.0):.2f}")
                        st.caption(reflection.get('feedback', ''))
                        
                        # Details
                        with st.expander("Pipeline Details"):
                            # Planner
                            planner = result.get('planner', {})
                            st.markdown("#### Planner")
                            st.write(f"**Retrieval Needed:** {planner.get('retrieval_needed', False)}")
                            st.caption(planner.get('reason', ''))
                            
                            # Retrieval
                            retrieval = result.get('retrieval', {})
                            st.markdown("#### Retrieval")
                            st.write(f"**Documents Retrieved:** {len(retrieval.get('docs', []))}")
                            if retrieval.get('docs'):
                                st.write("**Sources:**")
                                for doc in retrieval.get('docs', [])[:3]:
                                    source = doc.get('metadata', {}).get('source', 'Unknown')
                                    st.caption(f"• {source}")
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            st.info("No conversation history. Ask a question to get started!")
    
    with col2:
        st.header("Pipeline Stats")
        
        if st.session_state.conversation_history:
            total_questions = len(st.session_state.conversation_history)
            st.metric("Total Questions", total_questions)
            
            # Calculate average scores
            scores = []
            for conv in st.session_state.conversation_history:
                result = conv.get('result', {})
                if result.get('success'):
                    reflection = result.get('reflection', {})
                    score = reflection.get('score', 0.0)
                    scores.append(score)
            
            if scores:
                avg_score = sum(scores) / len(scores)
                st.metric("Average Score", f"{avg_score:.2f}")
                
                # Verdict distribution
                st.markdown("### Verdict Distribution")
                verdicts = {}
                for conv in st.session_state.conversation_history:
                    result = conv.get('result', {})
                    if result.get('success'):
                        verdict = result.get('reflection', {}).get('verdict', 'unknown')
                        verdicts[verdict] = verdicts.get(verdict, 0) + 1
                
                for verdict, count in verdicts.items():
                    st.write(f"**{verdict}:** {count}")
        else:
            st.info("No stats yet. Ask a question to see metrics!")


if __name__ == "__main__":
    main()