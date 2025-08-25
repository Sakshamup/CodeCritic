import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
import json
import re

# Set page config
st.set_page_config(
    page_title="AI Code Reviewer",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'review_history' not in st.session_state:
    st.session_state.review_history = []

@st.cache_resource
def initialize_llm(api_key):
    """Initialize Gemini LLM with LangChain"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def create_review_prompt():
    """Create prompt template for code review with readable format"""
    template = """
    You are an expert code reviewer. Analyze the following code and provide a comprehensive, human-readable review.

    Code Language: {language}
    
    Code:
    ```{language}
    {code}
    ```

    Please provide your review in a clear, readable format with the following sections:

    ## OVERALL ASSESSMENT
    Score: [X/10]
    Summary: [Brief overall assessment]

    ## STRENGTHS
    - [List what's done well]
    - [Good practices found]

    ## ISSUES FOUND
    ### High Priority Issues:
    - **[Issue Type]** (Line X): [Description]
      → Fix: [Suggestion]

    ### Medium Priority Issues:
    - **[Issue Type]** (Line X): [Description]
      → Fix: [Suggestion]

    ### Low Priority Issues:
    - **[Issue Type]** (Line X): [Description]
      → Fix: [Suggestion]

    ## IMPROVEMENT SUGGESTIONS
    - [General improvements]
    - [Best practices recommendations]

    ## DOCUMENTATION RECOMMENDATIONS
    - [Documentation suggestions]
    - [Comment improvements]

    Focus on:
    - Code quality and best practices
    - Potential bugs and security vulnerabilities
    - Performance optimizations
    - Readability and maintainability
    - Error handling
    - Documentation quality

    Provide specific, actionable feedback that a developer can immediately use to improve their code.
    """
    
    return PromptTemplate(
        input_variables=["language", "code"],
        template=template
    )

def create_advanced_agent(llm):
    """Create an advanced LangChain agent for detailed code analysis"""
    
    def analyze_code_quality(code_and_lang):
        """Comprehensive code quality analysis"""
        try:
            code, language = code_and_lang.split("|||")
            prompt = f"""
            Analyze this {language} code for quality, bugs, and improvements:
            
            {code}
            
            Provide analysis in this format:
            
            QUALITY_SCORE: X/10
            SUMMARY: One sentence summary
            STRENGTHS: List what's good
            CRITICAL_ISSUES: List critical problems with line numbers
            SUGGESTIONS: List improvements
            """
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error in analysis: {str(e)}"
    
    def check_security_issues(code_and_lang):
        """Check for security vulnerabilities"""
        try:
            code, language = code_and_lang.split("|||")
            prompt = f"""
            Check this {language} code for security vulnerabilities:
            
            {code}
            
            Focus on:
            - SQL injection risks
            - XSS vulnerabilities  
            - Input validation issues
            - Authentication/authorization problems
            
            List any security issues found with specific line references.
            """
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error in security check: {str(e)}"
    
    def suggest_optimizations(code_and_lang):
        """Suggest performance optimizations"""
        try:
            code, language = code_and_lang.split("|||")
            prompt = f"""
            Suggest performance optimizations for this {language} code:
            
            {code}
            
            Focus on:
            - Algorithm efficiency
            - Memory usage
            - Database query optimization
            - Loop improvements
            
            Provide specific suggestions with examples.
            """
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error in optimization analysis: {str(e)}"
    
    tools = [
        Tool(
            name="CodeQualityAnalyzer",
            func=analyze_code_quality,
            description="Analyzes overall code quality, bugs, and structure. Input format: 'code|||language'"
        ),
        Tool(
            name="SecurityChecker", 
            func=check_security_issues,
            description="Checks for security vulnerabilities and risks. Input format: 'code|||language'"
        ),
        Tool(
            name="PerformanceOptimizer",
            func=suggest_optimizations,
            description="Suggests performance improvements and optimizations. Input format: 'code|||language'"
        )
    ]
    
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False
    )
    
    return agent

def review_with_advanced_agent(agent, code, language):
    """Use advanced agent for comprehensive review"""
    try:
        query = f"""
        Please perform a comprehensive code review of this {language} code using all your available tools.
        
        1. First, analyze the overall code quality
        2. Then, check for security issues
        3. Finally, suggest performance optimizations
        
        Combine all findings into a final structured report with:
        - Overall score
        - Summary of findings
        - List of all issues by priority
        - Actionable recommendations
        
        Code:
        {code}
        """
        
        response = agent.run(query)
        return response, None
        
    except Exception as e:
        return None, str(e)

def create_simple_readable_prompt():
    """Create a simple prompt that generates very readable output"""
    template = """
    You are an expert code reviewer. Analyze this {language} code and provide feedback in this EXACT format:

    SCORE: X/10

    SUMMARY: Brief assessment of the code quality in one sentence.

    STRENGTHS:
    - What the code does well
    - Good practices used

    HIGH_PRIORITY_ISSUES:
    - Critical issue description (Line number) | Fix: How to resolve it
    - Another critical issue if any

    MEDIUM_PRIORITY_ISSUES:
    - Important issue description (Line number) | Fix: How to resolve it
    - Another medium issue if any

    LOW_PRIORITY_ISSUES:
    - Minor issue description (Line number) | Fix: How to resolve it
    - Another minor issue if any

    IMPROVEMENTS:
    - Suggestion for better code
    - Performance or design improvements

    DOCUMENTATION:
    - Documentation improvements needed
    - Comment suggestions

    Code to analyze:
    ```{language}
    {code}
    ```

    Follow the format EXACTLY as shown above. Use simple bullet points with dashes.
    """
    
    return PromptTemplate(
        input_variables=["language", "code"],
        template=template
    )

def parse_and_display_results(review_content):
    """Parse and display results in a clean, readable format"""
    
    st.markdown("### 📊 Code Review Results")
    st.markdown("---")
    
    # Split content into lines
    lines = [line.strip() for line in review_content.split('\n') if line.strip()]
    
    current_section = None
    
    for line in lines:
        # Handle section headers
        if line.startswith('SCORE:'):
            score_text = line.replace('SCORE:', '').strip()
            st.metric("🎯 Code Quality", score_text)
            st.markdown("")
            
        elif line.startswith('SUMMARY:'):
            summary = line.replace('SUMMARY:', '').strip()
            st.info(f"**📋 {summary}**")
            st.markdown("")
            
        elif line.startswith('STRENGTHS:'):
            st.markdown("### ✅ **Strengths**")
            current_section = "STRENGTHS"
            
        elif line.startswith('HIGH_PRIORITY_ISSUES:'):
            st.markdown("### 🔴 **High Priority Issues**")
            current_section = "HIGH_PRIORITY"
            
        elif line.startswith('MEDIUM_PRIORITY_ISSUES:'):
            st.markdown("### 🟡 **Medium Priority Issues**") 
            current_section = "MEDIUM_PRIORITY"
            
        elif line.startswith('LOW_PRIORITY_ISSUES:'):
            st.markdown("### 🟢 **Low Priority Issues**")
            current_section = "LOW_PRIORITY"
            
        elif line.startswith('IMPROVEMENTS:'):
            st.markdown("### 💡 **Suggestions for Improvement**")
            current_section = "IMPROVEMENTS"
            
        elif line.startswith('DOCUMENTATION:'):
            st.markdown("### 📝 **Documentation Recommendations**")
            current_section = "DOCUMENTATION"
            
        # Handle bullet points
        elif line.startswith('- '):
            content = line[2:].strip()  # Remove "- "
            
            if current_section == "STRENGTHS":
                st.success(f"✓ {content}")
                
            elif current_section == "HIGH_PRIORITY":
                if '|' in content:
                    issue, fix = content.split('|', 1)
                    st.error(f"**Issue:** {issue.strip()}")
                    st.markdown(f"   💡 **{fix.strip()}**")
                else:
                    st.error(f"**Issue:** {content}")
                st.markdown("")
                
            elif current_section == "MEDIUM_PRIORITY":
                if '|' in content:
                    issue, fix = content.split('|', 1)
                    st.warning(f"**Issue:** {issue.strip()}")
                    st.markdown(f"   💡 **{fix.strip()}**")
                else:
                    st.warning(f"**Issue:** {content}")
                st.markdown("")
                
            elif current_section == "LOW_PRIORITY":
                if '|' in content:
                    issue, fix = content.split('|', 1)
                    st.info(f"**Issue:** {issue.strip()}")
                    st.markdown(f"   💡 **{fix.strip()}**")
                else:
                    st.info(f"**Issue:** {content}")
                st.markdown("")
                
            elif current_section in ["IMPROVEMENTS", "DOCUMENTATION"]:
                st.info(f"• {content}")
    
    # Fallback if parsing fails
    if not any(section in review_content for section in ['SCORE:', 'STRENGTHS:', 'IMPROVEMENTS:']):
        st.markdown("### 📄 Raw Review Results")
        st.markdown(review_content)

def review_code(llm, code, language):
    """Review code using simple LLM approach with readable format"""
    try:
        prompt_template = create_simple_readable_prompt()
        prompt = prompt_template.format(language=language, code=code)
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content, None
            
    except Exception as e:
        return None, str(e)

# Main app
def main():
    st.title("🔍 AI Code Reviewer")
    st.markdown("Get instant code reviews powered by Google Gemini")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if api_key:
            st.success("API Key configured!")
        else:
            st.warning("Please enter your Google API key to continue")
            st.markdown("""
            **How to get API Key:**
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Create a new API key
            3. Copy and paste it here
            """)
        
        st.markdown("---")
        st.header("Review Mode")
        use_agent = st.checkbox(
            "Use AI Agent (Advanced)", 
            value=False,
            help="Uses LangChain agent with multiple specialized tools for deeper analysis"
        )
        
        if use_agent:
            st.info("🤖 Agent mode: Uses multiple AI tools for comprehensive analysis")
        else:
            st.info("⚡ Simple mode: Fast single-pass review")
        
        st.markdown("---")
        st.markdown("### Supported Languages")
        st.markdown("""
        - Python
        - JavaScript
        - Java
        - C++
        - C#
        - Go
        - Rust
        - PHP
        - Ruby
        - And more!
        """)
    
    # Main content
    if not api_key:
        st.info("👈 Please configure your Google API key in the sidebar to get started")
        return
    
    # Initialize LLM
    llm = initialize_llm(api_key)
    if not llm:
        st.error("Failed to initialize the AI model. Please check your API key.")
        return
    
    # Code input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Code Input")
        
        # Language selection
        languages = [
            "python", "javascript", "java", "cpp", "csharp", 
            "go", "rust", "php", "ruby", "typescript", "html", "css"
        ]
        selected_language = st.selectbox("Select Language", languages, index=0)
        
        # Code input
        code_input = st.text_area(
            "Paste your code here:",
            height=400,
            placeholder="Enter your code here for review..."
        )
        
        # Review button
        if st.button("🔍 Review Code", type="primary"):
            if not code_input.strip():
                st.warning("Please enter some code to review!")
                return
            
            with st.spinner("Analyzing your code..." + (" (Using AI Agent)" if use_agent else "")):
                if use_agent:
                    # Use advanced LangChain agent
                    try:
                        agent = create_advanced_agent(llm)
                        review_result, error = review_with_advanced_agent(agent, code_input, selected_language)
                    except Exception as e:
                        st.error(f"Agent failed: {str(e)}. Using simple mode.")
                        review_result, error = review_code(llm, code_input, selected_language)
                else:
                    # Use simple review approach
                    review_result, error = review_code(llm, code_input, selected_language)
                
                if error:
                    st.error(f"Error during review: {error}")
                elif review_result:
                    # Store in history
                    st.session_state.review_history.append({
                        'language': selected_language,
                        'code': code_input[:100] + "..." if len(code_input) > 100 else code_input,
                        'review': review_result,
                        'agent_used': use_agent,
                        'timestamp': st.experimental_get_query_params()
                    })
                    
                    # Display results using the new parser
                    st.success("Review completed!" + (" (AI Agent)" if use_agent else ""))
                    parse_and_display_results(review_result)
                else:
                    st.error("Failed to get review results")
    
    with col2:
        st.header("Review History")
        
        if st.session_state.review_history:
            for i, history_item in enumerate(reversed(st.session_state.review_history[-5:])):
                mode_indicator = "🤖" if history_item.get('agent_used', False) else "⚡"
                with st.expander(f"{mode_indicator} {history_item['language']} - Review {len(st.session_state.review_history) - i}"):
                    st.code(history_item['code'], language=history_item['language'])
                    if st.button(f"View Full Review {i}", key=f"view_{i}"):
                        parse_and_display_results(history_item['review'])
        else:
            st.info("No reviews yet. Submit your first code review!")
        
        # Clear history button
        if st.session_state.review_history:
            if st.button("Clear History"):
                st.session_state.review_history = []
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [Streamlit](https://streamlit.io), "
        "[LangChain](https://langchain.com), and [Google Gemini](https://ai.google.dev)"
    )

if __name__ == "__main__":
    main()
