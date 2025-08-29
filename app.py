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
import datetime

# Custom CSS for baby pink theme and improved UI
st.markdown("""
<style>
    /* Main app background - baby pink */
    .stApp {
        background-color: #f01347 !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #426ff5 !important;
        padding: 1rem;
    }
    
    /* Sidebar styling - light pink */
    .css-1d391kg, .css-1lcbmhc, .css-17lntkn {
        background-color: #3d50f5 !important;
        border-right: 1px solid #f8bbd9;
    }
    
    /* Sidebar content */
    .sidebar .sidebar-content {
        background-color: #2f3cf5 !important;
        padding: 1rem;
    }
    
    /* Alternative sidebar selectors for different Streamlit versions */
    div[data-testid="stSidebar"] {
        background-color: #2f3cf5 !important;
    }
    
    div[data-testid="stSidebar"] > div {
        background-color: #2f3cf5 !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #2a30db 0%, #e91e63 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(248, 187, 217, 0.3);
    }
    
    /* Card styling */
    .review-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(248, 187, 217, 0.2);
        margin: 1rem 0;
    }
    
    /* Code input styling */
    .code-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(248, 187, 217, 0.15);
        margin: 1rem 0;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(248, 187, 217, 0.2);
        margin: 0.5rem;
    }
    
    /* Complexity indicators */
    .complexity-low { background: #11f22b; }
    .complexity-medium { background: #f27d16; }
    .complexity-high { background: #f2071d; }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #2a30db 0%, #e91e63 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(248, 187, 217, 0.4);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #2a30db, #e91e63);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* History items */
    .history-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    /* Alert styling */
    .stAlert > div {
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="CodeCritic AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
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

def analyze_complexity(code, language):
    """Analyze code complexity metrics"""
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    total_lines = len(lines)
    
    # Count various complexity indicators
    nested_loops = 0
    conditional_statements = 0
    function_definitions = 0
    class_definitions = 0
    
    # Language-specific keywords
    if language.lower() == 'python':
        loop_keywords = ['for', 'while']
        condition_keywords = ['if', 'elif', 'else']
        function_keywords = ['def']
        class_keywords = ['class']
    elif language.lower() in ['javascript', 'typescript']:
        loop_keywords = ['for', 'while', 'do']
        condition_keywords = ['if', 'else']
        function_keywords = ['function', 'const', 'let', 'var']
        class_keywords = ['class']
    elif language.lower() == 'java':
        loop_keywords = ['for', 'while', 'do']
        condition_keywords = ['if', 'else']
        function_keywords = ['public', 'private', 'protected']
        class_keywords = ['class', 'interface']
    else:
        # Generic approach
        loop_keywords = ['for', 'while', 'do']
        condition_keywords = ['if', 'else']
        function_keywords = ['function', 'def', 'public', 'private']
        class_keywords = ['class']
    
    nesting_level = 0
    max_nesting = 0
    
    for line in lines:
        # Count indentation level (rough estimate of nesting)
        indent = len(line) - len(line.lstrip())
        current_nesting = indent // 4  # Assuming 4-space indentation
        max_nesting = max(max_nesting, current_nesting)
        
        # Count keywords
        words = line.lower().split()
        for word in words:
            if word in loop_keywords:
                nested_loops += 1
            elif word in condition_keywords:
                conditional_statements += 1
            elif word in function_keywords:
                function_definitions += 1
            elif word in class_keywords:
                class_definitions += 1
    
    # Calculate complexity score
    complexity_score = (
        (nested_loops * 2) +
        (conditional_statements * 1.5) +
        (max_nesting * 3) +
        (function_definitions * 0.5) +
        (class_definitions * 1)
    )
    
    # Determine complexity level
    if complexity_score < 10:
        complexity_level = "Low"
        complexity_class = "complexity-low"
    elif complexity_score < 25:
        complexity_level = "Medium"
        complexity_class = "complexity-medium"
    else:
        complexity_level = "High"
        complexity_class = "complexity-high"
    
    return {
        'total_lines': total_lines,
        'complexity_score': round(complexity_score, 1),
        'complexity_level': complexity_level,
        'complexity_class': complexity_class,
        'nested_loops': nested_loops,
        'conditional_statements': conditional_statements,
        'function_definitions': function_definitions,
        'class_definitions': class_definitions,
        'max_nesting': max_nesting
    }

def create_complexity_analysis_prompt():
    """Create prompt for AI-based complexity analysis"""
    template = """
    Analyze the complexity of this {language} code and provide detailed insights:

    Code:
    ```{language}
    {code}
    ```

    Please analyze and provide:

    COGNITIVE_COMPLEXITY: Rate from 1-10 based on how hard it is to understand
    CYCLOMATIC_COMPLEXITY: Estimate based on decision points and branches
    MAINTAINABILITY: Rate from 1-10 how easy it would be to modify
    
    COMPLEXITY_FACTORS:
    - Deep nesting levels
    - Complex conditional logic
    - Long methods/functions
    - Multiple responsibilities
    - Unclear variable names
    
    SIMPLIFICATION_SUGGESTIONS:
    - Specific recommendations to reduce complexity
    - Refactoring opportunities
    - Design pattern suggestions
    
    TIME_COMPLEXITY: Estimate the algorithmic time complexity (O notation)
    SPACE_COMPLEXITY: Estimate the space complexity (O notation)
    
    Provide specific, actionable recommendations to improve code complexity.
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
    
    def analyze_complexity_ai(code_and_lang):
        """AI-powered complexity analysis"""
        try:
            code, language = code_and_lang.split("|||")
            prompt_template = create_complexity_analysis_prompt()
            prompt = prompt_template.format(language=language, code=code)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error in complexity analysis: {str(e)}"
    
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
        ),
        Tool(
            name="ComplexityAnalyzer",
            func=analyze_complexity_ai,
            description="Analyzes code complexity and suggests simplifications. Input format: 'code|||language'"
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
        3. Next, suggest performance optimizations
        4. Finally, analyze code complexity and suggest simplifications
        
        Combine all findings into a final structured report with:
        - Overall score
        - Summary of findings
        - List of all issues by priority
        - Complexity analysis
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

def display_complexity_metrics(complexity_data):
    """Display complexity metrics in an attractive format"""
    st.markdown("### 📊 Complexity Analysis")
    
    # Main complexity indicator
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card {complexity_data['complexity_class']}">
            <h3>🎯 Overall</h3>
            <h2>{complexity_data['complexity_level']}</h2>
            <p>Score: {complexity_data['complexity_score']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("📝 Total Lines", complexity_data['total_lines'])
    
    with col3:
        st.metric("🔄 Loops", complexity_data['nested_loops'])
    
    with col4:
        st.metric("🌿 Max Nesting", complexity_data['max_nesting'])
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("❓ Conditionals", complexity_data['conditional_statements'])
    
    with col2:
        st.metric("⚙️ Functions", complexity_data['function_definitions'])
    
    with col3:
        st.metric("🏗️ Classes", complexity_data['class_definitions'])

def parse_and_display_results(review_content, complexity_data=None):
    """Parse and display results in a clean, readable format"""
    
    st.markdown('<div class="review-card">', unsafe_allow_html=True)
    st.markdown("## 📋 Code Review Results")
    
    # Display complexity metrics if available
    if complexity_data:
        display_complexity_metrics(complexity_data)
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 CodeCritic AI</h1>
        <p>Intelligent code analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API key
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google Gemini API key",
            placeholder="Enter your API key..."
        )
        
        if api_key:
            st.success("✅ API Key configured!")
        else:
            st.warning("⚠️ Please enter your Google API key")
            st.markdown("""
            **🔑 How to get API Key:**
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Create a new API key
            3. Copy and paste it here
            """)
        
        st.markdown("---")
        st.markdown("## 🚀 Review Mode")
        use_agent = st.checkbox(
            "🤖 Use AI Agent (Advanced)", 
            value=False,
            help="Uses LangChain agent with multiple specialized tools for deeper analysis"
        )
        
        include_complexity = st.checkbox(
            "📊 Include Complexity Analysis",
            value=True,
            help="Analyze code complexity metrics and provide simplification suggestions"
        )
        
        if use_agent:
            st.info("🤖 **Agent mode**: Comprehensive multi-tool analysis")
        else:
            st.info("⚡ **Simple mode**: Fast single-pass review")
            
        if include_complexity:
            st.info("📊 **Complexity analysis**: Included")
        
        st.markdown("---")
        st.markdown("## 🔧 Supported Languages")
        languages_list = [
            "🐍 Python", "📜 JavaScript", "☕ Java", 
            "⚡ C++", "🔷 C#", "🔥 Go", "🦀 Rust", 
            "🐘 PHP", "💎 Ruby", "📘 TypeScript"
        ]
        for lang in languages_list:
            st.markdown(f"• {lang}")
        
        st.markdown("---")
        st.markdown("## 📈 Features")
        st.markdown("""
        • **Quality Analysis**
                    
        • **Security Scanning**
                    
        • **Performance Review**
                    
        • **Complexity Metrics**
                    
        • **Best Practices**
                    
        • **Documentation Tips**
        """)
    
    # Main content
    if not api_key:
        st.markdown("""
        <div class="section-header">
            <h3>👈 Please configure your Google API key in the sidebar to get started</h3>
            <p>Your API key is needed to power the AI code analysis</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize LLM
    llm = initialize_llm(api_key)
    if not llm:
        st.error("❌ Failed to initialize the AI model. Please check your API key.")
        return
    
    # Code input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="code-section">', unsafe_allow_html=True)
        st.markdown("## 📝 Code Input")
        
        # Language selection
        languages = [
            "python", "javascript", "java", "cpp", "csharp", 
            "go", "rust", "php", "ruby", "typescript", "html", "css", "sql"
        ]
        selected_language = st.selectbox(
            "Select Programming Language", 
            languages, 
            index=0,
            help="Choose the programming language of your code"
        )
        
        # Code input
        code_input = st.text_area(
            "Paste your code here:",
            height=400,
            placeholder="Enter your code here for comprehensive analysis...",
            help="Paste your code and get instant quality, security, and complexity analysis"
        )
        
        # Review button
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            review_clicked = st.button("🔍 Analyze Code", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🧹 Clear Code", use_container_width=True):
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if review_clicked:
            if not code_input.strip():
                st.warning("⚠️ Please enter some code to review!")
                return
            
            # Analyze complexity first
            complexity_data = None
            if include_complexity:
                complexity_data = analyze_complexity(code_input, selected_language)
            
            mode_text = "Using AI Agent" if use_agent else "Quick Analysis"
            with st.spinner(f"🤖 Analyzing your code... ({mode_text})"):
                if use_agent:
                    # Use advanced LangChain agent
                    try:
                        agent = create_advanced_agent(llm)
                        review_result, error = review_with_advanced_agent(agent, code_input, selected_language)
                    except Exception as e:
                        st.error(f"⚠️ Agent failed: {str(e)}. Using simple mode.")
                        review_result, error = review_code(llm, code_input, selected_language)
                else:
                    # Use simple review approach
                    review_result, error = review_code(llm, code_input, selected_language)
                
                if error:
                    st.error(f"❌ Error during review: {error}")
                elif review_result:
                    # Store in history
                    st.session_state.review_history.append({
                        'language': selected_language,
                        'code': code_input[:100] + "..." if len(code_input) > 100 else code_input,
                        'review': review_result,
                        'complexity': complexity_data,
                        'agent_used': use_agent,
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display results
                    success_msg = f"✅ Analysis completed! ({mode_text})"
                    if include_complexity:
                        success_msg += " with complexity metrics"
                    st.success(success_msg)
                    
                    parse_and_display_results(review_result, complexity_data)
                else:
                    st.error("❌ Failed to get review results")
    
    with col2:
        st.markdown("## 📚 Review History")
        
        if st.session_state.review_history:
            for i, history_item in enumerate(reversed(st.session_state.review_history[-5:])):
                mode_indicator = "🤖" if history_item.get('agent_used', False) else "⚡"
                complexity_indicator = "📊" if history_item.get('complexity') else ""
                
                with st.expander(f"{mode_indicator}{complexity_indicator} {history_item['language'].upper()} - Review {len(st.session_state.review_history) - i}"):
                    st.markdown(f"**📅 Time:** {history_item.get('timestamp', 'N/A')}")
                    st.code(history_item['code'], language=history_item['language'])
                    
                    if st.button(f"👁️ View Full Review", key=f"view_{i}"):
                        parse_and_display_results(
                            history_item['review'], 
                            history_item.get('complexity')
                        )
        else:
            st.info("📝 No reviews yet. Submit your first code for analysis!")
        
        # History management
        if st.session_state.review_history:
            col_hist1, col_hist2 = st.columns(2)
            with col_hist1:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.review_history = []
                    st.rerun()
            with col_hist2:
                st.metric("📊 Total Reviews", len(st.session_state.review_history))

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #2a30db 0%, #e91e63 100%); border-radius: 10px; margin: 2rem 0;">
        <p>🚀 Built with ❤️ using <strong>Streamlit</strong>, <strong>LangChain</strong>, and <strong>Google Gemini</strong></p>
        <p>🔧 Enhanced with complexity analysis and modern UI design</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
