import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.schema import AIMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from deep_translator import GoogleTranslator
import re
import ast
import requests
from bs4 import BeautifulSoup
import time
app = Flask(__name__)
#CORS(app)
#CORS(app, resources={r"/*": {"origins": "*"}})  # Ensure all routes allow CORS
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configuration
app.config['VECTORSTORE'] = './vector_index'
os.makedirs(app.config['VECTORSTORE'], exist_ok=True)

# Initialize embeddings and model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
model = ChatGroq(
    groq_api_key="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", 
    model="llama-3.3-70b-versatile", 
    temperature=0.5
)

@app.route('/')
def index():
    
    return "GitLens Repository Summarization Service is running."
@app.route('/generate_commit_message', methods=['POST'])
def generate_commit_message():
    data = request.get_json()
    if not data or "diff" not in data:
        return jsonify({"error": "No diff data provided."}), 400

    git_diff = data["diff"]

    prompt = f"""
You are an expert Git assistant.

Based on the following Git diff, write a clear, concise, and conventional commit message. 
Structure the message in imperative tone (e.g., "Fix bug", "Add feature"), and include a short subject and optionally a body if needed.

Git Diff:
{git_diff}

css
Copy
Edit
Return ONLY the commit message, no explanations or formatting.
"""

    try:
        response = model.invoke(prompt)
        if isinstance(response, AIMessage):
            response = response.content
        return jsonify({"commit_message": response.strip()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided."}), 400

    repo_name = data.get('name', 'Unknown Repository')
    repo_description = data.get('description', 'No description available.')
    full_text = data.get('fullText', '')

    print("\n✅ Received Data from Frontend:")
    print(f"Repo Name: {repo_name}")
    print(f"Description: {repo_description}")
    print(f"Full Page Text (first 500 chars):\n{full_text[:500]}...\n")

    # Construct the prompt with all extracted text
    prompt = f"""
Repository Name: {repo_name}
Description: {repo_description}

Full Page Content:
{full_text}

Based on the above details, generate a detailed and accurate summary of the repository's purpose, structure, key functionalities, dependencies, and potential use cases.
Return the analysis in HTML format where:
- Main section headings use <h2> tags with bold styling.
- Subheadings use <h3> tags.
- Key points are wrapped in <strong> tags.
- The overall design is clean and visually appealing.

"""

    try:
        summary = model.invoke(prompt)
        if isinstance(summary, AIMessage):
            summary = summary.content
        return jsonify({"summary": summary}), 200
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500

@app.route('/generate_comments', methods=['POST'])
def generate_comments():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided."}), 400

    repo_name = data.get('name', 'Unknown Repository')
    repo_description = data.get('description', 'No description available.')
    full_text = data.get('fullText', '')
    prompt = f"""
    Repository Name: {repo_name}
    Description: {repo_description}
    Full Page Content:
    {full_text}

    Generate inline code comments explaining each function and its purpose.
    """

    try:
        summary = model.invoke(prompt)
        return jsonify({"output": summary.content}), 200
    except Exception as e:
        return jsonify({"error": f"Error generating comments: {str(e)}"}), 500

@app.route('/detect_ai_code', methods=['POST'])
def detect_ai_code():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided."}), 400

    repo_name = data.get('name', 'Unknown Repository')
    repo_description = data.get('description', 'No description available.')
    full_text = data.get('fullText', '')

    prompt = f"""
    Repository Name: {repo_name}
    Description: {repo_description}

    Full Page Content:
    {full_text}

    Analyze the provided repository content and determine what percentage is likely AI-generated vs. human-written code.
    Provide an estimate in the format: AI Code: X% | Human Code: Y%
    """

    try:
        response = model.invoke(prompt)
        result = response.content
        ai_percentage, human_percentage = extract_percentages(result)

        return jsonify({
            "ai_percentage": ai_percentage,
            "human_percentage": human_percentage
        }), 200
    except Exception as e:
        logging.error(f"Error detecting AI code: {e}")
        return jsonify({"error": f"Error detecting AI code: {str(e)}"}), 500
def translate_text(text, target_lang):
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return text  # Return original text on error

def extract_percentages(text):
    import re
    ai_match = re.search(r'AI Code:\s*(\d+)%', text)
    human_match = re.search(r'Human Code:\s*(\d+)%', text)
    
    ai_percentage = int(ai_match.group(1)) if ai_match else 50
    human_percentage = int(human_match.group(1)) if human_match else 50
    return ai_percentage, human_percentage
'''@app.route('/analyze_dependencies', methods=['POST', 'OPTIONS'])
def analyze_dependencies():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight successful"}), 200

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided."}), 400

    repo_name = data.get('name', 'Unknown Repository')
    repo_description = data.get('description', 'No description available.')
    full_text = data.get('fullText', '')

    print("\n✅ Received Data from Frontend for Dependency Analysis:")
    print(f"Repo Name: {repo_name}")
    print(f"Description: {repo_description}")

    # Construct the refined prompt for LLM
    prompt = f"""
    Analyze the given Python code and extract:
    - A list of function names and the functions they call.
    - A list of class names and their dependencies (parent classes, used functions).

    Code:
    ```python
    {full_text}
    ```

    Return the output as a structured JSON object, without any extra explanations:

    {{
        "functions": {{
            "function_name_1": ["called_function_1", "called_function_2"],
            "function_name_2": ["called_function_3"]
        }},
        "classes": {{
            "ClassName1": ["ParentClass1", "used_function_1"],
            "ClassName2": []
        }}
    }}
    """

    try:
        response = model.invoke(prompt)
        result = response.content

        return jsonify({"dependencies": result}), 200
    except Exception as e:
        logging.error(f"Error analyzing dependencies: {e}")
        return jsonify({"error": f"Error analyzing dependencies: {str(e)}"}), 500

'''
@app.route('/analyze_dependencies', methods=['POST', 'OPTIONS'])
def analyze_dependencies():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight successful"}), 200

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided."}), 400

    repo_name = data.get('name', 'Unknown Repository')
    repo_description = data.get('description', 'No description available.')
    full_text = data.get('fullText', '')

    print("\n✅ Received Data from Frontend for Dependency Analysis:")
    print(f"Repo Name: {repo_name}")
    print(f"Description: {repo_description}")

    # Construct the refined prompt for LLM
    prompt = f"""
    Analyze the given Python code and extract:
    - A list of function names along with the functions they call.
    - A list of class names along with their dependencies (parent classes, used functions).

    Code:
    ```python
    {full_text}
    ```

    Present the results **clearly** in this format:

    ----
    **Functions and Their Dependencies**  
    function_name_1 → called_function_1, called_function_2  
    function_name_2 → called_function_3  

    ----
    **Classes and Their Dependencies**  
    ClassName1 (inherits: ParentClass1) → uses: used_function_1  
    ClassName2 (inherits: None) → uses: None  
    ----
    """
    try:
        response = model.invoke(prompt)
        result = response.content

        return jsonify({"dependencies": result}), 200
    except Exception as e:
        logging.error(f"Error analyzing dependencies: {e}")
        return jsonify({"error": f"Error analyzing dependencies: {str(e)}"}), 500
def perform_static_analysis(code_text):
    """
    Perform static analysis using Python's AST to extract functions and classes.
    Returns a summary of the detected functions and classes.
    """
    try:
        tree = ast.parse(code_text)
    except Exception as e:
        return "Error parsing code for static analysis: " + str(e)
    
    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    
    analysis = (
        f"Functions defined: {', '.join(functions) if functions else 'None'}. "
        f"Classes defined: {', '.join(classes) if classes else 'None'}."
    )
    return analysis
@app.route('/generate_pr_details', methods=['POST'])
def generate_pr_details():
    """
    Expects JSON:
    {
      "diff": "...full diff string..."
    }
    Returns:
    {
      "title": "AI-generated PR title",
      "description": "AI-generated PR description"
    }
    """
    data = request.get_json()
    diff = data.get('diff', '')

    if not diff:
        return jsonify({"error": "No diff provided"}), 400

    # Use your language model to generate title + description
    prompt = f"""
You're an assistant that summarizes GitHub Pull Request diffs.

Diff:
{diff}

Analyze this diff and generate a meaningful:
1. PR Title — very short, clear, concise.
2. PR Description — structured, markdown-friendly, and informative.

Format:
---
Title: <one-line summary>
Description:
<bulleted or paragraph markdown-friendly summary>
---
Return only the title and description clearly labeled.
    """

    try:
        result = model.invoke(prompt)
        if isinstance(result, AIMessage):
            result = result.content

        # Basic parsing from LLM output
        title = ""
        description = ""

        lines = result.splitlines()
        for i, line in enumerate(lines):
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
            if line.lower().startswith("description:"):
                description = "\n".join(lines[i + 1:]).strip()
                break

        if not title:
            title = "Update based on recent changes"
        if not description:
            description = "This PR includes changes based on the recent code diff."

        return jsonify({
            "title": title,
            "description": description
        })

    except Exception as e:
        logging.error(f"🔥 Error in generate_pr_details: {e}")
        return jsonify({"error": "Failed to generate PR summary"}), 500


@app.route("/analyze_code", methods=["POST"])
def analyze_code():
    """
    Expects JSON with:
      - code: The source code to analyze
      - language: (Optional) target language code (default 'en')
    Performs static analysis on the code and generates a natural language summary.
    """
    data = request.get_json()
    if not data or "code" not in data:
        return jsonify({"error": "No code provided."}), 400

    code_text = data["code"]
    selected_lang = data.get("language", "en")

    # Perform static analysis using the AST
    analysis_insights = perform_static_analysis(code_text)
    print("Static Analysis Insights:", analysis_insights)

    prompt = f"""
The following code has been analyzed for structure:

Static Analysis Insights:
{analysis_insights}

Full Code:
{code_text}

Generate a clear and easy-to-understand natural language summary of the entire code in a detailed manner. Describe the code, functions, classes, and their relationships, along with their use cases, in a structured and comprehensive way. Ensure the explanation covers how different components interact and contribute to the overall functionality of the program.Return the analysis in HTML format where:
- Main section headings use <h2> tags with bold styling.
- Subheadings use <h3> tags.
- Key points are wrapped in <strong> tags.
 - Bullet points should appear on **separate lines** using `<ul><li>` format.
  - Ensure **proper line breaks** and **visual spacing** between sections.
  - Use **paragraphs `<p>`** to improve readability.
  - Do not compress all bullet points into one paragraph; instead, place each point clearly and separately.
  - Make the output **clean and visually appealing** for display in a modal or rich text viewer.
    """
    try:
        summary = model.invoke(prompt)
        if isinstance(summary, AIMessage):
            summary = summary.content

        if selected_lang != "en":
            summary = translate_text(summary, target_lang=selected_lang)
        return jsonify({"summary": summary}), 200
    except Exception as e:
        logging.error(f"Error generating code analysis summary: {e}")
        return jsonify({"error": f"Error generating code analysis summary: {str(e)}"}), 500


def clean_pr_number(raw_number):
    match = re.search(r'\d+', raw_number)  # Extract only digits
    return match.group() if match else raw_number  # Return cleaned number
def get_pr_data(repo_url, max_prs=5):
    if not repo_url.endswith("/pulls"):
        repo_url += "/pulls"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(repo_url, headers=headers, timeout=10)
        if response.status_code != 200:
            print("Failed to fetch PRs:", response.status_code)
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        pr_rows = soup.select("div.js-issue-row")[:max_prs]

        if not pr_rows:
            print("No PRs found on the page. Check the repository URL.")
            return None

        pr_data_list = []
        for pr in pr_rows:
            title_elem = pr.select_one("a.Link--primary")
            author_elem = pr.select_one("a.Link--muted")
            pr_number_elem = pr.select_one("span.opened-by")  

            if not title_elem or not author_elem or not pr_number_elem:
                continue  

            title = title_elem.text.strip()
            author = author_elem.text.strip()
            pr_number = pr_number_elem.text.strip().split("#")[-1]  # Extract PR number
            pr_link = "https://github.com" + title_elem["href"]
            state = "Open" if "open" in pr.attrs.get("aria-label", "").lower() else "Closed"

            print(f"Fetching PR Details from: {pr_link}")
            files_changed_tab = pr_link + "/files"
            files_response = requests.get(files_changed_tab, headers=headers, timeout=10)
            if files_response.status_code != 200:
                continue

            files_soup = BeautifulSoup(files_response.text, "html.parser")

            file_changes = []
            total_changes = 0

            for file_div in files_soup.select(".file"):
                file_name = file_div.select_one(".file-info a").text.strip() if file_div.select_one(".file-info a") else "Unknown"
                added_lines, removed_lines = [], []

                for line in file_div.select(".blob-code"):
                    if "blob-code-addition" in line["class"]:
                        added_lines.append(line.get_text(strip=True))
                    elif "blob-code-deletion" in line["class"]:
                        removed_lines.append(line.get_text(strip=True))

                total_changes += len(added_lines) + len(removed_lines)  # Count changes

                file_changes.append({
                    "File": file_name,
                    "Added Lines": added_lines,
                    "Removed Lines": removed_lines
                })

            pr_response = requests.get(pr_link, headers=headers, timeout=10)
            if pr_response.status_code != 200:
                continue

            pr_soup = BeautifulSoup(pr_response.text, "html.parser")
            status_elem = pr_soup.select_one(".State")
            if status_elem:
                status_class=status_elem.get("class", [])
                print("Status Class:", status_class)
                if "State--closed" in status_class:
                    merge_status = "Closed" 
                elif "State--open" in status_class:
                    merge_status = "Open"   
                else:
                    merge_status = "Unknown"

            pr_data_list.append({
                "Number": clean_pr_number(pr_number),  
                "Title": title,
                "Author": author,
                "State": merge_status,
                "Created At": pr.select_one("relative-time").text.strip(),
                "Total Changes": total_changes,
                "Files Modified": file_changes
            })
            time.sleep(0.0001)
        return pr_data_list

    except requests.exceptions.RequestException as e:
        print("Error fetching PRs:", str(e))
        return None
@app.route("/get_latest_pr_data", methods=["GET"])
def get_latest_pr_data_route():
    repo_url = request.args.get("repo_url")
    max_prs = request.args.get("max_prs", default=5, type=int)

    if not repo_url:
        return jsonify({"error": "Missing repo_url parameter"}), 400

    pr_data = get_pr_data(repo_url, max_prs)
    
    if pr_data is None:
        print("❌ Failed to fetch PR data")
        return jsonify({"error": "Failed to fetch PR data"}), 500

    print("✅ PR Data:", pr_data)
    return jsonify(pr_data)
@app.route("/pr_analysis", methods=["POST"])
def pr_analysis():
    print("Received request for PR analysis")
    data = request.get_json()
    #if not data:
     #   return jsonify({"error": "No data provided."}), 400
    
    repo_url = data.get("repo_url")
    pr_number = data.get("pr_number")
    
    if not repo_url or not pr_number:
        return jsonify({"error": "Missing repo_url or pr_number"}), 400
    print("Repo URL:", repo_url)
    print("PR Number:", pr_number)
    pr_data = get_pr_data(repo_url, 5)  
    pr_info = next((pr for pr in pr_data if pr.get("Number") == pr_number),None)
    print("Selected PR Info:", pr_info)
    if not pr_info:
        print("Error: PR not found")
        return jsonify({"error": "PR not found"}), 404
    
    prompt = f"""
    Pull Request Analysis:
    Title: {pr_info['Title']}
    Author: {pr_info['Author']}
    Created At: {pr_info['Created At']}
    Files Modified:{pr_info['Files Modified']}
    Total Changes: {pr_info['Total Changes']}
    Merge Status: {pr_info['State']}

    
    Analyze these changes:
    - How do they affect the functionality, structure, and performance?
    - Highlight improvements, potential issues, and security concerns.
    - Suggest best practices and optimizations.
    give the analysis in HTML Format  . 
    """
    summary = model.invoke(prompt)
    if isinstance(summary, AIMessage):
            summary = summary.content
    print("PR Analysis Summary:", summary)
    return jsonify({"summary": summary}), 200



if __name__ == '__main__':
    app.run(debug=True)

