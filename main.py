import os
import docx2txt
import spacy
from flask import Flask, render_template,request,jsonify,send_file
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
import difflib
import PyPDF2
# nltk.download('punkt')
# nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__, static_folder="public", static_url_path="/public")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Function to extract text from different file types
def extract_text(file_path):
    _, file_extension = os.path.splitext(file_path.lower())
    if file_extension == '.txt':
        with open(file_path, 'r') as file:
            text = file.read()
    elif file_extension == '.pdf':
        pdf_file = open(file_path, 'rb')
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(file_path)
        text = ''
        for page in pdf_reader.pages:
          text += page.extract_text()
        pdf_file.close()
    elif file_extension in ('.doc', '.docx'):
        text = docx2txt.process(file_path)
    else:
        raise ValueError("Unsupported file format")
    return text

# Preprocess text data (tokenization, stopwords removal, etc.)
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    return ' '.join(tokens)

# Calculate cosine similarity between two documents
def calculate_similarity(doc):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(doc)
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

def word_similarity(word1, word2):
    # Implement your own word similarity function here
    # For simplicity, let's assume a basic string matching for now
    return 1 if word1 == word2 else 0

def find_common_phrases(text1, text2):
    # Process the texts with SpaCy
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # Create a 2D array to store lengths of LCS
    m = len(doc1)
    n = len(doc2)
    dp = np.zeros((m + 1, n + 1))

    # Build LCS in bottom-up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif word_similarity(str(doc1[i - 1]), str(doc2[j - 1])) > 0.8:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Retrieve the LCS
    lcs_length = int(dp[m][n])
    lcs = []

    i, j = m, n
    while i > 0 and j > 0:
        if word_similarity(str(doc1[i - 1]), str(doc2[j - 1])) > 0.8:
            lcs.insert(0, str(doc1[i - 1]))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # Add "-br-" after each sentence in the LCS
    lcs_with_br = []
    for idx, word in enumerate(lcs):
        lcs_with_br.append(word)
        if idx < len(lcs) - 1 and lcs[idx + 1].endswith('.'):
            lcs_with_br.append("-br-")

    return ' '.join(lcs_with_br)

def extract_variables(node):
    return [target.id for target in node.targets if isinstance(target, ast.Name)]

def compare_variable_names(code1, code2):
    ast1 = ast.parse(code1)
    ast2 = ast.parse(code2)

    variables1 = set()
    variables2 = set()

    # Extract variable names from the first AST
    for node in ast.walk(ast1):
        if isinstance(node, ast.Assign):
            variables1.update(extract_variables(node))

    # Extract variable names from the second AST
    for node in ast.walk(ast2):
        if isinstance(node, ast.Assign):
            variables2.update(extract_variables(node))

    # Calculate similarity based on the number of common variable names
    common_variables = variables1.intersection(variables2)
    
    # Check if codes are similar
    similarity = len(common_variables) / max(len(variables1), len(variables2))

    return similarity, common_variables

def display_variable_diff(code1, code2, common_variables):
    d = difflib.unified_diff(code1.splitlines(), code2.splitlines(), lineterm='', fromfile='code1', tofile='code2')

    # Display only lines related to common variables
    diff_lines = [line for line in d if any(variable in line for variable in common_variables)]
    
    return diff_lines
    # print("\nVariable Differences:")
    # print('\n'.join(diff_lines))

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/text', methods=['GET', 'POST'])
def text():
    if request.method == 'POST':
        threshold = 100 if request.form['threshold'] == "" else int(request.form['threshold'])
        text1 = request.form['textArea1']
        text2 = request.form['textArea2']
        cleaned_text1 = preprocess_text(text1)
        cleaned_text2 = preprocess_text(text2)
        similarity = round((calculate_similarity([cleaned_text1, cleaned_text2]))*100)
        if similarity > threshold:
            sequence = find_common_phrases(text1,text2)
            sequence = sequence.split('-br-')
            sequence = [word for word in sequence if len(word) > 5]
            return jsonify({'similarity' : similarity,'sequence' : sequence,'threshold':threshold,'given':True})

        return jsonify({'similarity' : similarity,'threshold':threshold,'given':False})
    
    else:
        return render_template("home-text.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        data = request.get_json(force=True)
        data_length = len(data)
        results = {}
        for i in range(data_length):
            text = data.get(str(i))
            cleaned_text = preprocess_text(text)
            results[i] = cleaned_text
        doc = list(results.values())
        if(len(list(results.keys())) <= 2):
            similarity = round((calculate_similarity(doc))*100)
            sequence = find_common_phrases(doc[0],doc[1])
            sequence = sequence.split('-br-')
            sequence = [word for word in sequence if len(word) > 5]
            # print('SEQUENCE => ',sequence)
            return jsonify({'similarity' : similarity,'sequence' : sequence})
        else:
            similarity =  calculate_similarity(doc)
            similarity = [round(sim * 100) for sim in similarity]
            return jsonify({'msg':'Successfully uploaded','similarity' : similarity,})

    return render_template("home-upload.html")

@app.route('/meaning', methods=['GET', 'POST'])
def meaning():
    if request.method == 'POST':
        text1 = request.form['textArea1']
        text2 = request.form['textArea2']
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        similarity = doc1.similarity(doc2)
        is_similar = similarity >= 0.8
        return jsonify({'similarity':similarity, 'is_similar':is_similar})
    return render_template("home-meaning.html")

@app.route('/code', methods=['GET', 'POST'])
def code():
    if request.method == 'POST':
        code1 = request.form['textArea1']
        code2 = request.form['textArea2']
        similarity, common_variables = compare_variable_names(code1, code2)
        if similarity >= 0.5:
            diff_line = display_variable_diff(code1, code2, common_variables)
            for i in range(len(diff_line)):
                if diff_line[i][0] != '+' and diff_line[i][0] != '-':
                    m_str = '*'+ diff_line[i][1:]
                    diff_line[i] = m_str
            return jsonify({'is_similar' : True,'similar_line' : diff_line})
        else:
            return jsonify({'is_similar' : False})
        
    return render_template("home-code.html")

if __name__ == "__main__":
    app.run(debug=True)