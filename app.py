from flask import Flask, request, jsonify
from LLM.LLM_Gemini import process_query_pipeline
app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_query():
    data = request.get_json()
    if not data or 'query' not in data or 'django_dataset' not in data:
        return jsonify({'error': 'Both "query" and "django_dataset" are required'}), 400

    query = data['query']
    django_dataset = data['django_dataset']

    try:
        result = process_query_pipeline(query, django_dataset)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
 
