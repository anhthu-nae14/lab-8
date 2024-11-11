from flask import Flask, render_template_string
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

app = Flask(__name__)

def execute_notebook_and_extract_results(notebook_path):
    # Load and execute the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_data = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(notebook_data, {'metadata': {'path': './'}})

    # Extract relevant results based on output structure
    results = {
        "name": "Phan Anh Thư",
        "student_id": "2274802010872",
        "evaluation_title": "KNN Model Evaluation",
        "confusion_matrix": "",
        "metrics": {},
        "features": "",
        "labels": "",
        "additional_metrics": {}
    }

    for cell in notebook_data.cells:
        if cell.cell_type == 'code' and 'outputs' in cell:
            for output in cell.outputs:
                if output.output_type == 'stream' or output.output_type == 'execute_result':
                    text = output.get('text', '') or output.data.get('text/plain', '')
                    
                    # Detect and organize sections based on keywords in outputs
                    if "Confusion Matrix:" in text:
                        results["confusion_matrix"] = text.strip()
                    elif "Accuracy:" in text and "Recall:" in text and "Precision:" in text:
                        results["metrics"]["summary"] = text.strip()
                    elif "Feature 1" in text and "Feature 2" in text:
                        results["features"] = text.strip()
                    elif "Balanced Accuracy:" in text:
                        results["additional_metrics"]["balanced_accuracy"] = text.strip()
                    elif "Matthews Correlation Coefficient" in text:
                        results["additional_metrics"]["mcc"] = text.strip()
                    elif "Fowlkes-Mallows Index" in text:
                        results["additional_metrics"]["fmi"] = text.strip()
                    elif "Bias:" in text:
                        results["additional_metrics"]["bias"] = text.strip()

    return results

@app.route('/')
def show_results():
    notebook_path = 'Confusion Matrix.ipynb'  # Replace with your notebook's path
    results = execute_notebook_and_extract_results(notebook_path)
    
    # Define the HTML template with the additional metrics
    html_template = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kết quả tính toán</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f8f9fa; display: flex; justify-content: center; }
            .container { max-width: 600px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); text-align: center; }
            h1 { color: #007bff; }
            h2 { color: #343a40; margin-top: 20px; }
            p { color: #495057; }
            .section-title { font-size: 1.2em; font-weight: bold; color: #343a40; margin-top: 15px; }
            .result-item { margin: 5px 0; font-size: 1em; color: #495057; }
            pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; text-align: left; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{{ evaluation_title }}</h1>
            <p><strong>Họ tên:</strong> {{ name }}</p>
            <p><strong>MSSV:</strong> {{ student_id }}</p>
            
            <h2>Confusion Matrix</h2>
            <pre>{{ confusion_matrix }}</pre>
            
            <h2>Metrics</h2>
            <p class="result-item">{{ metrics.summary }}</p>
            
            <h2>Feature Data</h2>
            <pre>{{ features }}</pre>

            
            <h2>Additional Metrics</h2>
            {% for key, value in additional_metrics.items() %}
                <p class="result-item"><strong>{{ key.replace('_', ' ').capitalize() }}:</strong> {{ value }}</p>
            {% endfor %}
        </div>
    </body>
    </html>
    """

    # Render the HTML template with the extracted results
    return render_template_string(html_template, **results)

if __name__ == '__main__':
    app.run(debug=True)
