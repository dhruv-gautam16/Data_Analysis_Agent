import openai
import os
import pandas as pd
import gradio as gr
import logging
import matplotlib.pyplot as plt
from agency_swarm import Agent, Agency
from agency_swarm.tools import BaseTool
from pydantic import Field

logging.basicConfig(filename='app_debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_message(message):
    """Logs messages for debugging."""
    logging.debug(message)
    print(message)

api_key = "" #Insert your api key here
client = openai.OpenAI(api_key=api_key) 

data_df = None

class DataAnalysisTool(BaseTool):
    """ 🧠 Tool to analyze CSV data dynamically using OpenAI GPT-3.5-Turbo """
    query: str = Field(..., description="Natural language query for data analysis.")

    def run(self, data: pd.DataFrame):
        """Uses OpenAI GPT-3.5-Turbo to analyze and visualize data separately."""
        try:
            log_message(f"Running data analysis tool with query: {self.query}")
            log_message(f"Data sample:\n{data.head().to_string()}")

            visualization_keywords = ["plot", "graph", "chart", "visualize", "bar chart", "scatter plot", "histogram"]
            is_visualization = any(keyword in self.query.lower() for keyword in visualization_keywords)

            prompt = f"""
            You are a data scientist. Given the following dataset:
            {data.head().to_string()}

            Perform the following task:
            "{self.query}"

            {"If the query requires a visualization, generate Python code using Matplotlib. Ensure the code saves the figure as 'output.png' using plt.savefig('output.png')." if is_visualization else "Generate Python code that extracts insights from the dataset and assigns the result to a variable named 'result'."}

            Do not create a new dataset inside the code.
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data scientist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.5
            )

            code = response.choices[0].message.content.strip()

            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()

            log_message(f"Generated code:\n{code}")

            execution_globals = {"df": data, "plt": plt, "pd": pd}
            execution_locals = {}

            try:
                exec(code, execution_globals, execution_locals)

                if is_visualization:
                    image_path = "output.png"
                    plt.savefig(image_path)
                    plt.close() 
                    return image_path  

                result = execution_locals.get("result", None)
                return str(result) if result is not None else "❌ No valid results found."
            except Exception as e:
                log_message(f"❌ Execution error: {str(e)}")
                return f"❌ Execution error: {str(e)}"
        except Exception as e:
            log_message(f"❌ Error in DataAnalysisTool: {str(e)}")
            return f"❌ Error: {str(e)}"

data_agent = Agent(
    name="Data Insights Agent",
    description="Analyzes CSV data and generates insights using OpenAI GPT-3.5-Turbo.",
    instructions="Understand user queries and autonomously analyze the dataset using tools.",
    tools=[DataAnalysisTool],
    temperature=0.5,
    max_prompt_tokens=2500,
    model="gpt-3.5-turbo"
)


agency = Agency([data_agent])


def upload_file(file):
    """Loads the uploaded CSV file."""
    global data_df
    try:
        log_message(f"Uploading file: {file.name}")
        data_df = pd.read_csv(file.name)
        log_message(f"File '{file.name}' uploaded successfully!")
        return f"✅ File '{file.name}' uploaded successfully! You can now enter queries."
    except Exception as e:
        log_message(f"❌ Error reading file: {str(e)}")
        return f"❌ Error reading file: {str(e)}"


def process_query(query):
    """Executes user query on the dataset."""
    if data_df is None:
        log_message("⚠️ No file uploaded yet.")
        return "⚠️ Please upload a CSV file first."

    log_message(f"Processing query: {query}")
    tool = DataAnalysisTool(query=query)
    response = tool.run(data=data_df)

    if isinstance(response, str) and response.endswith(".png"):
        return response  

    log_message(f"Query result: {response}")
    return response

with gr.Blocks() as ui:
    gr.Markdown("# 🤖 Data Analysis Agent (Powered by OpenAI GPT-3.5-Turbo)")

    with gr.Row():
        file_input = gr.File(label="📂 Upload CSV File", type="filepath")
        file_output = gr.Textbox(label="📄 File Status", interactive=False)

    query_input = gr.Textbox(label="🔍 Enter your query")
    submit_button = gr.Button("🚀 Run Query")

    result_output_text = gr.Textbox(label="📊 Text Result", interactive=False, visible=False)
    result_output_image = gr.Image(label="📊 Graph Result", visible=False)

    file_input.change(upload_file, inputs=file_input, outputs=file_output)

    def handle_output(query):
        result = process_query(query)
        if isinstance(result, str) and result.endswith(".png"):
            return gr.update(visible=False), gr.update(value=result, visible=True)
        else:
            return gr.update(value=result, visible=True), gr.update(visible=False)

    submit_button.click(handle_output, inputs=query_input, outputs=[result_output_text, result_output_image])

if __name__ == "__main__":
    log_message("Launching Gradio UI...")
    ui.launch(debug=True)
