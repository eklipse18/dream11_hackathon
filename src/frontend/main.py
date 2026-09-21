import gradio as gr
from inference import process_file

with gr.Blocks(
    css="footer {visibility: hidden;}",
) as demo:
    gr.Markdown("## Upload a CSV or Excel file with your teams and match date")
    file = gr.File(label="Upload CSV or Excel", file_types=[".csv", ".xlsx"])
    out = gr.Textbox(label="Output", placeholder="Output will be displayed here...", lines=10)
    process = gr.Button("Process")
    process.click(process_file, inputs=file, outputs=out)

if __name__ == "__main__":
    demo.launch()