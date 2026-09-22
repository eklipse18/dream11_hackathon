import gradio as gr
from inference import process_match
import pandas as pd

match_df = pd.read_csv('src/frontend/2026_matchlist.csv')
match_list = match_df['match_id'].tolist()

with gr.Blocks(
    css="footer {visibility: hidden;}",
) as demo:
    _match = gr.Dropdown(label="Select Match", choices=[(f"{match_df[match_df['match_id'] == i].iloc[0]['date']} - {match_df[match_df['match_id'] == i].iloc[0]['batting_team']} vs {match_df[match_df['match_id'] == i].iloc[0]['bowling_team']}", i) for i in match_list])
    out = gr.Textbox(label="Output", placeholder="Output will be displayed here...", lines=10)
    process = gr.Button("Process")
    process.click(process_match, inputs=_match, outputs=out)

if __name__ == "__main__":
    demo.launch()