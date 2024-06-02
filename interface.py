import gradio as gr
from models.naive_bayes import *
from models.decision_tree import *
from sklearn.preprocessing import LabelEncoder

def handle_model(model, hours, prev_act, activities, sleep, q_paper):
    if model == 'Naive Bayes':
        values = [hours, prev_act, activities, sleep, q_paper]
        val = run_naive_bayes(values)
        return {pi: gr.Textbox(value=val[0]), metrics: gr.Textbox(value=val[1].strip())}
    elif model == 'Decision Tree':
        values = [hours, prev_act, activities, sleep, q_paper]
        val = run_decision_tree(values)
        return {pi: gr.Textbox(value=val[0]), metrics: gr.Textbox(value=val[1].strip())}

with gr.Blocks(title='Student Performance Index Predictor') as demo:
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(choices=['Naive Bayes', 'Decision Tree'], label='Pick a model')
        with gr.Column():
            with gr.Row():
                hours = gr.Textbox(label='Hours Studied')
                prev_act = gr.Textbox(label='Previous Scores')
                activities = gr.Textbox(label='Extracurricular Activities')
                sleep = gr.Textbox(label='Sleep Hours')
                q_paper = gr.Textbox(label='Sample Question Papers Practiced')

                predict = gr.Button(value='Predict performance index')
        with gr.Column():
            pi =  gr.Textbox(label='Predicted performance index of the student')
            metrics =  gr.Textbox(label='Performance metrics of model')

            predict.click(handle_model, inputs=[model, hours, prev_act, activities, sleep, q_paper], outputs=[pi,metrics])


demo.launch()