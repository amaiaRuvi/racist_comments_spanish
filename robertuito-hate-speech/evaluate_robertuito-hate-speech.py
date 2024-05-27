"""
python.exe -m pip install datasets transformers[torch] torch pysentimiento scikit-learn fire optuna, torch
"""
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
from utils.preprocessor import preprocess_comment
import torch
import fire


def evaluate_robertuito_hate_speech():
    print("Loading dataset...")
    database_checkpoint = "amaiaruvi/news_racist_comments_spanish"
    dataset = load_dataset(database_checkpoint)

    print("Loading de model: pysentimiento/robertuito-hate-speech")
    modelo = "pysentimiento/robertuito-hate-speech"
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModelForSequenceClassification.from_pretrained(modelo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    def custom_tokenizer(text, **kwargs):
        return tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            **kwargs
        )

    # Se especifica que para utilizar el modelo "pysentimiento/robertuito-hate-speech"
    # antes hay que preprocesar el texto con su función "preprocess_tweet".

    # Añadimos la separación entre comentario y título manualmente para poder
    # utilizar el tokenizador original en el pipeline. Pipeline no funciona si le
    # pasamos dos parámetros al tokenizador en volviéndolo en una función tokenize.
    print("Preprocessing data...")
    sep_token = tokenizer.sep_token
    preprocessed_data = dataset.map(lambda ex: {
        "text": sep_token.join([
            preprocess_comment(preprocess_tweet(ex["comment"], lang="es")),
            preprocess_comment(preprocess_tweet(ex["title"], lang="es"))
        ])
    })

    print("Option 1: Using pipeline for inference...")
    pipe = pipeline('text-classification', model=model, tokenizer=custom_tokenizer)
    pipeline_predictions = []
    for i, comment in enumerate(preprocessed_data['test']):
        result = pipe(comment['text'])
        pipeline_predictions.append(result)
    print("Output of the first prediction:", pipeline_predictions[0])

    print("Option 2: Using create_analyzer from pysentimiento for inference...")
    hate_speech_analyzer = create_analyzer(task="hate_speech", lang="es")
    analyzer_predictions = hate_speech_analyzer.predict(preprocessed_data['test']['text'])
    print("Output of the first prediction:", analyzer_predictions[0])

    # Extraer las etiquetas reales y las predicciones
    etiquetas_reales = preprocessed_data['test']['racist']
    etiquetas_predichas = [pred.output for pred in analyzer_predictions]
    etiquetas_predichas_transformadas = [1 if 'hateful' in elemento else 0 for elemento in etiquetas_predichas]

    # Calcular las métricas de rendimiento
    print("Calculating metrics...")
    reporte = classification_report(etiquetas_reales, etiquetas_predichas_transformadas, output_dict=False)
    print(reporte)


if __name__ == '__main__':
    fire.Fire(evaluate_robertuito_hate_speech)
