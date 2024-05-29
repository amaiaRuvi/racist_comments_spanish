from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils.preprocessor import preprocess_comment
import fire
import torch
import torch.nn.functional as F


def evaluate_beto_racist_comments_spanish():
    print("Loading dataset...")
    database_checkpoint = "amaiaruvi/racist_tweets_spanish_rioplatense"
    dataset = load_dataset(database_checkpoint)

    print("Loading de model: amaiaruvi/beto-racist-comments-spanish")
    modelo = "amaiaruvi/beto-racist-comments-spanish"
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

    # Añadimos la separación entre comentario y título manualmente para poder
    # utilizar el tokenizador original en el pipeline. Pipeline no funciona si le
    # pasamos dos parámetros al tokenizador en volviéndolo en una función tokenize.
    print("Preprocessing data...")
    sep_token = tokenizer.sep_token
    preprocessed_data = dataset.map(lambda ex: {
        "text": sep_token.join([
            preprocess_comment(ex["comment"]),
            preprocess_comment(ex["title"])
        ])
    })

    print("Option 1: Using pipeline for inference...")
    pipe = pipeline('text-classification', model=model, tokenizer=custom_tokenizer)
    pipeline_predictions = []
    for i, comment in enumerate(preprocessed_data['test']):
        result = pipe(comment['text'])
        pipeline_predictions.append(result)
    print("Output of the first prediction:", pipeline_predictions[0])

    # Extraer las etiquetas reales y las predicciones
    y_true = preprocessed_data['test']['racist']
    y_pred = [1 if 'LABEL_1' in elemento else 0 for elemento in pipeline_predictions]
    reporte = classification_report(y_true, y_pred, output_dict=False)
    print(reporte)


    print(y_true)
    print(y_pred)


if __name__ == '__main__':
    fire.Fire(evaluate_beto_racist_comments_spanish)
