from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from pysentimiento.preprocessing import preprocess_tweet
from utils.preprocessor import preprocess_comment
import fire
import torch


def get_metrics(predictions, labels):
    y_true = labels[:, 0]
    y_pred = predictions[:, 0]
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0,
    )

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def compute_metrics(predictions):
    outputs = predictions.predictions
    labels = predictions.label_ids

    binary_predictions = outputs > 0

    return get_metrics(binary_predictions, labels)


def train_evaluate_robertuito_hate_speech():
    print("Loading dataset...")
    database_checkpoint = "amaiaruvi/news_racist_comments_spanish"
    dataset = load_dataset(database_checkpoint)

    print("Loading de model: pysentimiento/robertuito-hate-speech")
    modelo = "pysentimiento/robertuito-hate-speech"
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModelForSequenceClassification.from_pretrained(modelo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    def custom_tokenizer(examples):
        return tokenizer(
            examples["comment"],
            examples["title"],
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

    # Se especifica que para utilizar el modelo "pysentimiento/robertuito-hate-speech" antes hay que preprocesar
    # el texto con su función "preprocess_tweet".
    print("Preprocessing data...")
    preprocessed_data = dataset.map(lambda ex: {
        "comment": preprocess_comment(preprocess_tweet(ex["comment"], lang="es")),
        "title": preprocess_comment(preprocess_tweet(ex["title"], lang="es")),
        "labels": torch.tensor([ex["racist"], 0, 0], dtype=torch.float)
    })

    print("Tokenizing data...")
    encoded_data = preprocessed_data.map(custom_tokenizer, batched=True)
    encoded_data = encoded_data.remove_columns(['link', 'title', 'comment', 'racist'])

    # Aquí cambiaríamos los hiperparámetros
    epochs = 5  # 8
    batch_size = 16  # 8
    learning_rate = 2e-5  # 4.5e-5
    weight_decay = 0.01  # 0.16
    warmup_proportion = 0.1  # 0.2
    total_steps = (epochs * len(dataset['train'])) / batch_size
    warmup_steps = int(warmup_proportion * total_steps)

    training_args = TrainingArguments(
        output_dir='./output/tuned-robertuito-hate-speech',
        logging_dir='./logs/tuned-robertuito-hate-speech',
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        train_dataset=encoded_data['train'],
        eval_dataset=encoded_data['validation'],
        args=training_args,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    )

    # Entrenamiento
    print("Training the model...")
    training_output = trainer.train()
    print("Training is finished.")
    print("Global Step:", training_output.global_step)
    print("Training Loss:", training_output.training_loss)
    print("Metrics:", training_output.metrics)

    # Evaluación del modelo
    trainer.evaluate(encoded_data['test'])

    # Crear el `Trainer` con el conjunto de evaluación y la función para métricas
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=encoded_data["test"],  # Conjunto de evaluación
        compute_metrics=compute_metrics,  # Función para calcular métricas
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    )

    # Evaluar el modelo
    print("Evaluating the model...")
    trainer.evaluate()
    print("Evaluation process is finished.")

    test_predictions = trainer.predict(encoded_data["test"])
    y_true = test_predictions.label_ids[:, 0]
    y_pred = (test_predictions.predictions > 0)[:, 0]
    reporte = classification_report(y_true, y_pred, output_dict=False)
    print(reporte)


if __name__ == '__main__':
    fire.Fire(train_evaluate_robertuito_hate_speech)
