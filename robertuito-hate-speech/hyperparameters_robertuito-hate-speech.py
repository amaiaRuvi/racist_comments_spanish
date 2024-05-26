from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from pysentimiento.preprocessing import preprocess_tweet
from utils.preprocessor import preprocess_comment
import fire
import torch
import optuna


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


def find_hyperparameters_robertuito_hate_speech(trial):
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

    epochs = trial.suggest_int("epochs", 3, 10)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, step=0.000005)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3, step=0.01)
    warmup_proportion = trial.suggest_float("warmup_proportion", 0.05, 0.3, step=0.05)
    total_steps = (epochs * len(dataset['train'])) / batch_size
    warmup_steps = int(warmup_proportion * total_steps)

    training_args = TrainingArguments(
        output_dir='./output/optimized-tuned-robertuito-hate-speech',
        logging_dir='./logs/optimized-tuned-robertuito-hate-speech',
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

    # Entrenar el modelo
    trainer.train()
    # Evaluar el modelo
    eval_result = trainer.evaluate()
    # Optuna maximiza, así que devolvemos la métrica negativa si se minimiza
    return eval_result['eval_f1']


def get_best_hyperparameters():
    study = optuna.create_study(direction="maximize")
    study.optimize(find_hyperparameters_robertuito_hate_speech, n_trials=25)

    # Imprimir los mejores hiperparámetros encontrados
    print("Best hyperparameters: ", study.best_params)

    # Evaluar el mejor modelo en el conjunto de prueba
    best_trial = study.best_trial
    best_hyperparameters = best_trial.params
    print(best_hyperparameters)


if __name__ == '__main__':
    fire.Fire(get_best_hyperparameters)
