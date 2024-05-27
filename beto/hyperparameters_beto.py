from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from utils.preprocessor import preprocess_comment
import fire
import torch
import optuna

def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def find_hyperparameters_beto(trial):
    print("Loading dataset...")
    database_checkpoint = "amaiaruvi/news_racist_comments_spanish"
    dataset = load_dataset(database_checkpoint)

    print("Loading de model: BETO (dccuchile/bert-base-spanish-wwm-uncased)")
    modelo = "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    # BertForSequenceClassification ya incluye una capa de clasificación para tareas como clasificación de texto. Solo necesitas especificar el número de clases (en este caso, 2).
    model = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = model.to(device)

    def custom_tokenizer(examples):
        return tokenizer(
            examples["comment"],
            examples["title"],
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )

    print("Preprocessing data...")
    preprocessed_data = dataset.map(lambda ex: {
        "comment": preprocess_comment(ex["comment"]),
        "title": preprocess_comment(ex["title"]),
        "label": ex["racist"]
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
        output_dir='./output/tuned-BETO',
        logging_dir='./logs/tuned-BETO',
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        report_to=[],
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
    print("Training...")
    trainer.train()
    # Evaluar el modelo
    print("Evaluating...")
    eval_result = trainer.evaluate()
    # Optuna maximiza, así que devolvemos la métrica negativa si se minimiza
    return eval_result['eval_f1']


def get_best_hyperparameters():
    study = optuna.create_study(direction="maximize")
    study.optimize(find_hyperparameters_beto, n_trials=25)

    # Imprimir los mejores hiperparámetros encontrados
    print("Best hyperparameters: ", study.best_params)

    # Evaluar el mejor modelo en el conjunto de prueba
    best_trial = study.best_trial
    best_hyperparameters = best_trial.params
    print(best_hyperparameters)


if __name__ == '__main__':
    fire.Fire(get_best_hyperparameters)
