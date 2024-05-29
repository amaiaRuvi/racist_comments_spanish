from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from utils.preprocessor import preprocess_comment
import fire
import torch
import torch.nn.functional as F


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

def evaluate_beto_racist_comments_spanish():
    print("Loading dataset...")
    database_checkpoint = "amaiaruvi/news_racist_comments_spanish"
    dataset = load_dataset(database_checkpoint)

    print("Loading de model: amaiaruvi/beto-racist-comments-spanish")
    modelo = "amaiaruvi/beto-racist-comments-spanish"
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

    # Aquí cambiaríamos los hiperparámetros
    epochs = 8
    batch_size = 16
    learning_rate = 2.5e-5
    weight_decay = 0.3

    warmup_proportion = 0.1
    total_steps = (epochs * len(dataset['train'])) / batch_size
    warmup_steps = int(warmup_proportion * total_steps)

    training_args = TrainingArguments(
        output_dir='./results',
        logging_dir='./logs',
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        eval_accumulation_steps=1,
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

    print("Training the model...")
    trainer.train()

    print("Loading dataset...")
    rioplatense_database_checkpoint = "amaiaruvi/racist_tweets_spanish_rioplatense"
    rioplatense_dataset = load_dataset(database_checkpoint)

    print("Preprocessing data...")
    preprocessed_data_rioplatense = rioplatense_dataset.map(lambda ex: {
        "comment": preprocess_comment(ex["comment"]),
        "title": preprocess_comment(ex["title"]),
        "label": ex["racist"]
    })

    print("Tokenizing data...")
    encoded_data_rioplatense = preprocessed_data_rioplatense.map(custom_tokenizer, batched=True)
    encoded_data_rioplatense = encoded_data_rioplatense.remove_columns(['link', 'title', 'comment', 'racist'])

    print("Predictions:")
    test_predictions_rioplatense = trainer.predict(encoded_data_rioplatense["test"])
    y_true = test_predictions_rioplatense.label_ids

    logits = test_predictions_rioplatense.predictions
    # Convertir los logits a un tensor de PyTorch
    logits_tensor = torch.tensor(logits)
    # Aplicar la función softmax a los logits para obtener probabilidades
    probabilities = F.softmax(logits_tensor, dim=1)
    # Obtener las clases predichas (índice de la probabilidad más alta)
    y_pred = torch.argmax(probabilities, dim=1)
    reporte = classification_report(y_true, y_pred, output_dict=False)
    print(reporte)

if __name__ == '__main__':
    fire.Fire(evaluate_beto_racist_comments_spanish)
