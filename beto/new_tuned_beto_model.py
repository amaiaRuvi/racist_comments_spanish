from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from utils.preprocessor import preprocess_comment
from tokenizers import BertWordPieceTokenizer
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


def tune_new_beto_model():
    print("Loading dataset...")
    database_checkpoint = "amaiaruvi/news_racist_comments_spanish"
    dataset = load_dataset(database_checkpoint)

    print("Loading de model: BETO (dccuchile/bert-base-spanish-wwm-uncased)")
    modelo = "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    # BertForSequenceClassification ya incluye una capa de clasificación para tareas como clasificación de texto. Solo necesitas especificar el número de clases (en este caso, 2).
    model = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Train a new tokenizer with train set, looking for the 'words' that are repeated at least 10 times.")
    new_tokenizer = BertWordPieceTokenizer(lowercase=True)
    texts = [preprocess_comment(ex["comment"]) for ex in dataset['train']]
    new_tokenizer.train_from_iterator(
        texts, min_frequency=10
    )

    old_tokens = set(tokenizer.get_vocab())
    missing_tokens = [tok for tok in new_tokenizer.get_vocab() if tok not in old_tokens]
    print("There are new " + len(missing_tokens) + " tokens that can be added.")

    # Delete non-interesting tokens:
    tokens_to_delete = [
        "##aj",
        "##aja",
        "##ajaj",
        "##ajajajaj",
        "–",
        "—",
        "…"
    ]
    add_tokens = [t for t in missing_tokens if t not in tokens_to_delete and t.isdigit() is False]

    print("Adding tokens to the tokenizer...")
    tokenizer.add_tokens(add_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Now the model must be trained
    print("Preprocessing data...")
    preprocessed_data = dataset.map(lambda ex: {
        "comment": preprocess_comment(ex["comment"]),
        "title": preprocess_comment(ex["title"]),
        "label": ex["racist"]
    })

    def custom_tokenizer(examples):
        return tokenizer(
            examples["comment"],
            examples["title"],
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

    print("Tokenizing data...")
    encoded_data = preprocessed_data.map(custom_tokenizer, batched=True)
    encoded_data = encoded_data.remove_columns(['link', 'title', 'comment', 'racist'])

    # Aquí cambiaríamos los hiperparámetros
    epochs = 8
    batch_size = 8
    learning_rate = 2.5e-5
    weight_decay = 0.3

    warmup_proportion = 0.1
    total_steps = (epochs * len(dataset['train'])) / batch_size
    warmup_steps = int(warmup_proportion * total_steps)

    training_args = TrainingArguments(
        output_dir='/output/new-tuned-BETO',
        logging_dir='./logs/new-tuned-BETO',
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

    print("Evaluating with validation set.")
    trainer.evaluate()

    print("Predictions:")
    test_predictions = trainer.predict(encoded_data["test"])
    y_true = test_predictions.label_ids

    logits = test_predictions.predictions
    # Convertir los logits a un tensor de PyTorch
    logits_tensor = torch.tensor(logits)
    # Aplicar la función softmax a los logits para obtener probabilidades
    probabilities = F.softmax(logits_tensor, dim=1)
    # Obtener las clases predichas (índice de la probabilidad más alta)
    y_pred = torch.argmax(probabilities, dim=1)
    reporte = classification_report(y_true, y_pred, output_dict=False)
    print(reporte)

    new_model_name = "beto-finetuned-racist-news-comments-spanish"
    new_model_path = f"./models/{new_model_name}"
    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)

    model.push_to_hub("amaiaruvi/beto-finetuned-racist-news-comments-spanish")
    tokenizer.push_to_hub("amaiaruvi/beto-finetuned-racist-news-comments-spanish")


if __name__ == '__main__':
    fire.Fire(tune_new_beto_model)
