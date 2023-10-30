from transformers import BertTokenizer, TFBertModel
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Flatten, Concatenate, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Étape 1: Initialisation des modèles et des tokenizers
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')

def preprocess_text(text, max_length=128):
    """
    Prétraitement des textes à l'aide de BERT tokenizer.
    """
    tokens = tokenizer.encode_plus(text,
                                   add_special_tokens=True,
                                   max_length=max_length,
                                   return_tensors='tf',
                                   padding='max_length',
                                   truncation=True)["input_ids"]
    return tokens.numpy().squeeze()

df = pd.read_csv("/Users/flavien/Desktop/Df_Encoded_test.csv")
texts_data = np.array([preprocess_text(text) for text in df['description_complete_prétraite'].values], dtype=np.int32)

# Étape 2: Prétraitement des images avec ResNet50
images_data = np.load("/Users/flavien/Desktop/224_X_Train_4D/224_X_Train_4D.npy")
base_model = ResNet50(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False
image_features = base_model.predict(images_data)

# Étape 3: Fusion des modèles
image_input = Input(shape=(7, 7, 2048), name="image_input")
text_input = Input(shape=(128,), dtype=tf.int32, name="text_input")
image_layer = Flatten()(image_input)
text_layer = bert_model(text_input)[1]
combined = Concatenate()([image_layer, text_layer])
dense = Dense(128, activation='relu')(combined)
output = Dense(27, activation='softmax')(dense)

combined_model = Model(inputs=[image_input, text_input], outputs=output)
combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Sauvegarder le modèle fusionné avant l'entraînement
combined_model.save("/Users/flavien/Desktop/Fusion/combined_model_untrained.h5")

# Étape 4: Préparation des étiquettes et séparation des données
label_encoder = LabelEncoder()

def encode_labels(labels):
    """
    Encode les étiquettes avec le LabelEncoder.
    """
    return label_encoder.fit_transform(labels)

integer_encoded = encode_labels(df['prdtypecode'])

def one_hot_encode(encoded_labels, classes_count):
    """
    Convertit les étiquettes encodées en codage one-hot.
    """
    return tf.keras.utils.to_categorical(encoded_labels, num_classes=classes_count)

labels_onehot = one_hot_encode(integer_encoded, 27)
train_texts, val_texts, train_images, val_images, train_labels, val_labels = train_test_split(texts_data, image_features, labels_onehot, test_size=0.2)

# Étape 5: Entraînement avec checkpoints
checkpoint_path = "/Users/flavien/Desktop/Fusion/weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto',
                             save_freq='epoch')

history = combined_model.fit(
    {"image_input": train_images, "text_input": train_texts},
    train_labels,
    validation_data=({"image_input": val_images, "text_input": val_texts}, val_labels),
    epochs=1,
    batch_size=32,
    callbacks=[checkpoint]
)

# Sauvegarder le modèle complet après l'entraînement
combined_model.save("/Users/flavien/Desktop/Fusion/combined_model_trained.h5")